# -*- coding: utf-8 -*-
"""r20v_runner.py - 双段v2 + 条件日期域:
查询含日期模式时, 第二段附加日期向量域检索(独立配额并入, 不挤原句域);
查询无日期时与 v2(r20s) 逐位一致。"""
import json, os, sys, time, urllib.request, re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = "C:/locomo_refined/memsys"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
PREV = HERE + "/subset100b.json"
GLM_URL = "https://api.z.ai/api/anthropic/v1/messages"
GLM_KEY = os.environ.get("GLM_KEY", "")
NL = chr(10)

os.environ["MEM_FILE"] = HERE + "/mem.jsonl"
sys.path.insert(0, HERE)
import sessionmem

n_all = n_raw = 0
RAW_BY_SID = defaultdict(list)
SUM_BY_SID = defaultdict(list)
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    n_all += 1
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") == "raw":
        n_raw += 1
        RAW_BY_SID[r.get("session_id")].append(r)
    else:
        SUM_BY_SID[r.get("session_id")].append(r)
DATE_BY_SID = defaultdict(list)
for l in open(HERE + "/raw_date_index.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        DATE_BY_SID[d["sid"]].append(d)
print("version=%s lib=%d raw=%d date_vec=%d" %
      (sessionmem.SESSIONMEM_VERSION, n_all, n_raw, sum(len(v) for v in DATE_BY_SID.values())), flush=True)
assert sessionmem.SESSIONMEM_VERSION.startswith("r22") and n_raw > 5000, "GATE1 FAIL"

SESS_NUM, SESS_DATE = {}, {}
cur_n, cur_d = None, ""
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session (\d+) — ([^\]]+)\]", (r.get("raw") or "").strip())
    if m:
        cur_n, cur_d = int(m.group(1)), m.group(2).strip()
    SESS_NUM[r.get("memory_id")] = cur_n
    SESS_DATE[r.get("memory_id")] = cur_d

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

DATE_PAT = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[\s,]*\d{0,2},?\s*\d{4}|\b(19|20)\d{2}\b", re.I)
def query_has_date(q):
    return bool(DATE_PAT.search(q))

ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
ANCHOR_V = None
def is_negative_semantic(ans):
    global ANCHOR_V
    if not (ans or "").strip():
        return True
    if ANCHOR_V is None:
        ANCHOR_V = l2n(np.array(sessionmem.embed(ANCHORS), dtype=np.float32))
    av = sessionmem.embed([ans[:200]])[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

def glm(prompt, max_tokens=400):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request(GLM_URL, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": GLM_KEY, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            t = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

def build_rows(recs, cap=250):
    parts = []
    for r in recs:
        raw = (r.get("raw") or "").strip()
        if raw.startswith("[Session"):
            continue
        n = SESS_NUM.get(r.get("memory_id"))
        d = SESS_DATE.get(r.get("memory_id")) or ""
        parts.append("[Session %s — %s] %s" % (n if n is not None else "?", d, raw[:cap]))
    return NL.join(parts) if parts else "(无记忆命中)"

def raw_search3(query, hyp, ans_, sid, k):
    """v2 原样: 三路(原句向量域)混排。"""
    sub = RAW_BY_SID.get(sid) or []
    if not sub:
        return []
    M = l2n(np.array([r["vector"] for r in sub], dtype=np.float32))
    pool = {}
    for t, w in ((query, 1.0), (hyp, 0.95), (ans_, 0.95)):
        if not t:
            continue
        qv = sessionmem.embed([t])[0]
        for j in np.argsort(-(M @ qv))[:k]:
            idx = int(j)
            sc = float((M[idx] @ qv)) * w
            mid = sub[idx].get("memory_id")
            if mid not in pool or sc > pool[mid][0]:
                pool[mid] = (sc, sub[idx])
    ranked = sorted(pool.values(), key=lambda x: -x[0])[:k]
    return [r for _, r in ranked]

def date_boost(query, sid, k=5):
    """条件日期域: 独立配额, 只在查询含日期时激活。"""
    dsub = DATE_BY_SID.get(sid) or []
    if not dsub:
        return []
    Md = l2n(np.array([d["vector"] for d in dsub], dtype=np.float32))
    qv = sessionmem.embed([query])[0]
    out = []
    for j in np.argsort(-(Md @ qv))[:k]:
        rec = next((r for r in RAW_BY_SID.get(sid, []) if r.get("memory_id") == dsub[int(j)]["mid"]), None)
        if rec:
            out.append(rec)
    return out

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

def work(q):
    ci = q["conversation_idx"]
    item = data[ci]
    sid = "loco-" + str(item.get("sample_id", ci))
    has_date = query_has_date(q["question"])
    try:
        res = sessionmem.search_hyde(q["question"], top_n=8, current_session=sid,
                                     edge=True, mems=SUM_BY_SID.get(sid) or [])
        recs1 = res["records"]
        hyde_meta = res["meta"].get("hyde") or {}
        raw_q = raw_search3(q["question"], None, None, sid, k=5)
        seen, merged = set(), []
        for r in recs1 + raw_q:
            kk = r.get("raw_of") or r.get("memory_id")
            if kk in seen:
                continue
            seen.add(kk)
            merged.append(r)
        ctx1 = build_rows(merged)
    except Exception as e:
        print("RETR_FAIL", q["qa_id"], repr(e)[:120], flush=True)
        ctx1 = "(retrieval error)"
        hyde_meta = {}
    p1 = ("根据记忆回答。若无关回答「不知道」。" + NL +
          "记忆(每条已标注所属会话与日期):" + NL + ctx1 + NL + NL +
          "问题: " + q["question"] + NL + TIME_RULE)
    ans1 = glm(p1)
    final, second = ans1, False
    if is_negative_semantic(ans1):
        try:
            recs2 = raw_search3(q["question"], hyde_meta.get("hyp"), hyde_meta.get("ans"), sid, k=25)
            if has_date:                      # 条件日期域: 独立配额并入, 不挤原句域
                seen2 = set(r.get("raw_of") or r.get("memory_id") for r in recs2)
                for r in date_boost(q["question"], sid, k=5):
                    kk = r.get("raw_of") or r.get("memory_id")
                    if kk not in seen2:
                        recs2.append(r)
                        seen2.add(kk)
            if recs2:
                ctx2 = build_rows(recs2)
                p2 = ("根据以下原始对话记录回答问题。" + NL +
                      "记录(已标注会话与日期):" + NL + ctx2 + NL + NL +
                      "问题: " + q["question"] + NL + TIME_RULE +
                      " 确实没有答案才回答「不知道」。")
                ans2 = glm(p2, max_tokens=300)
                if not is_negative_semantic(ans2):
                    final, second = ans2, True
        except Exception as e:
            print("PASS2_FAIL", q["qa_id"], repr(e)[:100], flush=True)
    return dict(qa_id=q["qa_id"], predicted_answer=final, second_pass=second)

data = json.load(open(RAW, encoding="utf-8"))
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(PREV, encoding="utf-8"))
todo = [q for q in questions if q["qa_id"] in set(r["qa_id"] for r in prev)]
n_dq = sum(1 for q in todo if query_has_date(q["question"]))
print("questions:", len(todo), "with-date:", n_dq, flush=True)

assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出")
assert query_has_date("Who did Maria have dinner with on May 3, 2023?")
assert not query_has_date("What are Jolene's favorite books?")
print("date-gate self-test ok", flush=True)

t0 = time.time()
with ThreadPoolExecutor(6) as ex:
    submission = list(ex.map(work, todo))
n_second = sum(1 for s in submission if s.get("second_pass"))
print("SUBMISSION_DONE", len(submission), "second_pass_used=", n_second,
      round(time.time() - t0, 1), "s", flush=True)
with open(HERE + "/submission100v.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions100v.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN", flush=True)
