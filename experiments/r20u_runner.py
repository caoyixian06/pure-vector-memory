# -*- coding: utf-8 -*-
"""r20u_runner.py - 双段v2 + 日期向量:
在76分v2(r20s)基础上仅改一处: 第二段原句检索从单域(原句向量)改为三路混排
(query/hyp/ans × 原句向量+带日期向量), 其他逐位同v2。"""
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
    if not l.strip():
        continue
    d = json.loads(l)
    DATE_BY_SID[d["sid"]].append(d)
print("version=%s lib=%d raw=%d date_vec=%d" %
      (sessionmem.SESSIONMEM_VERSION, n_all, n_raw, sum(len(v) for v in DATE_BY_SID.values())), flush=True)
assert sessionmem.SESSIONMEM_VERSION.startswith("r22") and n_raw > 5000, "GATE1 FAIL"
assert sum(len(v) for v in DATE_BY_SID.values()) > 5000, "GATE2 FAIL: date index missing"

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
    """三路文本 × 双域向量(原句+带日期)混排。"""
    sub = RAW_BY_SID.get(sid) or []
    dsub = DATE_BY_SID.get(sid) or []
    if not sub:
        return []
    M = l2n(np.array([r["vector"] for r in sub], dtype=np.float32))
    Md = l2n(np.array([d["vector"] for d in dsub], dtype=np.float32)) if dsub else None
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
        if Md is not None:
            for j in np.argsort(-(Md @ qv))[:k]:
                idx = int(j)
                sc = float((Md[idx] @ qv)) * w
                mid = dsub[idx].get("mid")
                if mid not in pool or sc > pool[mid][0]:
                    rec = next((r for r in sub if r.get("memory_id") == mid), None)
                    if rec:
                        pool[mid] = (sc, rec)
    ranked = sorted(pool.values(), key=lambda x: -x[0])[:k]
    return [r for _, r in ranked]

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

def work(q):
    ci = q["conversation_idx"]
    item = data[ci]
    sid = "loco-" + str(item.get("sample_id", ci))
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
print("questions:", len(todo), flush=True)

assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出,湖上日出作品很著名")
q0 = todo[0]
sid0 = "loco-" + str(data[q0["conversation_idx"]].get("sample_id", q0["conversation_idx"]))
_r1 = sessionmem.search_hyde(q0["question"], top_n=8, current_session=sid0, edge=True,
                             mems=SUM_BY_SID.get(sid0) or [])
_hm = _r1["meta"].get("hyde") or {}
_r2 = raw_search3(q0["question"], _hm.get("hyp"), _hm.get("ans"), sid0, k=25)
assert _r1["records"] and len(_r2) >= 10, "GATE4 FAIL"
print("smoke ok: pass1=%d pass2=%d" % (len(_r1["records"]), len(_r2)), flush=True)

t0 = time.time()
with ThreadPoolExecutor(6) as ex:
    submission = list(ex.map(work, todo))
n_second = sum(1 for s in submission if s.get("second_pass"))
print("SUBMISSION_DONE", len(submission), "second_pass_used=", n_second,
      round(time.time() - t0, 1), "s", flush=True)
with open(HERE + "/submission100u.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions100u.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN", flush=True)
