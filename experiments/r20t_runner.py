# -*- coding: utf-8 -*-
"""r20t_runner.py - 双段v3 + CQR:
第一段: v4形态 + CQR查询改写(search_cqr dual); 不动其他。
第二段(v2基础上): ①上下文按Session分组+组头日期(治会话内错误日期锚定)
②假说三路真混排(hyde meta的hyp/ans参与raw_search3) ③枚举指令(多条事实列全)。
触发: 语义判据(同v2)。"""
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
print("version=%s raw_index=%s lib=%d raw_kind=%d" %
      (sessionmem.SESSIONMEM_VERSION, sessionmem.RAW_INDEX, n_all, n_raw), flush=True)
assert sessionmem.SESSIONMEM_VERSION.startswith("r22") and sessionmem.RAW_INDEX, "GATE1 FAIL"
assert n_raw > 5000, "GATE2 FAIL"

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

def build_rows_grouped(recs, cap=250):
    """第二段分组呈现: 按Session分组, 组头带日期, 组内按对话序(治错误日期锚定)。"""
    def _ord(r):
        m = re.search(r"_m(\d+)", r.get("memory_id") or "")
        return int(m.group(1)) if m else 10**9
    groups = defaultdict(list)
    seen_txt = set()
    for r in recs:
        raw = (r.get("raw") or "").strip()
        if raw.startswith("[Session") or raw in seen_txt:
            continue
        seen_txt.add(raw)
        groups[SESS_NUM.get(r.get("memory_id"))].append(
            dict(raw=raw[:cap], d=SESS_DATE.get(r.get("memory_id")) or "", order=_ord(r)))
    parts = []
    for n in sorted(groups, key=lambda x: (x is None, x)):
        g = sorted(groups[n], key=lambda i: i["order"])
        d = next((i["d"] for i in g if i["d"]), "")
        parts.append("==== Session %s(%s) ====" % (n if n is not None else "?", d))
        for i in g:
            parts.append(i["raw"])
    return NL.join(parts) if parts else "(无记忆命中)"

def raw_search3(query, hyp, ans_, sid, k):
    sub = RAW_BY_SID.get(sid) or []
    if not sub:
        return []
    M = l2n(np.array([r["vector"] for r in sub], dtype=np.float32))
    pool = {}
    for t, w in ((query, 1.0), (hyp, 1.0), (ans_, 1.0)):
        if not t:
            continue
        qv = sessionmem.embed([t])[0]
        sims = M @ qv
        for j in np.argsort(-sims)[:k]:
            idx = int(j)
            sc = float(sims[idx]) * w
            mid = sub[idx].get("memory_id")
            if mid not in pool or sc > pool[mid][0]:
                pool[mid] = (sc, sub[idx])
    ranked = sorted(pool.values(), key=lambda x: -x[0])[:k]
    return [r for _, r in ranked]

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

def work(q):
    ci = q["conversation_idx"]
    item = data[ci]
    sid = "loco-" + str(item.get("sample_id", ci))
    try:
        res = sessionmem.search_cqr(q["question"], top_n=8, current_session=sid,
                                    edge=True, mems=SUM_BY_SID.get(sid) or [],
                                    cqr_mode="dual")
        recs1 = res["records"]
        try:
            hyde_meta = sessionmem.hyde_hypotheses(q["question"]) or {}
        except Exception:
            hyde_meta = {}
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
        print("RETR_FAIL", q["qa_id"], repr(e)[:140], flush=True)
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
                ctx2 = build_rows_grouped(recs2)
                p2 = ("根据以下原始对话记录回答问题。记录已按会话分组,组头标注该场会话的日期——"
                      "注意区分不同场次的日期,不要把某一场的日期当成唯一日期。" + NL +
                      "记录:" + NL + ctx2 + NL + NL +
                      "问题: " + q["question"] + NL + TIME_RULE +
                      " 若答案包含多条事实,把记录中能找到的全部列出。确实没有答案才回答「不知道」。")
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
print("semantic trigger self-test ok", flush=True)

q0 = todo[0]
sid0 = "loco-" + str(data[q0["conversation_idx"]].get("sample_id", q0["conversation_idx"]))
try:
    _r1 = sessionmem.search_cqr(q0["question"], top_n=8, current_session=sid0, edge=True,
                                mems=SUM_BY_SID.get(sid0) or [], cqr_mode="dual")
    _recs1 = _r1["records"]
    _hm = sessionmem.hyde_hypotheses(q0["question"]) or {}
except Exception as e:
    print("CQR_SMOKE_FAIL:", repr(e)[:160], flush=True)
    raise SystemExit(3)
_r2 = raw_search3(q0["question"], _hm.get("hyp"), _hm.get("ans"), sid0, k=25)
assert _recs1 and len(_r2) >= 10, "GATE4 FAIL"
print("smoke ok: pass1=%d pass2_pool=%d hyp=%s" %
      (len(_recs1), len(_r2), "yes" if _hm.get("hyp") else "no"), flush=True)

t0 = time.time()
with ThreadPoolExecutor(6) as ex:
    submission = list(ex.map(work, todo))
n_second = sum(1 for s in submission if s.get("second_pass"))
print("SUBMISSION_DONE", len(submission), "second_pass_used=", n_second,
      round(time.time() - t0, 1), "s", flush=True)
with open(HERE + "/submission100t.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions100t.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN", flush=True)
