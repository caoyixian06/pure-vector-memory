# -*- coding: utf-8 -*-
"""r23a_runner.py - A臂: 双段v2底盘 + 假说缓存 + 槽位规则(用户设计)。
槽位: 日期规范化键(YYYY-MM[-DD], 记录日期来自Session头) + 人名槽(speaker_a/b表)。
第二段排序: score = 0.7*cos + 0.3*slot。"""
import json, os, sys, time, urllib.request, re, hashlib
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
print("version=%s lib=%d raw=%d" % (sessionmem.SESSIONMEM_VERSION, n_all, n_raw), flush=True)
assert sessionmem.SESSIONMEM_VERSION.startswith("r22") and n_raw > 5000, "GATE1 FAIL"

SESS_NUM, SESS_DATE = {}, {}
cur_n, cur_d = None, ""
MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
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
        cur_n = int(m.group(1))
        cur_d = m.group(2).strip()
    SESS_NUM[r.get("memory_id")] = cur_n
    SESS_DATE[r.get("memory_id")] = cur_d

def norm_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(1).lower() in MONTHS:
        return "%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()])
    return ""

# 会话人名表(从 questions 的 speaker_a/b + 库内 raw 前缀兜底)
SPEAKERS = defaultdict(set)
_qtmp = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
for q in _qtmp:
    for k in ("speaker_a", "speaker_b"):
        if q.get(k):
            SPEAKERS[q["conversation_idx"]].add(q[k])
for sid, recs in RAW_BY_SID.items():
    for r in recs[:50]:
        m = re.match(r"([A-Z][a-z]+):", (r.get("raw") or ""))
        if m:
            SPEAKERS[-1].add(m.group(1))
ALL_NAMES = set().union(*SPEAKERS.values()) if SPEAKERS else set()

def q_date_keys(q):
    keys = set()
    for m in re.finditer(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(1).lower()], int(m.group(2))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", q, re.I):
        keys.add("%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()]))
    return keys

def q_names(q):
    return {n for n in ALL_NAMES if re.search(r"\b%s\b" % re.escape(n), q)}

def slot_score(qkeys, qns, rec):
    d = SESS_DATE.get(rec.get("memory_id")) or ""
    rd = norm_date(d)
    ds = 0.0
    if rd:
        for k in qkeys:
            if k == rd:
                ds = 1.0
                break
            if len(k) == 7 and rd.startswith(k):
                ds = max(ds, 0.5)
    ns = 0.0
    raw = rec.get("raw") or ""
    if qns:
        hit = sum(1 for n in qns if re.search(r"\b%s\b" % re.escape(n), raw))
        ns = min(1.0, hit / max(1, len(qns)))
    return 0.6 * ds + 0.4 * ns

# ---- 假说缓存(跨轮固定检索输入) ----
HYDE_CACHE = HERE + "/hyde_cache.jsonl"
_hcache = {}
if os.path.exists(HYDE_CACHE):
    for l in open(HYDE_CACHE, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            _hcache[d["qh"]] = d
def cached_hyp(query):
    qh = hashlib.md5(query.encode()).hexdigest()[:16]
    if qh in _hcache:
        return _hcache[qh].get("hyp") or "", _hcache[qh].get("ans") or ""
    h = sessionmem.hyde_hypotheses(query) or {}
    rec = dict(qh=qh, query=query, hyp=h.get("hyp") or "", ans=h.get("ans") or "")
    _hcache[qh] = rec
    with open(HYDE_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + NL)
    return rec["hyp"], rec["ans"]

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

def raw_search_slot(query, hyp, ans_, sid, k, ci=None):
    """A臂: 三路向量 + 槽位线性组合 0.7cos+0.3slot。"""
    sub = RAW_BY_SID.get(sid) or []
    if not sub:
        return []
    M = l2n(np.array([r["vector"] for r in sub], dtype=np.float32))
    qkeys, qns = q_date_keys(query), q_names(query)
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
    scored = []
    for mid, (sc, rec) in pool.items():
        final = 0.7 * (sc + 1.0) / 2.0 + 0.3 * slot_score(qkeys, qns, rec)
        scored.append((final, rec))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

data = json.load(open(RAW, encoding="utf-8"))
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(PREV, encoding="utf-8"))
todo = [q for q in questions if q["qa_id"] in set(r["qa_id"] for r in prev)]
print("questions:", len(todo), flush=True)

assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出")
assert q_date_keys("on May 3, 2023") == {"2023-05-03"}
assert "Maria" in q_names("Who did Maria have dinner with")
print("slot self-test ok", flush=True)

def work(q):
    ci = q["conversation_idx"]
    sid = "loco-" + str(data[ci].get("sample_id", ci))
    try:
        res = sessionmem.search_hyde(q["question"], top_n=8, current_session=sid,
                                     edge=True, mems=SUM_BY_SID.get(sid) or [])
        recs1 = res["records"]
        hyp, ans_h = cached_hyp(q["question"])
        raw_q = raw_search_slot(q["question"], None, None, sid, k=5, ci=ci)
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
        hyp, ans_h = "", ""
    p1 = ("根据记忆回答。若无关回答「不知道」。" + NL +
          "记忆(每条已标注所属会话与日期):" + NL + ctx1 + NL + NL +
          "问题: " + q["question"] + NL + TIME_RULE)
    ans1 = glm(p1)
    final, second = ans1, False
    if is_negative_semantic(ans1):
        try:
            recs2 = raw_search_slot(q["question"], hyp, ans_h, sid, k=25, ci=ci)
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

t0 = time.time()
with ThreadPoolExecutor(6) as ex:
    submission = list(ex.map(work, todo))
n_second = sum(1 for s in submission if s.get("second_pass"))
print("SUBMISSION_DONE", len(submission), "second_pass_used=", n_second,
      round(time.time() - t0, 1), "s", flush=True)
with open(HERE + "/submission23a.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions23a.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN", flush=True)
