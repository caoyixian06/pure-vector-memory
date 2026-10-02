# -*- coding: utf-8 -*-
"""r24bge_runner.py - BGE-M3 域首测(A臂底盘移植):
检索换 BGE dense(1024)+sparse 混合分: score = 0.6*cos_dense + 0.4*sparse内积(归一后)
第一段: BGE 混合分检索 summary 域 top8(带 A2 会话先验)
第二段: 触发(语义判据)后 BGE 混合分检索 raw 域 top25 + 槽位规则(A臂 0.6日期+0.4人名)
答题 GLM anthropic 端点; 假说缓存复用。判分用新判官(anthropic 通道)。"""
import io
import json, os, sys, time, urllib.request, re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
KEY = "${GLM_KEY}"

HERE = "C:/locomo_refined/memsys"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
PREV = HERE + "/subset100b.json"
NL = chr(10)

sys.path.insert(0, HERE)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("bge-m3 loaded", flush=True)

# ---- 库加载(BGE 域) ----
D = np.load(HERE + "/mem_bge_dense.npz")["dense"]
SP = []
MID = []
KIND = []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    SP.append(r["sparse"])
    MID.append(r["mid"])
    KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    REC[r.get("memory_id")] = r
print("lib:", D.shape, "summary:", KIND.count("summary"), "raw:", KIND.count("raw"), flush=True)

# sparse 词表索引: token_id -> [(lib_idx, w)]
from collections import defaultdict as dd
SPINV = dd(list)
for i, sp in enumerate(SP):
    for t, w in sp.items():
        SPINV[t].append((i, w))

def bge_encode(texts):
    out = model.encode(texts, return_dense=True, return_sparse=True)
    return out["dense_vecs"], out["lexical_weights"]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def hybrid_scores(qtext):
    dv, lw = bge_encode([qtext])
    qd = l2n(np.asarray(dv, dtype=np.float32))[0]
    cos = D @ qd
    qs = {str(a): float(b) for a, b in lw[0].items()}
    sps = np.zeros(len(SP), dtype=np.float32)
    for t, w in qs.items():
        for i, w2 in SPINV.get(t, ()):
            sps[i] += w * w2
    nrm = float(np.linalg.norm(qs.values())) if False else (sum(v * v for v in qs.values()) ** 0.5) or 1.0
    sps = sps / nrm
    return 0.6 * cos + 0.4 * sps, cos, sps

def search_bge(qtext, kind, sid, top_n, a2=0.05):
    """BGE 混合分检索指定域(kind), A2 会话先验加分。"""
    hs, cos, sps = hybrid_scores(qtext)
    idxs = [i for i, k in enumerate(KIND) if k == kind]
    sub_cos, sub_sps, sub_ids = cos[idxs], sps[idxs], idxs
    sid_of = [REC[MID[i]].get("session_id") for i in idxs]
    sc = 0.6 * sub_cos + 0.4 * sub_sps
    for j, s in enumerate(sid_of):
        if s == sid:
            sc[j] += a2
    order = np.argsort(-sc)[:top_n]
    return [REC[MID[sub_ids[o]]] for o in order], [float(sc[o]) for o in order]

# ---- 槽位规则(A臂) ----
MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
SESS_DATE = {}
cur_d = ""
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session \d+ — (.+?)\]", (r.get("raw") or "").strip())
    if m:
        cur_d = m.group(1).strip()
    SESS_DATE[r.get("memory_id")] = cur_d
SESS_NUM = {}
cur_n = None
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session (\d+)", (r.get("raw") or "").strip())
    if m:
        cur_n = int(m.group(1))
    SESS_NUM[r.get("memory_id")] = cur_n

def norm_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(1).lower() in MONTHS:
        return "%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()])
    return ""

def q_date_keys(q):
    keys = set()
    for m in re.finditer(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(1).lower()], int(m.group(2))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", q, re.I):
        keys.add("%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()]))
    return keys

_qtmp = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
ALL_NAMES = set()
for q in _qtmp:
    for k in ("speaker_a", "speaker_b"):
        if q.get(k):
            ALL_NAMES.add(q[k])

def q_names(q):
    return {n for n in ALL_NAMES if re.search(r"\b%s\b" % re.escape(n), q)}

def slot_score(qkeys, qns, rec):
    rd = norm_date(SESS_DATE.get(rec.get("memory_id")) or "")
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

# ---- 语义触发判据(BGE 域锚句) ----
ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
_ade, _alw = bge_encode(ANCHORS)
ANCHOR_V = l2n(np.asarray(_ade, dtype=np.float32))
def is_negative_semantic(ans):
    if not (ans or "").strip():
        return True
    de, _ = bge_encode([ans[:200]])
    av = l2n(np.asarray(de, dtype=np.float32))[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

def glm(prompt, max_tokens=400):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
                                         json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": KEY, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            t = "".join(b.get("text", "") for b in d.get("content", [])
                        if b.get("type") == "text").strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

# ---- 假说缓存(兼容: 旧缓存存文本, BGE 域直接编码文本) ----
HYDE_CACHE = HERE + "/hyde_cache.jsonl"
_hcache = {}
if os.path.exists(HYDE_CACHE):
    for l in open(HYDE_CACHE, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            _hcache[d["qh"]] = d
import hashlib
def cached_hyp(query):
    qh = hashlib.md5(query.encode()).hexdigest()[:16]
    if qh in _hcache:
        return _hcache[qh].get("hyp") or "", _hcache[qh].get("ans") or ""
    return "", ""

def build_rows(recs, scores=None, cap=250):
    parts = []
    for r in recs:
        raw = (r.get("raw") or "").strip()
        if raw.startswith("[Session"):
            continue
        n = SESS_NUM.get(r.get("memory_id"))
        d = SESS_DATE.get(r.get("memory_id")) or ""
        parts.append("[Session %s — %s] %s" % (n if n is not None else "?", d, raw[:cap]))
    return NL.join(parts) if parts else "(无记忆命中)"

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

data = json.load(open(RAW, encoding="utf-8"))
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(PREV, encoding="utf-8"))
todo = [q for q in questions if q["qa_id"] in set(r["qa_id"] for r in prev)]
print("questions:", len(todo), flush=True)

# 冒烟: 检索链路 + 语义判据
assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出,湖上日出作品很著名")
q0 = todo[0]
sid0 = "loco-" + str(data[q0["conversation_idx"]].get("sample_id", q0["conversation_idx"]))
r1, _ = search_bge(q0["question"], "summary", sid0, 8)
r2, _ = search_bge(q0["question"], "raw", sid0, 25)
assert len(r1) >= 5 and len(r2) >= 15, "GATE FAIL: bge search"
print("smoke ok: sum=%d raw=%d" % (len(r1), len(r2)), flush=True)

import threading
_ELOCK = threading.Lock()

def work(q):
    ci = q["conversation_idx"]
    sid = "loco-" + str(data[ci].get("sample_id", ci))
    qkeys, qns = q_date_keys(q["question"]), q_names(q["question"])
    try:
        with _ELOCK:
            recs1, sc1 = search_bge(q["question"], "summary", sid, 8)
        seen = set(r.get("raw_of") or r.get("memory_id") for r in recs1)
        hyp, ans_h = cached_hyp(q["question"])
        merged = list(recs1)
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
            with _ELOCK:
                recs2, sc2 = search_bge(q["question"], "raw", sid, 25)
            if hyp:
                with _ELOCK:
                    rh, _ = search_bge(hyp, "raw", sid, 10)
                seen2 = set(r.get("raw_of") or r.get("memory_id") for r in recs2)
                for r in rh:
                    k = r.get("raw_of") or r.get("memory_id")
                    if k not in seen2:
                        recs2.append(r)
                        seen2.add(k)
            # 槽位规则(A臂): 混合分 0.7 + 槽位 0.3
            scored = []
            for r in recs2:
                base = 0.5
                scored.append((0.7 * base + 0.3 * slot_score(qkeys, qns, r), r))
            scored.sort(key=lambda x: -x[0])
            top = [r for _, r in scored[:25]]
            if top:
                ctx2 = build_rows(top)
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
with ThreadPoolExecutor(4) as ex:
    submission = list(ex.map(work, todo))
print("SUBMISSION_DONE", len(submission), "second_pass=", sum(1 for s in submission if s.get("second_pass")),
      round(time.time() - t0, 1), "s", flush=True)
with open(HERE + "/submission24bge.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions24bge.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN", flush=True)
