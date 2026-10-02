# -*- coding: utf-8 -*-
"""lme_probe6.py — 一锤定音: 话题型寒暄vs功能型寒暄×金 的方向(两库对照)
假说: 两库的话题型寒暄(含问题词的短句)都>金 → 规律完全通用,表面差异=寒暄构成比"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
COURT = re.compile(r"\b(hi|hey|hello|thanks|thank you|great|awesome|cool|nice|sure|okay|ok|wow|sounds good|good to know)\b", re.I)

t0 = time.time()
# ===== LME =====
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sess, sid2h = {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    sid2h[sid] = h
    if h not in sess:
        sess[h] = turns
hashes = list(sess.keys())
RAWS_L, SIDS_L = [], []
for si, h in enumerate(hashes):
    sid = "lme-s" + h[:12]
    RAWS_L.append("[hdr]")
    SIDS_L.append(sid)
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS_L.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
        SIDS_L.append(sid)
D_L = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X_L = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
SID2ROWS = {}
for i, s in enumerate(SIDS_L):
    SID2ROWS.setdefault(s, []).append(i)

# ===== LoCoMo =====
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
RAWS_C = [rec_raw(m) for m in MID]
RAWN_C = [norm(x) for x in RAWS_C]
D_C = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q_C = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q_C[q["qa_id"]] = q
IDS_C = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
X_C = np.load(HERE + "/xz_cache.npz")["X"]
IDX_C = {q: i for i, q in enumerate(IDS_C)}

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN_C[i] for k in keys))

P("loaded %.0fs" % (time.time() - t0))
rng = np.random.RandomState(0)

def classify_and_cos(r, qstems, dvec, qvec):
    words = r.split()
    isc = bool(COURT.search(r)) or len(words) <= 6
    if not isc:
        return "content", float(dvec @ qvec)
    rsts = set(stem(w) for w in re.findall(r"[a-z']+", r.lower()) if len(w) > 2)
    k = "topic-chat" if (rsts & qstems) else "func-chat"
    return k, float(dvec @ qvec)

# LME: 金会话内的三类
resL = {"topic-chat": [], "func-chat": [], "content": []}
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    gs = np.array(sorted(gs))
    if len(gs) < 10:
        continue
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    pick = rng.choice(gs, size=min(20, len(gs)), replace=False)
    for i in pick:
        k, c = classify_and_cos(RAWS_L[i], qstems, D_L[i], X_L[qi])
        resL[k].append(c)

# LoCoMo: 金会话(金所在conv)内的三类
resC = {"topic-chat": [], "func-chat": [], "content": [], "gold": []}
for qa in IDS_C:
    g = gold_set(qa)
    if not g:
        continue
    qi = IDX_C[qa]
    conv = "loco-" + qa.split("#")[0]
    rows = [i for i in range(len(MID)) if MID[i].startswith(conv) or re.sub(r"_(rbak\d+k|m\d+)$", "", MID[i]) == conv]
    rows = [i for i in rows if "_rbak" not in MID[i]]
    if len(rows) < 10:
        continue
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", (Q_C[qa].get("question") or "").lower()) if w not in QSTOP and len(w) > 2)
    pick = rng.choice(rows, size=min(20, len(rows)), replace=False)
    for i in pick:
        if i in g:
            resC["gold"].append(float(D_C[i] @ X_C[qi]))
        else:
            k, c = classify_and_cos(RAWS_C[i], qstems, D_C[i], X_C[qi])
            resC[k].append(c)

P("\n===== 分类寒暄的方向测试(金会话/金会话所在对话内) =====")
P("LME(金会话内, n: %s):" % {k: len(v) for k, v in resL.items()})
for k in ("topic-chat", "func-chat", "content"):
    if resL[k]:
        P("  %-11s cos=%.3f" % (k, np.mean(resL[k])))
P("LoCoMo(金所在对话内, n: %s):" % {k: len(v) for k, v in resC.items()})
for k in ("gold", "topic-chat", "func-chat", "content"):
    if resC[k]:
        P("  %-11s cos=%.3f" % (k, np.mean(resC[k])))
P("\n判定: 两库的topic-chat若都>金/content → 规律完全通用(表面差异=寒暄构成比)")
P("done %.0fs" % (time.time() - t0))
