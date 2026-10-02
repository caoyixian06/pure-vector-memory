# -*- coding: utf-8 -*-
"""lme_probe5.py — 因果链数据裁决: ①两库问题-金词重叠(出题假说) ②两库结构同构性
③LME金会话内寒暄vs内容句的问题cos(同会话反向是否存在——用户"库同构"假说的关键测试)"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
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
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
SID2ROWS_L = {}
for i, s in enumerate(SIDS_L):
    SID2ROWS_L.setdefault(s, []).append(i)
P("LME loaded %.0fs" % (time.time() - t0))

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
CONV_C = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR_C = {}
seen = set()
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR_C[i] = 1
    seen.add(k)
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(len(MID)):
        if any(k in RAWN_C[i] for k in keys):
            g.add(i)
    return g
X_C = np.load(HERE + "/xz_cache.npz")["X"]
IDX_C = {q: i for i, q in enumerate(IDS_C)}
P("LoCoMo loaded %.0fs" % (time.time() - t0))

# ===== ① 问题-金词重叠(出题假说): 问题词干与金词干的重叠率 =====
rng = np.random.RandomState(0)
def qgold_overlap(qtext, gold_texts):
    qst = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    if not qst or not gold_texts:
        return None
    gst = set()
    for g in gold_texts:
        gst |= set(stem(w) for w in re.findall(r"[a-z']+", g.lower()) if len(w) > 2)
    return len(qst & gst) / len(qst)

ovl_L, ovl_C = [], []
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS_L.get("lme-s" + sid2h[s][:12], []))
    gtxt = [RAWS_L[i] for i in rng.choice(sorted(gs), size=min(5, len(gs)), replace=False)] if gs else []
    o = qgold_overlap(q["question"], gtxt)
    if o is not None:
        ovl_L.append(o)
for qa in IDS_C:
    g = gold_set(qa)
    if not g:
        continue
    gi = rng.choice(sorted(g), size=min(5, len(g)), replace=False).tolist()
    o = qgold_overlap(Q_C[qa].get("question") or "", [RAWS_C[i] for i in gi])
    if o is not None:
        ovl_C.append(o)
P("\n① 问题-金词干重叠率(出题假说: 复述式出题→高重叠)")
P("   LoCoMo=%.3f (n=%d)   LME=%.3f (n=%d)" % (np.mean(ovl_C), len(ovl_C), np.mean(ovl_L), len(ovl_L)))

# ===== ② 库结构同构性: 记录长度/会话自相似/寒暄密度 =====
def conv_self_sim(RAWS, KEYS):
    sims = []
    convs = {}
    for i, k in enumerate(KEYS):
        convs.setdefault(k, []).append(i)
    cl = list(convs.values())
    rng2 = np.random.RandomState(1)
    for rows in rng2.choice(len(cl), size=min(300, len(cl)), replace=False):
        r = cl[rows]
        if len(r) < 10:
            continue
        pick = rng2.choice(r, size=8, replace=False)
        tk = [toks(RAWS[i]) for i in pick]
        ss = []
        for a2 in range(8):
            for b2 in range(a2 + 1, 8):
                ss.append(len(tk[a2] & tk[b2]) / max(1, len(tk[a2] | tk[b2])))
        sims.append(np.mean(ss))
    return np.mean(sims)
lenL = np.mean([len(r.split()) for r in RAWS_L])
lenC = np.mean([len(r.split()) for r in RAWS_C])
courtL = np.mean([1.0 if COURT.search(r) else 0.0 for r in RAWS_L])
courtC = np.mean([1.0 if COURT.search(r) else 0.0 for r in RAWS_C])
P("\n② 库结构同构性")
P("   记录词数: LoCoMo=%.1f LME=%.1f" % (lenC, lenL))
P("   寒暄句密度: LoCoMo=%.3f LME=%.3f" % (courtC, courtL))
P("   会话自相似(8采样Jaccard): LoCoMo=%.3f LME=%.3f" % (
    conv_self_sim(RAWS_C, CONV_C), conv_self_sim(RAWS_L, SIDS_L)))

# ===== ③ LME同会话反向测试: 金会话内 寒暄句vs内容句 的问题cos =====
chatty_cos, content_cos = [], []
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS_L.get("lme-s" + sid2h[s][:12], []))
    gs = np.array(sorted(gs))
    if len(gs) < 10:
        continue
    pick = rng.choice(gs, size=min(15, len(gs)), replace=False)
    for i in pick:
        r = RAWS_L[i]
        isc = bool(COURT.search(r)) or len(r.split()) <= 6
        c = float(D[i] @ X[qi])
        (chatty_cos if isc else content_cos).append(c)
P("\n③ LME金会话内: 问题-寒暄句cos=%.3f(n=%d) vs 问题-内容句cos=%.3f(n=%d)" % (
    np.mean(chatty_cos), len(chatty_cos), np.mean(content_cos), len(content_cos)))
P("   (LoCoMo已知: 同会话寒暄0.668 > 金0.612 = 反向)")
P("done %.0fs" % (time.time() - t0))
