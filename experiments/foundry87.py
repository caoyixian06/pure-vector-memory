# -*- coding: utf-8 -*-
"""foundry87.py — 词对词软配对(用户六设计): 问题词×候选词的词向量语义配对分数, 头部判别区分度"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
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

t0 = time.time()
zw = np.load(HERE + "/word_vecs.npz")
P("word_vecs keys: %s" % str(list(zw.keys())))
WORDS = None
for k in ("words", "W", "vocab"):
    if k in zw:
        WORDS = [str(x) for x in zw[k]]
        break
if WORDS is None:
    WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
    WORDS = list(WH.keys())
WV = l2n(zw["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
P("词表: %d词  V=%s %.0fs" % (len(WORDS), str(WV.shape), time.time() - t0))
hitq = 0
totq = 0

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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = np.load(HERE + "/xz_cache.npz")["X"]
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v10.json", encoding="utf-8"))
MID2I = {m: i for i, m in enumerate(MID)}

def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR = {}
seen = {}
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR[i] = seen[k]; PAIR[seen[k]] = i
    else:
        seen[k] = i

def gold_groups(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN[i])
        if not hits:
            continue
        for i in list(hits):
            tt = PAIR.get(i)
            if tt is not None:
                hits.add(tt)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups

# 每题: 问题词向量集合; 软配对分(对候选) = mean_q max_{c in cand} cos(q,c)
def qword_vecs(qa):
    global hitq, totq
    ws = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    vecs = []
    for w in ws:
        totq += 1
        if w in W2I:
            hitq += 1
            vecs.append(WV[W2I[w]])
    return np.array(vecs, dtype=np.float32) if vecs else None

def cand_words_vecs(i):
    ws = set(stem(w) for w in re.findall(r"[a-z']+", RAW[i].lower()) if len(w) > 2)
    idxs = [W2I[w] for w in ws if w in W2I]
    return WV[idxs] if idxs else None

def soft_align(qv, cv):
    # qv (nq,256) cv(nc,256): mean over q of max over c
    if qv is None or cv is None or len(cv) == 0:
        return None
    return float((qv @ cv.T).max(axis=1).mean())

pos_sa, neg_sa, pos_dq, neg_dq = [], [], [], []
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    goldset = set()
    for g in grps:
        goldset |= g
    qv = qword_vecs(qa)
    qi = IDS.index(qa)
    for r, i in enumerate(seq[:15]):
        cv = cand_words_vecs(i)
        sa = soft_align(qv, cv)
        if sa is None:
            continue
        dqc = float(D[i] @ X[qi])
        if i in goldset:
            if r >= 5:
                pos_sa.append(sa); pos_dq.append(dqc)
        else:
            if r < 5:
                neg_sa.append(sa); neg_dq.append(dqc)

P("问题词命中率: %d/%d = %.1f%%" % (hitq, totq, 100.0 * hitq / max(1, totq)))
P("正样本(6-15名金) n=%d  负样本(前5噪) n=%d" % (len(pos_sa), len(neg_sa)))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

P("软配对分: 金=%.3f 噪=%.3f AUC=%.3f" % (np.mean(pos_sa), np.mean(neg_sa), auc(pos_sa, neg_sa)))
P("对照dqc:  金=%.3f 噪=%.3f AUC=%.3f" % (np.mean(pos_dq), np.mean(neg_dq), auc(pos_dq, neg_dq)))
P("done %.0fs" % (time.time() - t0))
