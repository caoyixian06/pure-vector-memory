# -*- coding: utf-8 -*-
"""foundry132b.py — F132后续: 排除同会话混杂 + 会话聚合视角
1) 金金内聚 vs 金-硬噪(同会话) vs 金-硬噪(跨会话): 0.18结构差是真信号还是会话效应
2) 同样256维向量, 聚合到会话层(max): 金会话能否胜出(margin/hit@1)
"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry132b_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s); LOG.write(s + "\n"); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        d = json.loads(l)
        REC[d.get("memory_id")] = d
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
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
uidx, uinv = np.unique(CONVKEY, return_inverse=True)
NS = len(uidx)
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
Q256 = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

cand = [k for k, qa in enumerate(IDS) if gold_set(qa)]
rng = np.random.RandomState(20260919)
pick = rng.choice(len(cand), min(200, len(cand)), replace=False)
SEL = [cand[i] for i in pick]
GSEL = [gold_set(IDS[k]) for k in SEL]
P("loaded NR=%d NS=%dconv, selected %d questions" % (NR, NS, len(SEL)))

gg_l, gh_same_l, gh_cross_l = [], [], []
sess_hit1, sess_margin = [], []
same_frac = []
for t, k_i in enumerate(SEL):
    G = GSEL[t]
    gidx = np.array(sorted(G), dtype=np.int64)
    qv = Q256[k_i]
    cq = QW @ qv
    gmask = np.zeros(NR, dtype=bool)
    gmask[gidx] = True
    nn_idx = np.where(~gmask)[0]
    hard50 = nn_idx[np.argsort(-cq[nn_idx])[:50]]
    gold_convs = set(int(uinv[g]) for g in gidx)
    hsame = np.array([int(uinv[h]) in gold_convs for h in hard50])
    same_frac.append(float(hsame.mean()))
    if len(gidx) >= 2:
        GM = QW[gidx] @ QW[gidx].T
        iu = np.triu_indices(len(gidx), 1)
        gg_l.append(float(GM[iu].mean()))
        if hsame.any():
            gh_same_l.append(float((QW[gidx] @ QW[hard50[hsame]].T).mean()))
        if (~hsame).any():
            gh_cross_l.append(float((QW[gidx] @ QW[hard50[~hsame]].T).mean()))
    # 会话聚合(同一批256维向量)
    smax = np.full(NS, -1e9, dtype=np.float32)
    np.maximum.at(smax, uinv, cq)
    smask = np.zeros(NS, dtype=bool)
    smask[list(gold_convs)] = True
    top1 = int(np.argmax(smax))
    sess_hit1.append(1.0 if smask[top1] else 0.0)
    gvals = smax[smask]
    nvals = smax[~smask]
    sess_margin.append(float(gvals.max() - (nvals.max() if len(nvals) else -1.0)))

P("")
P("===== 同会话混杂对照 (n=%d题) =====" % len(SEL))
P("硬噪top50中同会话占比: 均值 %.1f%%" % (100.0 * np.mean(same_frac)))
if gg_l:
    P("金金内聚cos:            %.4f (n=%d题)" % (np.mean(gg_l), len(gg_l)))
if gh_same_l:
    P("金-硬噪(同会话)cos:     %.4f (n=%d题)" % (np.mean(gh_same_l), len(gh_same_l)))
if gh_cross_l:
    P("金-硬噪(跨会话)cos:     %.4f (n=%d题)" % (np.mean(gh_cross_l), len(gh_cross_l)))
if gg_l and gh_same_l:
    P("同会话内部结构差(金金-金同会话硬噪): %.4f" % (np.mean(gg_l) - np.mean(gh_same_l)))
P("")
P("===== 会话聚合视角(同一批256维向量, max聚合) =====")
P("金会话hit@1: %.1f%%" % (100.0 * np.mean(sess_hit1)))
P("会话margin(最佳金会话-最佳噪会话) 中位 %.4f | >0占 %.1f%%" % (
    np.median(sess_margin), 100.0 * (np.array(sess_margin) > 0).mean()))
P("")
P("F132B_DONE %.0fs" % (time.time() - t0))
LOG.close()
