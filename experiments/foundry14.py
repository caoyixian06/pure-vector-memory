# -*- coding: utf-8 -*-
"""foundry14.py — 维度信息普查: 256+1024每个维度的金/噪判别力量化(三视角×拆半×组合)
视角: A记录原始值r[d] / B交互x[d]·r[d] / C差|x[d]−r[d]|
判定用户假说"每个维度都存在信息": AUC分布直方图 + 拆半稳定 + 量化维度组合 vs 衍生特征0.844
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry14_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
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
RAWN = [norm(rec_raw(m)) for m in MID]
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
C0 = l2n(X @ D.T)
_cw = np.load(HERE + "/q256_cache.npz")
Q256 = _cw["Q"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
P("loaded %.0fs" % (time.time() - t0))

def gold_list(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = []
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.append(i)
    return g

import random
rng = random.Random(11)
RI, QI, Y = [], [], []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    glist = gold_list(qa)
    if not glist:
        continue
    gset = set(glist)
    top50 = [i for i in np.argsort(-C0[qi])[:50]]
    noises = [i for i in top50 if i not in gset][:20]
    for lab, cands in ((1, glist), (0, noises)):
        for c in cands:
            RI.append(c)
            QI.append(k_i)
            Y.append(lab)
RI = np.array(RI)
QI = np.array(QI)
Y = np.array(Y, dtype=np.int8)
nR = len(Y)
P("rows=%d gold=%d %.0fs" % (nR, int(Y.sum()), time.time() - t0))

# ===== 矩阵化特征 =====
RD = D[RI]
XD = X[QI]
RQ = QW[RI]
XQ = Q256[QI]
PROD_D = XD * RD
DIFF_D = np.abs(XD - RD)
PROD_Q = XQ * RQ
DIFF_Q = np.abs(XQ - RQ)
P("matrices built %.0fs" % (time.time() - t0))

# ===== 向量化逐维AUC(秩和法, 按行拆半) =====
gsplit = np.array([int(hashlib.md5((str(q) + "f14").encode()).hexdigest(), 16) % 2 == 0 for q in QI])

def col_aucs(F, mask, y):
    """对mask行算每列的AUC(y=1为金)"""
    Fm = F[mask]
    ym = y[mask]
    n1 = int(ym.sum())
    n0 = len(ym) - n1
    if n1 == 0 or n0 == 0:
        return np.zeros(F.shape[1]), np.zeros(F.shape[1])
    order = np.argsort(Fm, axis=0)
    ranks = np.empty(Fm.shape, dtype=np.float64)
    rr = np.arange(1, len(ym) + 1).reshape(-1, 1)
    np.put_along_axis(ranks, order, rr, axis=0)
    rsum = ranks[ym == 1].sum(axis=0)
    return (rsum - n1 * (n1 + 1) / 2) / (n0 * n1), None

def probe(name, F):
    a1, _ = col_aucs(F, gsplit, Y)
    a2, _ = col_aucs(F, ~gsplit, Y)
    oriented = np.maximum(a1, 1 - a1)
    o2 = np.maximum(a2, 1 - a2)
    st = float(np.corrcoef(a1, a2)[0, 1])
    P("\n[%s]" % name)
    P("  维度数=%d | AUC中位=%.3f | >0.52: %d (%.0f%%) | >0.55: %d | >0.60: %d" % (
        F.shape[1], np.median(oriented), int((oriented > 0.52).sum()),
        100.0 * (oriented > 0.52).mean(), int((oriented > 0.55).sum()), int((oriented > 0.60).sum())))
    P("  拆半相关=%.3f" % st)
    top = np.argsort(-oriented)[:12]
    P("  Top12: " + ", ".join("d%d=%.3f" % (int(j), oriented[j]) for j in top))
    both = (a1 > 0.52) & (a2 > 0.52)
    P("  拆半双侧>0.52的稳定维: %d" % int(both.sum()))
    return a1, a2, both

P("\n===== 1024空间 =====")
a1d, a2d, both_d = probe("1024·记录原始值", RD)
probe("1024·交互积", PROD_D)
probe("1024·差", DIFF_D)
P("\n===== 256空间 =====")
a1q, a2q, both_q = probe("256·记录原始值", RQ)
probe("256·交互积", PROD_Q)
probe("256·差", DIFF_Q)

# ===== 量化维度组合 vs 衍生特征0.844 =====
P("\n===== 组合对决(逻辑回归, held-out) =====")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
def combo(F, tag):
    sc = StandardScaler().fit(F[gsplit])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[gsplit]), Y[gsplit])
    a = roc_auc_score(Y[~gsplit], clf.predict_proba(sc.transform(F[~gsplit]))[:, 1])
    P("%s: held-out AUC=%.3f" % (tag, a))
    return a
topD = np.argsort(-np.maximum(a1d, 1 - a1d))[:100]
topQ = np.argsort(-np.maximum(a1q, 1 - a1q))[:50]
combo(np.hstack([PROD_D[:, topD], DIFF_D[:, topD], PROD_Q[:, topQ], DIFF_Q[:, topQ]]),
      "量化维度(150个稳定维组合)")
P("FOUNDRY14_DONE %.0fs" % (time.time() - t0))
LOG.close()
