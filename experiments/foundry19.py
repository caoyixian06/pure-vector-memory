# -*- coding: utf-8 -*-
"""foundry19.py — 用上"维度/记录间关系": ①孪生换位 ②PCA真坐标系GBDT ③时间街区PC特征
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry19_results.txt"
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
ISRAW = np.array([(json.loads(l).get("kind") or "summary") == "raw"
                  for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()])
TWIN = {}
MID2I = {m: i for i, m in enumerate(MID)}
for i, m in enumerate(MID):
    r = REC.get(m, {})
    tw = r.get("raw_of") or r.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)

def gold_set_raw(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if ISRAW[i] and any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

K = (5, 10, 30)
n = 0
gain5 = gain10 = gain30 = 0
base = [0, 0, 0]
swap = [0, 0, 0]
for qa in IDS:
    G = gold_set_raw(qa)
    if not G:
        continue
    n += 1
    qi = IDX[qa]
    order = list(np.argsort(-FINAL[qi]))
    # 基线: 混排
    for k_i, k in enumerate(K):
        if all(r in order[:k] for r in G):
            base[k_i] += 1
    # 孪生换位: 从深位把G的成员提到其孪生/更前位置(贪心: G成员按孪生或自身最高名次提升)
    pos = {r: i for i, r in enumerate(order)}
    neworder = list(order)
    moved = set()
    for r in G:
        p = pos.get(r)
        # 找它的孪生位置
        tp = None
        for other, t in TWIN.items():
            if t == r and other in pos:
                tp = pos[other]
        if p is None and tp is not None:
            p = tp
        if p is not None and p < 30:
            moved.append if False else None
            moved.add((min(p, 10**9), r, p))
    # 简化实现: 把G中成员(含经孪生可达者)全部提升到最前(保持相对序), 检查all@K提升潜力
    promoted = [r for r in order if r in G]
    rest = [r for r in order if r not in G]
    promoted_sorted = sorted(promoted, key=lambda r: pos.get(r, 10**9))
    neworder_oracle = promoted_sorted + rest
    for k_i, k in enumerate(K):
        if all(r in neworder_oracle[:k] for r in G):
            swap[k_i] += 1
            if k_i == 0 and base[0] == 0:
                pass
for k_i, k in enumerate(K):
    if k_i == 0:
        gain5 = swap[0] - base[0]
P("①孪生/自身提升oracle: base@5=%d/%d → oracle@5=%d/%d (提升=%d题)" % (
    base[0], n, swap[0], n, swap[0] - base[0]))
P("   @10: %d→%d  @30: %d→%d" % (base[1], swap[1], base[2], swap[2]))

# ===== ②PCA真坐标系 + HistGBD =====
P("\n②PCA旋转GBDT(时间题+开放域题的池内金排名)...")
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import random
rng = random.Random(11)
RI, QI, Y = [], [], []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    g = gold_set_raw(qa)
    if not g:
        continue
    top50 = [i for i in np.argsort(-C0[qi])[:50]]
    golds = [i for i in top50 if i in g]
    noises = [i for i in top50 if i not in g][:15]
    for lab, cands in ((1, golds), (0, noises)):
        for c in cands:
            RI.append(c); QI.append(k_i); Y.append(lab)
RI = np.array(RI); QI = np.array(QI); Y = np.array(Y, dtype=np.int8)
RD = D[RI]
XP = X[QI]
P("rows=%d %.0fs" % (len(Y), time.time() - t0))

# PCA 32维(在库全体上拟合)
pca = PCA(n_components=32, random_state=0)
DP = pca.fit(D).transform(D)
P("PCA explained variance top10: %.3f" % float(pca.explained_variance_ratio_[:10].sum()))
RP = DP[RI]
XP = DP[QI]
gsplit = np.array([int(hashlib.md5((str(q) + "p19").encode()).hexdigest(), 16) % 2 == 0 for q in QI])

def rank_eval(F, tag):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F[gsplit])
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(F[gsplit], Y[gsplit])
    s = np.full(len(Y), -1e9)
    s[~gsplit] = clf.decision_function(F[~gsplit])
    h1 = h5 = n = 0
    for qi in np.unique(QI[~gsplit]):
        m = QI == qi
        yq, sq = Y[m], s[m]
        if yq.sum() == 0:
            continue
        g = int(np.argmax(yq))
        n += 1
        r = int((sq >= sq[g]).sum())
        h1 += (r == 1)
        h5 += (r <= 5)
    P("%-20s hit@1=%.1f%% hit@5=%.1f%% (n=%d)" % (tag, 100.0*h1/max(1,n), 100.0*h5/max(1,n), n))

rank_eval(RD, "原始坐标GBDT")
rank_eval(RP, "PCA32旋转GBDT")
rank_eval(np.hstack([RD, RP, XP - RP]), "原始+旋转+差")
P("FOUNDRY19_DONE %.0fs" % (time.time() - t0))
LOG.close()
