# -*- coding: utf-8 -*-
"""foundry20.py — 会话级留一(LOCO)去过拟合检验: GBDT重排器真水位
10场对话逐场留出: 训练9场, 测试1场(模型从未见过该对话)。对照FINAL同题成绩。
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry20_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))

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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
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
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
CONV = [qa.split("#")[0] for qa in IDS]
CONVS = sorted(set(CONV))
IDXM = {qa: i for i, qa in enumerate(IDS)}
P("loaded %d题 %d对话 %.0fs" % (len(IDS), len(CONVS), time.time() - t0))

FORMD = [616, 538, 218, 408]
STAB = np.load(HERE + "/f16_stab.npy") if os.path.exists(HERE + "/f16_stab.npy") else np.arange(100)
QTOK = [toks(Q[qa]["question"]) for qa in IDS]
QLEN = [max(1, len(t)) for t in QTOK]

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

def feat_rows(qi, qa, pool):
    qtok = QTOK[qi]
    qlen = QLEN[qi]
    qv256 = Q256[Q256I[qa]]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0]
    Fm = np.zeros((len(pool), 216), dtype=np.float32)
    XDq = X[qi][STAB]
    for rr, c in enumerate(pool):
        ct = toks(RAW[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lenratio = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
        v_qc = float(qv256 @ QW[c])
        d_qc = float(D[c] @ qv1024)
        Fm[rr, :12] = [jac, qcov, ccov, qmark, iecho, lenratio, v_qc, v_qc - d_qc,
                       d_qc, d_qc - v_qc, d_qc - qcov, qcov - d_qc]
        Fm[rr, 12:16] = [D[c, d] for d in FORMD]
        Fm[rr, 16:116] = D[c][STAB]
        Fm[rr, 116:216] = np.abs(XDq - D[c][STAB])
    return Fm

Q256I = {qa: i for i, qa in enumerate(IDS)}
# ===== 全量池特征一次构建(1374题×150池, 控制耗时) =====
POOLSIZE = 150
FALL = []
YALL = []
POOL = []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = gold_set(qa)
    pool = [i for i in np.argsort(-C0[qi])[:POOLSIZE]]
    glist = [i for i in pool if i in G][:8]
    noises = [i for i in pool if i not in G][:12]
    cands = glist + noises
    Fm = feat_rows(qi, qa, cands)
    FALL.append(Fm)
    YALL.extend([1]*len(glist) + [0]*len(noises))
    POOL.append(cands)
    if k_i % 300 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
# 拼接成按题索引的列表(变长, 不合并大矩阵)
FALL_sizes = [len(f) for f in FALL]
P("features built: %d行 %.0fs" % (sum(FALL_sizes), time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
K = (5, 10, 30)
pooled = {"gbdt": [0]*3, "final": [0]*3}
folds = []
for hold in CONVS:
    te_qs = [qa for qa in IDS if qa.split("#")[0] == hold]
    if len(te_qs) < 10:
        continue
    trF_all = []
    trY_all = []
    for f, qa in zip(FALL, IDS):
        if qa.split("#")[0] == hold:
            continue
        G = gold_set(qa)
        glist = [i for i in POOL[IDXM[qa]] if i in G][:8]
        for rr in range(min(len(f), len(glist))):
            trF_all.append(f[rr])
            trY_all.append(1)
        for rr in range(len(glist), min(len(f), len(glist) + 12)):
            trF_all.append(f[rr])
            trY_all.append(0)
    Ftr = np.array(trF_all, dtype=np.float32)
    Ytr = np.array(trY_all, dtype=np.int8)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(Ftr, Ytr)
    f5 = f10 = f30 = cnt = 0
    fbase5 = fbase10 = fbase30 = 0
    for qa in te_qs:
        qi = IDX[qa]
        G = gold_set(qa)
        if not G:
            continue
        cnt += 1
        fi = IDXM[qa]
        Fm = FALL[fi]
        s = clf.decision_function(Fm)
        pool = POOL[fi]
        order = [pool[i] for i in np.argsort(-s)]
        for k_j, k in enumerate(K):
            if all(r in order[:k] for r in G):
                if k_j == 0: f5 += 1
                if k_j == 1: f10 += 1
                if k_j == 2: f30 += 1
        forder = [i for i in np.argsort(-FINAL[qi])[:30]]
        for k_j, k in enumerate(K):
            if all(r in forder[:k] for r in G):
                if k_j == 0: fbase5 += 1
                if k_j == 1: fbase10 += 1
                if k_j == 2: fbase30 += 1
    folds.append((hold, cnt, f5, fbase5, f10, f10b := fbase10, f30, fbase30))
    P("hold=%-8s n=%3d GBDT@5=%.1f%% FINAL@5=%.1f%% | @30 GBDT=%.1f%% FINAL=%.1f%%" % (
        hold, cnt, 100.0*f5/max(1,cnt), 100.0*fbase5/max(1,cnt),
        100.0*f30/max(1,cnt), 100.0*fbase30/max(1,cnt)))

tot_gbdt5 = sum(f[2] for f in folds); tot_base5 = sum(f[3] for f in folds)
tot_n = sum(f[1] for f in folds)
tot_gbdt30 = sum(f[6] for f in folds); tot_base30 = sum(f[7] for f in folds)
P("\nLOCO合并: GBDT@5=%.1f%% vs FINAL@5=%.1f%% | @30 GBDT=%.1f%% vs FINAL=%.1f%%" % (
    100.0*tot_gbdt5/max(1,tot_n), 100.0*tot_base5/max(1,tot_n),
    100.0*tot_gbdt30/max(1,tot_n), 100.0*tot_base30/max(1,tot_n)))
P("FOUNDRY20_DONE %.0fs" % (time.time() - t0))
LOG.close()
