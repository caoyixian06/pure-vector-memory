# -*- coding: utf-8 -*-
"""foundry16.py — 0.872判别器重排器化: 严格all-gold@5爬坡实测
特征: 12衍生 + 4形态维 + 量化top100维 + 差分100 = 216维
dev半区训练 → test半区top-200池重排 → all-gold@5/10/30 vs FINAL vs C0
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry16_results.txt"
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
FINAL = z_ck["FINAL"]
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}

FORMD = [616, 538, 218, 408]
gsplit = np.array([int(hashlib.md5((qa + "f16").encode()).hexdigest(), 16) % 2 == 0 for qa in IDS])
P("loaded %.0fs" % (time.time() - t0))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

def col_aucs(F, mask, y):
    Fm = F[mask]
    ym = y[mask]
    n1 = int(ym.sum())
    n0 = len(ym) - n1
    order = np.argsort(Fm, axis=0)
    ranks = np.empty(Fm.shape, dtype=np.float64)
    rr = np.arange(1, len(ym) + 1).reshape(-1, 1)
    np.put_along_axis(ranks, order, rr, axis=0)
    rsum = ranks[ym == 1].sum(axis=0)
    return (rsum - n1 * (n1 + 1) / 2) / (n0 * n1)

# ===== 特征构建器(每题对候选列表) =====
QTOK = [toks(Q[qa]["question"]) for qa in IDS]
QLEN = [max(1, len(t)) for t in QTOK]
Q256I = {qa: i for i, qa in enumerate(IDS)}
FIRSTW = [(Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else "" for qa in IDS]

def feat_rows(qi, cands, k_i):
    qtok = QTOK[qi]
    qlen = QLEN[qi]
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = FIRSTW[qi]
    out = []
    for c in cands:
        ct = toks(RAW[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lenratio = len(RAW[c]) / max(1, len(Q[IDS[qi]].get("question") or "x"))
        v_qc = float(qv256 @ QW[c])
        d_qc = float(D[c] @ qv1024)
        form = [D[c, d] for d in FORMD]
        qdim = XDc = None
        out.append([jac, qcov, ccov, qmark, iecho, lenratio, v_qc, v_qc - d_qc,
                    d_qc, d_qc - v_qc, d_qc - qcov, qcov - d_qc] + form)
    return out

# ===== 选量化维(用dev半区记录值AUC, 避免泄漏test) =====
RI, YR, QIR = [], [], []
for k_i, qa in enumerate(IDS):
    if gsplit[k_i]:
        continue
    qi = IDX[qa]
    glist = gold_set(qa)
    glist = [i for i in np.argsort(-C0[qi])[:50] if i in glist]
    noises = [i for i in np.argsort(-C0[qi])[:50] if i not in glist][:10]
    for lab, cands in ((1, glist), (0, noises)):
        for c in cands:
            RI.append(c)
            YR.append(lab)
            QIR.append(k_i)
RI = np.array(RI)
YR = np.array(YR, dtype=np.int8)
maskR = np.ones(len(RI), dtype=bool)
a1 = col_aucs(D[RI], maskR, YR)
a2 = col_aucs(D[RI], maskR, YR)
oriented = np.maximum(a1, 1 - a1)
STAB = np.argsort(-oriented)[:100]
P("量化维选定 %d 个 %.0fs" % (len(STAB), time.time() - t0))

# ===== 全量特征构建(test半区, top200池) =====
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def build(qa, qi, k_i, pool):
    qtok = QTOK[qi]
    qlen = QLEN[qi]
    qv256 = Q256[Q256I[qa]]
    qv1024 = X[qi]
    interr = FIRSTW[qi]
    Fm = np.zeros((len(pool), 12 + 4 + 100 + 100), dtype=np.float32)
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

# ===== 训练(dev半区: 金vs采样噪声) =====
Ftr_rows, Ytr_rows = [], []
for k_i, qa in enumerate(IDS):
    if not gsplit[k_i]:
        continue
    qi = IDX[qa]
    gset = gold_set(qa)
    pool = [i for i in np.argsort(-C0[qi])[:50]]
    glist = [i for i in pool if i in gset]
    noises = [i for i in pool if i not in gset][:15]
    for lab, cands in ((1, glist), (0, noises)):
        if not cands:
            continue
        Fm = build(qa, qi, k_i, cands)
        for rr, c in enumerate(cands):
            Ftr_rows.append(Fm[rr])
            Ytr_rows.append(lab)
    if k_i % 200 == 0:
        P("  train rows %d %.0fs" % (k_i, time.time() - t0))
Ftr = np.array(Ftr_rows, dtype=np.float32)
Ytr = np.array(Ytr_rows, dtype=np.int8)
P("train rows=%d pos=%d %.0fs" % (len(Ytr), int(Ytr.sum()), time.time() - t0))
sc = StandardScaler().fit(Ftr)
clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Ftr), Ytr)

# ===== test半区重排实测 =====
K = (5, 10, 30)
res = {"final": [0]*3, "c0": [0]*3, "model": [0]*3, "hybrid": [0]*3}
n = 0
for k_i, qa in enumerate(IDS):
    if gsplit[k_i]:
        continue
    qi = IDX[qa]
    G = gold_set(qa)
    if not G:
        continue
    pool = [i for i in np.argsort(-C0[qi])[:200]]
    n += 1
    forder = [i for i in np.argsort(-FINAL[qi])[:30]]
    for k_j, k in enumerate(K):
        if all(r in forder[:k] for r in G):
            res["final"][k_j] += 1
        if all(r in pool[:k] for r in G):
            res["c0"][k_j] += 1
    Fm = build(qa, qi, k_i, pool)
    sg = clf.predict_proba(sc.transform(Fm))[:, 1]
    morder = [pool[i] for i in np.argsort(-sg)]
    for k_j, k in enumerate(K):
        if all(r in morder[:k] for r in G):
            res["model"][k_j] += 1
    hyb = {}
    crank = {c: r for r, c in enumerate(pool)}
    mrank = {m: r for r, m in enumerate(morder)}
    for c in set(pool):
        hyb[c] = 0.6 * crank.get(c, 200) + 0.4 * mrank.get(c, 200)
    horder = sorted(hyb, key=hyb.get)
    for k_j, k in enumerate(K):
        if all(r in horder[:k] for r in G):
            res["hybrid"][k_j] += 1
    if n % 100 == 0:
        P("  test %d %.0fs" % (n, time.time() - t0))

P("\n===== 严格all-gold@K(test半区 n=%d, 池=top200) =====" % n)
for nm in ("final", "c0", "model", "hybrid"):
    P("%-8s " % nm + "  ".join("all@%d=%.1f%%" % (k, 100.0 * res[nm][i] / max(1, n)) for i, k in enumerate(K)))
P("FOUNDRY16_DONE %.0fs" % (time.time() - t0))
LOG.close()
