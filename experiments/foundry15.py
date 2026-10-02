# -*- coding: utf-8 -*-
"""foundry15.py — ①合并探测: 量化维度+12衍生特征 vs 各自单飞 ②黄金维解剖: d616等编码什么属性
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry15_results.txt"
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
KINDARR = [1 if (json.loads(l).get("kind") or "summary") == "raw" else 0
           for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
RI, QI, Y, FOLD = [], [], [], []
FN = ["t_jacc", "t_qcov", "t_ccov", "t_qmark", "t_interr", "t_lenratio",
      "v256_qc", "v256_gap", "d1024_qc", "d1024_gap", "x_reword", "x_echo"]
QTOK = [toks(Q[qa]["question"]) for qa in IDS]
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    glist = gold_list(qa)
    if not glist:
        continue
    gset = set(glist)
    top50 = [i for i in np.argsort(-C0[qi])[:50]]
    noises = [i for i in top50 if i not in gset][:20]
    qtok = QTOK[k_i]
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    first_w = (Q[qa].get("question") or "").strip().lower()
    interr = first_w.split()[0] if first_w else ""
    for lab, cands in ((1, glist), (0, noises)):
        for c in cands:
            ct = toks(RAW[c])
            inter = qtok & ct
            jac = len(inter) / max(1, len(qtok | ct))
            qcov = len(inter) / qlen
            ccov = len(inter) / max(1, len(ct))
            qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
            iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
            lenratio = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
            v_qc = float(Q256[k_i] @ QW[c])
            d_qc = float(D[c] @ qv1024)
            FOLD.append([jac, qcov, ccov, qmark, iecho, lenratio,
                         v_qc, v_qc - d_qc, d_qc, d_qc - v_qc,
                         d_qc - qcov, qcov - d_qc])
            RI.append(c)
            QI.append(k_i)
            Y.append(lab)
    if k_i % 400 == 0:
        P("  rows %d %.0fs" % (k_i, time.time() - t0))
RI = np.array(RI)
QI = np.array(QI)
Y = np.array(Y, dtype=np.int8)
FOLD = np.array(FOLD, dtype=np.float32)
P("rows=%d %.0fs" % (len(Y), time.time() - t0))

# 量化维(复算记录原始值AUC选top)
RD = D[RI]
XD = X[QI]
gsplit = np.array([int(hashlib.md5((str(q) + "f15").encode()).hexdigest(), 16) % 2 == 0 for q in QI])

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

halftr = np.array([int(hashlib.md5((str(q) + "c").encode()).hexdigest(), 16) % 2 == 0 for q in QI])
a1 = col_aucs(RD, halftr, Y)
a2 = col_aucs(RD, ~halftr, Y)
oriented = np.maximum(a1, 1 - a1)
stable = (a1 > 0.52) & (a2 > 0.52)
top100 = np.argsort(-np.where(stable, oriented, 0))[:100]
QUANT = np.hstack([RD[:, top100], np.abs(XD[:, top100] - RD[:, top100])])
P("quant dims=%d %.0fs" % (QUANT.shape[1], time.time() - t0))

# ===== 合并探测 =====
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
def combo(F, tag):
    sc = StandardScaler().fit(F[gsplit])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[gsplit]), Y[gsplit])
    a = roc_auc_score(Y[~gsplit], clf.predict_proba(sc.transform(F[~gsplit]))[:, 1])
    P("%-24s held-out AUC=%.3f" % (tag, a))
    return a
P("\n===== 合并探测 =====")
combo(FOLD, "衍生12特征")
combo(QUANT, "量化维200")
combo(np.hstack([FOLD, QUANT]), "合并")

# ===== 黄金维解剖 =====
P("\n===== 黄金维解剖(d616四分位的记录属性) =====")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
def has_time(text):
    tl = str(text).lower()
    return any(m in tl for m in MONTHS) or re.search(r"\b(19|20)\d\d\b", tl) is not None
for dim in (616, 538, 218, 408):
    dv = D[:, dim]
    qs = np.quantile(dv, [0.25, 0.5, 0.75])
    bins = np.digitize(dv, qs)
    P("d%d:" % dim)
    for b in range(4):
        mask = bins == b
        lens = np.array([len(RAW[i]) for i in range(NR)])
        P("  Q%d: 长度=%.0f 时间词=%.1f%% 问句=%.1f%% raw形态=%.1f%%" % (
            b + 1, lens[mask].mean(), 100.0 * np.mean([has_time(RAW[i]) for i in np.where(mask)[0]]),
            100.0 * np.mean([RAW[i].rstrip().endswith("?") for i in np.where(mask)[0]]),
            100.0 * np.mean([KINDARR[i] for i in np.where(mask)[0]])))
P("FOUNDRY15_DONE %.0fs" % (time.time() - t0))
LOG.close()
