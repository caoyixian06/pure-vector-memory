# -*- coding: utf-8 -*-
"""foundry26.py — 压缩层优化: 全武器LOCO压缩器 (池650→30行)
特征321: 12衍生+4形态+100dense+100差+100qwen+colbert+sparse+C0+FINAL+2邻句
协议: 会话留一10折。靶: all-gold@15/@30 + any@30。
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry26_results.txt"
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
ISRAW = np.array([(json.loads(l).get("kind") or "summary") == "raw"
                  for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()])
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
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QI_of = {qa: i for i, qa in enumerate(IDS)}

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]

# ===== colbert/sparse 打分 =====
import torch
mats, offsets = [], []
run = 0
nch = len([f for f in os.listdir(HERE + "/colbert_parts") if f.startswith("c") and f.endswith(".npz")])
for ci in range(nch):
    _p = np.load(HERE + "/colbert_parts/c%04d.npz" % ci)
    m = _p["mat"]
    counts = _p["counts"]
    mats.append(m)
    for c in counts:
        offsets.append((run, run + int(c)))
        run += int(c)
Pm = torch.from_numpy(np.concatenate(mats)).to("cuda")
Pm = Pm / (Pm.norm(dim=1, keepdim=True) + 1e-9)
OFF = np.array(offsets, dtype=np.int64)
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
enc = bge.encode([Q[qa]["question"] for qa in IDS], return_dense=False,
                 return_sparse=True, return_colbert_vecs=True)
COLB = []
for k_i in range(len(IDS)):
    qv = np.asarray(enc["colbert_vecs"][k_i], dtype=np.float16)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    qt = torch.from_numpy(qv).to("cuda")
    s = (qt @ Pm.T).cpu().numpy()
    COLB.append(np.maximum.reduceat(s, OFF[:, 0], axis=1).mean(axis=0))
import scipy.sparse as sp
SP = []
for ci in range(nch):
    with open(HERE + "/colbert_parts/s%04d.pkl" % ci, "rb") as f:
        SP.extend(pickle.load(f))
QSP = enc["lexical_weights"]
vocab = {}
rows, cols, vals = [], [], []
for r, d in enumerate(SP + QSP):
    for tok, w in d.items():
        j = vocab.setdefault(tok, len(vocab))
        rows.append(r); cols.append(j); vals.append(float(w))
MALL = sp.csr_matrix((vals, (rows, cols)), shape=(len(SP) + len(QSP), len(vocab)))
SALL = np.asarray((MALL[:NR] @ MALL[NR:].T).todense())
P("channels ready %.0fs" % (time.time() - t0))

# ===== 池 + 特征 =====
TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

POOLSZ = 650
CONVS = sorted(set(qa.split("#")[0] for qa in IDS))
FOLD_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(FOLD_of.values()))
FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    colb = COLB[k_i]
    sc = SALL[:, k_i]
    pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-colb)[:200]) | set(np.argsort(-sc)[:150])
    for i in list(np.argsort(-FINAL[qi]))[:50]:
        pool.add(max(0, i - 1))
        pool.add(min(NR - 1, i + 1))
    pool = sorted(pool)[:POOLSZ]
    POOLA[k_i] = pool
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    top50set = set(np.argsort(-C0[qi])[:50])
    Fm = np.zeros((len(pool), 321), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = ctoks(c)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if ((interr and interr in RAW[c].lower())) else 0.0
        lenratio = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
        v_qc = float(qv256 @ QW[c])
        d_qc = float(D[c] @ qv1024)
        nb1 = c - 1 if c > 0 and CONVKEY[c - 1] == CONVKEY[c] else -1
        nb2 = c + 1 if c + 1 < NR and CONVKEY[c + 1] == CONVKEY[c] else -1
        nbmax = 0.0
        for nb in (nb1, nb2):
            if nb >= 0:
                nbmax = max(nbmax, float(C0[qi][nb]))
        nbin = 1.0 if (nb1 in top50set) or (nb2 in top50set) else 0.0
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lenratio, v_qc, v_qc - d_qc,
                        d_qc, d_qc - v_qc, d_qc - qcov, qcov - d_qc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(colb[c])
        Fm[rr, 317] = float(sc[c])
        Fm[rr, 318] = float(C0[qi][c])
        Fm[rr, 319] = float(FINAL[qi][c])
        Fm[rr, 320:321] = [nbmax * 0 + nbin]
    FEATS[k_i] = Fm
    POOLA[k_i] = pool
    if k_i % 200 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("features built %.0fs" % (time.time() - t0))

# ===== LOCO 10折 =====
from sklearn.ensemble import HistGradientBoostingClassifier
K = (5, 15, 30)
res = {"gbdt": [0]*3, "final": [0]*3, "c0": [0]*3, "colbert": [0]*3}
anyr = {"gbdt": 0, "final": 0, "c0": 0, "colbert": 0}
n = 0
for hold in FOLDS:
    trF, trY = [], []
    for k_i, qa in enumerate(IDS):
        if FOLD_of[qa] == hold:
            continue
        Fm = FEATS[k_i]
        pool = POOLA[k_i]
        G = GSETS[k_i]
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        if noise_rows:
            noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15])
        else:
            noisepick = []
        for rr in gold_rows:
            trF.append(Fm[rr]); trY.append(1)
        for rr in noisepick:
            trF.append(Fm[rr]); trY.append(0)
    Ftr = np.array(trF, dtype=np.float32)
    Ytr = np.array(trY, dtype=np.int8)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(Ftr, Ytr)
    for k_i, qa in enumerate(IDS):
        if FOLD_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        qi = IDX[qa]
        Fm = FEATS[k_i]
        pool = POOLA[k_i]
        s = clf.decision_function(Fm)
        o = [pool[i] for i in np.argsort(-s)]
        og = {i: r + 1 for r, i in enumerate(o)}
        rks = [og.get(i, 10**9) for i in G]
        for k_j, k in enumerate(K):
            if max(rks) <= k:
                res["gbdt"][k_j] += 1
        if min(rks) <= 30:
            anyr["gbdt"] += 1
        for nm, sc_vec in (("final", FINAL[qi]), ("c0", C0[qi]), ("colbert", COLB[k_i])):
            o2 = np.argsort(-sc_vec)
            rks2 = [int(np.where(o2 == i)[0][0]) + 1 for i in G]
            for k_j, k in enumerate(K):
                if max(rks2) <= k:
                    res[nm][k_j] += 1
            if min(rks2) <= 30:
                anyr[nm] += 1
    P("  fold %s done" % hold)

P("\n===== 压缩层LOCO计分板 (n=%d) =====" % n)
for nm in ("gbdt", "final", "c0", "colbert"):
    a, b, c = res[nm]
    P("%-9s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%% any@30=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n), 100.0*anyr[nm]/max(1,n)))
P("FOUNDRY26_DONE %.0fs" % (time.time() - t0))
LOG.close()
