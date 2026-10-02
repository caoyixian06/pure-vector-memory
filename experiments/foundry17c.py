# -*- coding: utf-8 -*-
"""foundry17c.py — FINAL×colbert×sparse 融合: 已知纯向量极限的最后一拼
V1 z(F)+z(C)  V2 z(F)+0.5z(C)  V3 0.5z(F)+z(C)  V4 z(F)+z(C)+0.5z(S)
靶: 严格all-gold@5/10/30 + any@5。
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry17c_results.txt"
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

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
P("loaded %.0fs" % (time.time() - t0))

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
Pm = torch.from_numpy(np.concatenate(mats)).to(DEV := "cuda")
Pm = Pm / (Pm.norm(dim=1, keepdim=True) + 1e-9)
OFF = np.array(offsets, dtype=np.int64)
P("colbert P ready %s %.0fs" % (tuple(Pm.shape), time.time() - t0))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def colbert_scores(qa_list):
    qs = [Q[qa]["question"] for qa in qa_list]
    enc = bge.encode(qs, return_dense=False, return_sparse=True, return_colbert_vecs=True)
    out = []
    for k_i in range(len(qa_list)):
        qv = np.asarray(enc["colbert_vecs"][k_i], dtype=np.float16)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        qt = torch.from_numpy(qv).to(DEV)
        s = qt @ Pm.T
        s = s.cpu().numpy()
        segmax = np.maximum.reduceat(s, OFF[:, 0], axis=1)
        out.append(segmax.mean(axis=0))
    return out, enc["lexical_weights"]

colb_scores, QSP = colbert_scores(IDS)
P("colbert scored %.0fs" % (time.time() - t0))

SP = []
for ci in range(nch):
    with open(HERE + "/colbert_parts/s%04d.pkl" % ci, "rb") as f:
        SP.extend(pickle.load(f))

def build_csr(dicts):
    vocab = {}
    rows, cols, vals = [], [], []
    for r, d in enumerate(dicts):
        for tok, w in d.items():
            j = vocab.setdefault(tok, len(vocab))
            rows.append(r); cols.append(j); vals.append(float(w))
    import scipy.sparse as sp
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(dicts), len(vocab)))

def sparse_scores(qa_list):
    qd = [QSP[k_i] for k_i in range(len(qa_list))]
    M = build_csr(SP + qd)
    Mp = M[:len(SP)]
    Mq = M[len(SP):]
    S = Mp @ Mq.T
    return [np.asarray(S[:, k_i].todense()).ravel() for k_i in range(len(qa_list))]

sp_scores = sparse_scores(IDS)
P("sparse scored %.0fs" % (time.time() - t0))

def zsd(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)

K = (5, 10, 30)
V = {"V1_zF+zC": [0]*3, "V2_zF+0.5zC": [0]*3, "V3_0.5zF+zC": [0]*3,
     "V4_zF+zC+0.5zS": [0]*3, "V5_zF": [0]*3, "V6_zC": [0]*3}
Vany = {k: [0] for k in V}
n = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if not G:
        continue
    n += 1
    qi = IDX[qa]
    zF = zsd(FINAL[qi])
    zC = zsd(colb_scores[k_i])
    zS = zsd(sp_scores[k_i])
    variants = {"V1_zF+zC": zF + zC, "V2_zF+0.5zC": zF + 0.5 * zC,
                "V3_0.5zF+zC": 0.5 * zF + zC, "V4_zF+zC+0.5zS": zF + zC + 0.5 * zS,
                "V5_zF": zF, "V6_zC": zC}
    for nm, sc in variants.items():
        o = np.argsort(-sc)
        rk = {i: r + 1 for r, i in enumerate(o)}
        rks = [rk.get(i, 10**9) for i in G]
        for k_j, k in enumerate(K):
            if max(rks) <= k:
                V[nm][k_j] += 1
        if min(rks) <= 5:
            Vany[nm][0] += 1
    if n % 400 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== FINAL×colbert×sparse 融合(n=%d) =====" % n)
for nm in V:
    a, b, c = V[nm]
    P("%-16s all@5=%.1f%% all@10=%.1f%% all@30=%.1f%% any@5=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n), 100.0*Vany[nm][0]/max(1,n)))
P("FOUNDRY17C_DONE %.0fs" % (time.time() - t0))
LOG.close()
