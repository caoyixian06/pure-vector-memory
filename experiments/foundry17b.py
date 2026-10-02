# -*- coding: utf-8 -*-
"""foundry17b.py — M3完全体检索评测: dense/colbert(CoMaxSim)/sparse/融合
靶: 严格all-gold@5/10/30 + any@5 (1374题)。纯向量, 零API。
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry17b_results.txt"
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

# ===== colbert库加载 + torch MaxSim =====
import torch
DEV = "cuda"
mats, offsets, counts_all = [], [], []
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
    counts_all.extend([int(c) for c in counts])
Pm = torch.from_numpy(np.concatenate(mats)).to(DEV)
Pm = Pm / (Pm.norm(dim=1, keepdim=True) + 1e-9)
OFF = np.array(offsets, dtype=np.int64)
assert len(OFF) == NR
P("colbert P matrix: %s %.0fs" % (tuple(Pm.shape), time.time() - t0))

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
        s = qt @ Pm.T                       # (qt, total_tokens)
        s = s.cpu().numpy()
        # segment max per turn
        segmax = np.maximum.reduceat(s, OFF[:, 0], axis=1)   # (qt, NR)
        sc = segmax.mean(axis=0)                             # (NR,)
        out.append(sc)
    return out

# ===== sparse库 =====
SP = []
for ci in range(nch):
    with open(HERE + "/colbert_parts/s%04d.pkl" % ci, "rb") as f:
        SP.extend(pickle.load(f))
P("sparse dicts=%d" % len(SP))

qenc = bge.encode([Q[qa]["question"] for qa in IDS], return_dense=False, return_sparse=True)
QSP = qenc["lexical_weights"]

def build_csr(dicts):
    vocab = {}
    rows, cols, vals = [], [], []
    for r, d in enumerate(dicts):
        for tok, w in d.items():
            j = vocab.setdefault(tok, len(vocab))
            rows.append(r); cols.append(j); vals.append(float(w))
    import scipy.sparse as sp
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(dicts), len(vocab))), vocab

def sparse_scores(qa_list):
    qd = [QSP[k_i] for k_i in range(len(qa_list))]
    allv = SP + qd
    M, vocab = build_csr(allv)
    Mp = M[:len(SP)]
    Mq = M[len(SP):]
    S = (Mp @ Mq.T)
    out = []
    for k_i in range(len(qa_list)):
        out.append(np.asarray(S[:, k_i].todense()).ravel())
    return out

P("sparse ready %.0fs" % (time.time() - t0))

def zsd(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)

def goldstats(ranks_list, tag):
    K = (5, 10, 30)
    c_all = [0]*3
    c_any = [0]*3
    n = 0
    for qi, ranks in enumerate(ranks_list):
        G = GSETS[qi]
        if not G:
            continue
        n += 1
        rk = [ranks.get(i, 10**9) for i in G]
        for k_i, k in enumerate(K):
            if max(rk) <= k:
                c_all[k_i] += 1
            if min(rk) <= k:
                c_any[k_i] += 1
    P("%-14s " % tag + "  ".join("all@%d=%.1f%%" % (k, 100.0*c_all[i]/max(1,n)) for i,k in enumerate(K))
      + "   any@5=%.1f%%" % (100.0*c_any[0]/max(1,n)))

# ===== 评测(抽样300题加速colbert, 稳定即可判读; 全量可后补) =====
import random
EVAL = IDS
colb_scores = colbert_scores(EVAL)
sp_scores = sparse_scores(EVAL)
P("scored all %.0fs" % (time.time() - t0))

rank_colb, rank_sp, rank_den, rank_fus = [], [], [], []
for k_i, qa in enumerate(EVAL):
    qi = IDX[qa]
    colb = zsd(colb_scores[k_i])
    den = zsd(C0[qi])
    sp_ = zsd(sp_scores[k_i])
    fus = colb + den + 0.5*sp_
    o = np.argsort(-colb); rank_colb.append({i: r+1 for r, i in enumerate(o)})
    o = np.argsort(-sp_);  rank_sp.append({i: r+1 for r, i in enumerate(o)})
    o = np.argsort(-den);  rank_den.append({i: r+1 for r, i in enumerate(o)})
    o = np.argsort(-fus);  rank_fus.append({i: r+1 for r, i in enumerate(o)})

P("\n===== M3完全体严格all-gold =====")
goldstats(rank_den, "dense")
goldstats(rank_sp, "sparse")
goldstats(rank_colb, "colbert")
goldstats(rank_fus, "三合一")
P("FOUNDRY17B_DONE %.0fs" % (time.time() - t0))
LOG.close()
