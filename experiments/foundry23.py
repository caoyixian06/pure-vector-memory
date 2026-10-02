# -*- coding: utf-8 -*-
"""foundry23.py — 检索层通关扫描: 并集配比全扫(目标 all-gold@池 >= 93%)
变体: FINAL/COLBERT/Sparse 三源不同配比 + 邻句增强, 报各池尺寸下的all-gold召回
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry23_results.txt"
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
GSETS = [gold_set_raw(qa) for qa in IDS]
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
Pm = torch.from_numpy(np.concatenate(mats)).to("cuda")
Pm = Pm / (Pm.norm(dim=1, keepdim=True) + 1e-9)
OFF = np.array(offsets, dtype=np.int64)
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
enc = bge.encode([Q[qa]["question"] for qa in IDS], return_dense=False,
                 return_sparse=True, return_colbert_vecs=True)
QSP = enc["lexical_weights"]
P("colbert ready %.0fs" % (time.time() - t0))

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

Mall = build_csr(SP + QSP)
Msp = Mall[:NR]
Msq = Mall[NR:]
SPMAT = (Msp @ Msq.T)
P("sparse scored %.0fs" % (time.time() - t0))

colb_scores_cache = []
for k_i in range(len(IDS)):
    qv = np.asarray(enc["colbert_vecs"][k_i], dtype=np.float16)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    qt = torch.from_numpy(qv).to("cuda")
    s = (qt @ Pm.T).cpu().numpy()
    colb_scores_cache.append(np.maximum.reduceat(s, OFF[:, 0], axis=1).mean(axis=0))
P("colbert scored %.0fs" % (time.time() - t0))

def zsd(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)

VARIANTS = {
    "F150+C150+nb":     [("F", 150), ("C", 150), ("NB", 50)],
    "F200+C100":        [("F", 200), ("C", 100), ("NB", 0)],
    "F100+C200":        [("F", 100), ("C", 200), ("NB", 0)],
    "F150+C150+S100":   [("F", 150), ("C", 150), ("S", 100)],
    "F200+C150+NB50":   [("F", 200), ("C", 150), ("NB", 50)],
    "F250+C200+NB50+S": [("F", 250), ("C", 200), ("NB", 50), ("S", 150)],
    "F300+C250+NB75+S": [("F", 300), ("C", 250), ("NB", 75), ("S", 200)],
}
KS = (150, 250, 400, 500)
res = {v: [0]*len(KS) for v in VARIANTS}
sizes = {v: 0 for v in VARIANTS}
n = 0
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
        continue
    n += 1
    forder = list(np.argsort(-FINAL[qi]))
    corder = list(np.argsort(-colb_scores_cache[k_i]))
    sorder = list(np.argsort(-SPMAT[:, k_i].toarray().ravel()))
    for vname, parts in VARIANTS.items():
        pool = set()
        for src, kk in parts:
            if src == "F":
                pool |= set(forder[:kk])
            elif src == "C":
                pool |= set(corder[:kk])
            elif src == "S":
                pool |= set(sorder[:kk])
            elif src == "NB":
                for i in forder[:kk]:
                    if i > 0:
                        pool.add(i - 1)
                    if i + 1 < NR:
                        pool.add(i + 1)
        sizes[vname] = max(sizes[vname], len(pool))
        for k_j, k in enumerate(KS):
            if all(r in pool for r in G):
                res[vname][k_j] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== 检索层通关扫描 (n=%d) =====" % n)
P("变体                    " + "".join("all@%-4d " % k for k in KS) + " 最大池")
for vname in VARIANTS:
    P("%-22s" % vname + "".join("  %5.1f%%" % (100.0 * res[vname][i] / max(1, n)) for i in range(len(KS)))
      + "   %d" % sizes[vname])
P("FOUNDRY23_DONE %.0fs" % (time.time() - t0))
LOG.close()
