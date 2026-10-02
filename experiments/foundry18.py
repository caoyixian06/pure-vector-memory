# -*- coding: utf-8 -*-
"""foundry18.py — 原文专属排序: 摘要出清后的all-gold@5(被浪费的座位)
V_A FINAL原文专属 / V_B colbert原文专属 / V_C 融合原文专属 / 基线对照
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry18_results.txt"
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
RAWIDX = np.where(ISRAW)[0]
P("raw记录=%d/%d (%.1f%%)" % (ISRAW.sum(), NR, 100.0 * ISRAW.mean()))

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
GRAW = [set(i for i in G if ISRAW[i]) for G in GSETS]
P("金证据全raw可满足的题: %d/%d" % (sum(1 for g in GRAW if g), len(GSETS)))

# colbert分数(重算, 80s)
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
                 return_sparse=False, return_colbert_vecs=True)
P("colbert ready %.0fs" % (time.time() - t0))

def zsd(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)

K = (5, 10, 30)
V = {"FINAL混排": [0]*3, "FINAL原文专属": [0]*3, "colbert原文专属": [0]*3,
     "融合原文专属": [0]*3, "FINAL原文@原始窗": [0]*3}
n = 0
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GRAW[k_i]
    if not G:
        continue
    n += 1
    qv = np.asarray(enc["colbert_vecs"][k_i], dtype=np.float16)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    qt = torch.from_numpy(qv).to("cuda")
    s = (qt @ Pm.T).cpu().numpy()
    colb = np.maximum.reduceat(s, OFF[:, 0], axis=1).mean(axis=0)
    # 原文专属排序
    rf = [i for i in np.argsort(-FINAL[qi]) if ISRAW[i]]
    rc = [i for i in np.argsort(-colb) if ISRAW[i]]
    fus = zsd(FINAL[qi]) + zsd(colb)
    rfus = [i for i in np.argsort(-fus) if ISRAW[i]]
    orders = {"FINAL混排": [i for i in np.argsort(-FINAL[qi])],
              "FINAL原文专属": rf, "colbert原文专属": rc, "融合原文专属": rfus}
    for nm, order in orders.items():
        rank = {i: r + 1 for r, i in enumerate(order)}
        rks = [rank.get(i, 10**9) for i in G]
        for k_j, k in enumerate(K):
            if max(rks) <= k:
                V[nm][k_j] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== 原文专属 all-gold@K (n=%d) =====" % n)
for nm in V:
    a, b, c = V[nm]
    P("%-14s all@5=%.1f%% all@10=%.1f%% all@30=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n)))
P("FOUNDRY18_DONE %.0fs" % (time.time() - t0))
LOG.close()
