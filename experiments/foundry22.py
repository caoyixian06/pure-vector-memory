# -*- coding: utf-8 -*-
"""foundry22.py — 两台修复: A互补并集池(检索层) B分片入座(窗口层)
A: FINAL-150 ∪ colbert-150 ∪ top50邻句 池通关率
B: 池内MMR/簇代表选座 vs 纯分数top5 转化率
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry22_results.txt"
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
                 return_sparse=False, return_colbert_vecs=True)
P("colbert ready %.0fs" % (time.time() - t0))

K = (5, 10)
res = {"A_基线FINAL": [0]*2, "A_互补并集": [0]*2, "B_FINALtop5": [0]*2, "B_分片入座": [0]*2}
n = 0
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
        continue
    n += 1
    qv = np.asarray(enc["colbert_vecs"][k_i], dtype=np.float16)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    qt = torch.from_numpy(qv).to("cuda")
    s = (qt @ Pm.T).cpu().numpy()
    colb = np.maximum.reduceat(s, OFF[:, 0], axis=1).mean(axis=0)
    forder = list(np.argsort(-FINAL[qi]))
    corder = list(np.argsort(-colb))
    # ===== A 互补并集池: FINAL-150 ∪ colbert-150 ∪ FINAL-top50的邻句 =====
    poolA = set(forder[:150]) | set(corder[:150])
    for i in forder[:50]:
        for a2 in (i - 1, i + 1):
            if 0 <= a2 < NR:
                poolA.add(a2)
    for k_j, k in enumerate((150, 250)):
        if all(r in poolA for r in G):
            res["A_互补并集"][k_j] += 1
    for k_j, k in enumerate((150, 250)):
        if all(r in set(forder[:150]) | set(corder[:150]) for r in G):
            res["A_基线FINAL"][k_j] += 1
    # ===== B 分片入座(在FINAL-200池内, raw记录做MMR多样性选5座) =====
    rawpool = [i for i in np.argsort(-FINAL[qi])[:200] if ISRAW[i]]
    golds_in = G
    # 纯分数top5
    top5 = rawpool[:5]
    for k_j, k in enumerate((5, 10)):
        if all(r in top5[:k] for r in G):
            res["B_FINALtop5"][k_j] += 1
    # MMR分片: 5席, 每席选与已选记录最不相似的最高FINAL分记录
    seats = []
    cand = list(rawpool)
    fv = D[cand]
    while len(seats) < 5 and cand:
        if not seats:
            best = cand[0]
        else:
            bestc, bestv = None, -1e9
            for c in cand:
                mn = min(float(D[c] @ D[s2]) for s2 in seats)
                v = float(FINAL[qi][c]) / 50.0 + mn
                if v > bestv:
                    bestv, bestc = v, c
            best = bestc
        seats.append(best)
        cand.remove(best)
    for k_j, k in enumerate((5, 10)):
        if all(r in seats[:k] if k <= 5 else r in seats + rawpool[5:k] for r in G):
            res["B_分片入座"][k_j] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== A. 检索层池通关(池150/250) =====")
for nm in ("A_基线FINAL", "A_互补并集"):
    a, b = res[nm]
    P("%-14s all-gold@150=%.1f%% @250=%.1f%%" % (nm, 100.0*a/max(1,n), 100.0*b/max(1,n)))
P("\n===== B. 窗口层入座(raw座) =====")
for nm in ("B_FINALtop5", "B_分片入座"):
    a, b = res[nm]
    P("%-14s all-gold@5=%.1f%% @10(raw序)=%.1f%%" % (nm, 100.0*a/max(1,n), 100.0*b/max(1,n)))
P("FOUNDRY22_DONE %.0fs" % (time.time() - t0))
LOG.close()
