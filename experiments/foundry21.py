# -*- coding: utf-8 -*-
"""foundry21.py — 分层通关计分板: 检索层(all-gold@200池) + 窗口层(池内金进5席转化率)
池: C0top200 / FINALtop200 / colberttop200 / 并集池。统一尺: raw金片段、原文turn命中。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry21_results.txt"
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

POOLSZ = 200
score = {"c0": {"pool": 0, "win5": 0}, "final": {"pool": 0, "win5": 0},
         "colbert": {"pool": 0, "win5": 0}, "union": {"pool": 0, "win5": 0},
         "union_dense": {"pool": 0, "win5": 0}}
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
    pools = {
        "c0": list(np.argsort(-C0[qi])[:POOLSZ]),
        "final": [i for i in np.argsort(-FINAL[qi])[:POOLSZ]],
        "colbert": list(np.argsort(-colb)[:POOLSZ]),
    }
    u = set(pools["c0"]) | set(pools["colbert"])
    pools["union"] = sorted(u, key=lambda i: -max(C0[qi][i], colb[i]))
    u2 = set(pools["c0"]) | set(pools["colbert"])
    pools["union_dense"] = sorted(u, key=lambda i: -max(C0[qi][i], colb[i], FINAL[qi][i]))
    for nm, pool in pools.items():
        inpool = all(i in set(pool) for i in G)
        if inpool:
            score[nm]["pool"] += 1
        # 窗口层: 池内按本池排序的raw记录顺序, 5席全装
        raworder = [i for i in pool if ISRAW[i]]
        rks = []
        pos = {i: r for r, i in enumerate(raworder)}
        for i in G:
            rks.append(pos.get(i, 10**9))
        if max(rks) <= 5:
            score[nm]["win5"] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== 分层通关计分板 (n=%d, 池=%d条, 金=raw片段) =====" % (n, POOLSZ))
P("%-10s %16s %20s" % ("池", "检索层 all-gold@池", "窗口层 金进5席转化"))
for nm in score:
    pr = 100.0 * score[nm]["pool"] / max(1, n)
    wr = 100.0 * score[nm]["win5"] / max(1, n)
    conv = 100.0 * score[nm]["win5"] / max(1, score[nm]["pool"])
    P("%-10s %13.1f%% %18.1f%% (转化=%.1f%%)" % (nm, pr, wr, conv))
P("FOUNDRY21_DONE %.0fs" % (time.time() - t0))
LOG.close()
