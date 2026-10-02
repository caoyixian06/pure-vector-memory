# -*- coding: utf-8 -*-
"""foundry70b.py — 池大小统计: fc(F68原版) vs clean(C0+256双通道) 实际候选池分布"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

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
X = np.load(HERE + "/xz_cache.npz")["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QWQ256 = l2n(Q256 @ QW.T)
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
POOLSZ = 700
NALL = len(IDS)

def build_pool(k_i, mode):
    if mode == "fc":
        pool = set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-C0[k_i])[:250])
        top30 = list(np.argsort(-FINAL[k_i])[:30])
    else:
        pool = set(np.argsort(-C0[k_i])[:350]) | set(np.argsort(-QWQ256[k_i])[:350])
        top30 = list(np.argsort(-C0[k_i])[:30])
    for i in top30:
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    return sorted(pool)[:POOLSZ]

for mode, tag in (("fc", "fc池(FINAL250∪C0250+邻域)"), ("clean", "clean池(C0350∪256ch350+邻域)")):
    sizes = np.array([len(build_pool(k, mode)) for k in range(NALL)])
    P("%s: mean=%.0f  min=%d  p25=%.0f  median=%.0f  p75=%.0f  max=%d  顶满700的题数=%d/%d" % (
        tag, sizes.mean(), sizes.min(), np.percentile(sizes, 25), np.median(sizes),
        np.percentile(sizes, 75), sizes.max(), (sizes >= 700).sum(), NALL))
P("total %.0fs NR=%d" % (time.time() - t0, NR))
