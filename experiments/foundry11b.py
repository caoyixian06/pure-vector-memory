# -*- coding: utf-8 -*-
"""foundry11b.py — 判定变体全扫描: 同一系统在不同"金证据计数法"下的all-gold@5
V1 记录级: 每个含证据文本的记录都要进top5(重复副本各自要进)
V2 片段级: 每个证据片段(evidence_messages每条)至少有一个top5记录携带
V4 孪生级: 记录或其孪生(twin/raw_of)在top5即算
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry11b_results.txt"
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
TWIN = {}
MID2I = {m: i for i, m in enumerate(MID)}
for i, m in enumerate(MID):
    r = REC.get(m, {})
    tw = r.get("raw_of") or r.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
C0 = l2n(X @ D.T)
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
P("loaded %.0fs" % (time.time() - t0))

KS = (5, 10)
V = {"v1_final": [0]*2, "v2_final": [0]*2, "v4_final": [0]*2,
     "v1_c0": [0]*2, "v2_c0": [0]*2, "v4_c0": [0]*2}
n = 0
pieces_hist = {}
for qa in IDS:
    q = Q[qa]
    keys = [norm(em.get("text") or "")[:60] for em in (q.get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    if not keys:
        continue
    n += 1
    pieces_hist[len(keys)] = pieces_hist.get(len(keys), 0) + 1
    qi = IDX[qa]
    grecs = [i for i in range(NR) if any(kk in RAWN[i] for kk in keys)]
    for rank_name, order in (("final", np.argsort(-FINAL[qi])), ("c0", np.argsort(-C0[qi]))):
        for k_i, k in enumerate(KS):
            top = order[:k]
            topn = [RAWN[i] for i in top]
            # V1 记录级
            v1 = all(i in set(top) for i in grecs)
            # V2 片段级
            v2 = all(any(kk in tn for tn in topn) for kk in keys)
            # V4 孪生级
            gexp = set(grecs)
            for i in grecs:
                if i in TWIN:
                    gexp.add(TWIN[i])
            v4 = all(i in set(top) for i in gexp)
            V["v1_" + rank_name][k_i] += v1
            V["v2_" + rank_name][k_i] += v2
            V["v4_" + rank_name][k_i] += v4
    if n % 300 == 0:
        P("  scan %d %.0fs" % (n, time.time() - t0))

P("\n证据片段数分布: " + str(dict(sorted(pieces_hist.items()))))
P("\n===== 判定变体 all-gold@K (n=%d) =====" % n)
for nm in ("v1_final", "v2_final", "v4_final", "v1_c0", "v2_c0", "v4_c0"):
    P("%-10s @5=%.1f%%  @10=%.1f%%" % (nm, 100.0 * V[nm][0] / max(1, n), 100.0 * V[nm][1] / max(1, n)))
P("FOUNDRY11B_DONE %.0fs" % (time.time() - t0))
LOG.close()
