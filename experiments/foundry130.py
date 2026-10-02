# -*- coding: utf-8 -*-
import io, json, re, sys, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry130_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True); LOG.write(str(s)+chr(10)); LOG.flush()
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
t0 = time.time()
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip(): continue
    try: REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception: pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw","text","content"):
        if isinstance(r.get(f), str) and r[f].strip(): return r[f]
    return ""
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l); Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))
NR = len(MID)
rng = np.random.RandomState(0)
samp = rng.choice(NR, size=min(20000, NR), replace=False)
CENT = l2n(D[samp].mean(0, keepdims=True))[0]
Dr = l2n(D - np.outer(D @ CENT, CENT))
Xr = l2n(X - np.outer(X @ CENT, CENT))
C = np.cov(D[samp].T.astype(np.float64))
evals, evecs = np.linalg.eigh(C)
keep = evals > max(1e-8, evals.max() * 1e-4)
W = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
Dw = l2n((D @ W).astype(np.float32)); Xw = l2n((X @ W).astype(np.float32))
P("bases ready %.0fs white_dim=%d" % (time.time()-t0, Dw.shape[1]))
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or []) if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(NR) if any(k in RAWN[i] for k in keys))
BASES = {"cos": (D, X), "resid": (Dr, Xr), "white": (Dw, Xw)}
rs = {k: 0 for k in BASES}; rt = {k: 0 for k in BASES}
n = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G: continue
    n += 1
    gc = CONVKEY[list(G)[0]]
    for name, (Db, Xb) in BASES.items():
        cq = Db @ Xb[k_i]
        sm = {}
        for cv in UCONVS:
            m2 = CONVKEY == cv
            if m2.any(): sm[cv] = float(cq[m2].max())
        order = sorted(sm, key=lambda c: -sm[c])[:5]
        if gc in set(order): rs[name] += 1
        rows = np.where((CONVKEY == order[0]) & np.array(["_rbak" not in MID[i] for i in range(NR)]))[0]
        if len(rows) >= 5:
            rr = rows[np.argsort(-cq[rows])[:5]]
            if G & set(int(x) for x in rr): rt[name] += 1
P("n=%d" % n)
for name in BASES:
    P("%-6s sess@5=%.1f%% turn@5=%.1f%%" % (name, 100.0*rs[name]/n, 100.0*rt[name]/n))
P("F130_DONE %.0fs" % (time.time()-t0))
