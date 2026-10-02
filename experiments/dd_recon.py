import io, json, os, sys, re
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = "C:/locomo_refined/memsys"

mids = []
with open(os.path.join(base, "mem_bge_sparse.jsonl"), encoding="utf-8") as f:
    for line in f:
        mids.append(json.loads(line)["mid"])
print("A_sparse_n", len(mids))

D = np.load(os.path.join(base, "mem_bge_dense.npz"))
print("B_files", D.files)
arr = D[D.files[0]]
print("B_shape", arr.shape, arr.dtype, round(float(np.linalg.norm(arr[:5], axis=1).mean()), 4))

recs = [json.loads(l) for l in open(os.path.join(base, "mem.jsonl"), encoding="utf-8")]
print("C_n", len(recs))
print("C_keys", ",".join(sorted(recs[0].keys())))
print("C_kinds", dict(Counter(r.get("kind") for r in recs)))
print("C_twin", sum(1 for r in recs if r.get("twin_of")), "raw_of", sum(1 for r in recs if r.get("raw_of")))
print("C_nsess", len(set(r.get("session_id") for r in recs)))
midset = set(mids)
print("C_memid_in_sparse", sum(1 for r in recs if r.get("memory_id") in midset))
print("C_sids_sample", sorted(set(r.get("session_id") for r in recs))[:3])
print("C_mids_sample", [r.get("memory_id") for r in recs[:3]])

F = np.load(os.path.join(base, "fusion_cache.npz"))
for k in F.files:
    print("D", k, F[k].shape, F[k].dtype)

R = np.load(os.path.join(base, "rerank_stage1.npz"))
for k in R.files:
    print("E", k, R[k].shape, R[k].dtype)

o = json.load(open(os.path.join(base, "out_r37.json"), encoding="utf-8"))
print("F_type", type(o).__name__)
if isinstance(o, dict):
    print("F_topkeys", ",".join(list(o.keys())[:8]))
qlist = o if isinstance(o, list) else o.get("results", o.get("questions"))
print("F_nq", len(qlist))
print("F_qkeys", ",".join(sorted(qlist[0].keys())))
print("F_scores", dict(Counter(x.get("llm_score") for x in qlist)))
print("F_evlen", dict(Counter(len(x.get("evidence_messages") or []) for x in qlist)))
em = (qlist[0].get("evidence_messages") or [{}])[0]
print("F_em_keys", ",".join(sorted(em.keys())))
r0 = recs[0]
for k in sorted(r0.keys()):
    v = r0[k]
    s = str(v)
    print("G", k, type(v).__name__, s[:60].encode("ascii", "replace").decode())
print("DONE")
