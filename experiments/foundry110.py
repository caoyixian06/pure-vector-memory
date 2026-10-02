# -*- coding: utf-8 -*-
"""foundry110.py — 公式通用性终极检验: LME最优公式(max×1024会话聚合)直搬LoCoMo
零训练零标注, 只看总分对错"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry110_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))
P("loaded %.0fs 库会话数=%d" % (time.time() - t0, len(UCONVS)))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

res = {"s1": 0, "s5": 0, "n": 0}
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    gold_conv = CONVKEY[list(G)[0]]
    cq = D @ X[k_i]
    sess_score = {}
    for cv in UCONVS:
        mask = CONVKEY == cv
        sess_score[cv] = float(cq[mask].max())
    order = sorted(sess_score, key=lambda c: -sess_score[c])
    res["n"] += 1
    if order[0] == gold_conv:
        res["s1"] += 1
    if gold_conv in set(order[:5]):
        res["s5"] += 1
    if k_i % 400 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))

P("\n===== LME公式直搬LoCoMo(会话级, n=%d) =====" % res["n"])
P("金会话@1=%.1f%%  @5=%.1f%%  (库=%d会话选1)" % (
    100.0 * res["s1"] / res["n"], 100.0 * res["s5"] / res["n"], len(UCONVS)))
P("对照LME: 89.8@5/97.8@15")
P("F110_DONE %.0fs" % (time.time() - t0))
