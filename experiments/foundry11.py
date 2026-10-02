# -*- coding: utf-8 -*-
"""foundry11.py — exact尺基线: 所有金证据turn进top5(原文命中, 纯向量)
口径: 全部金证据记录的排名都<=K。C0纯cos 与 FINAL 各测一版, 附单金/多金拆分。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry11_results.txt"
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
C0 = l2n(X @ D.T)
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
P("loaded %.0fs" % (time.time() - t0))

KS = (5, 10, 30, 50)
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

cnt = {"c0": {("all", k): 0 for k in KS}, "final": {("all", k): 0 for k in KS}}
cnt["c0"].update({("any", k): 0 for k in KS})
cnt["final"].update({("any", k): 0 for k in KS})
n = 0
single = {"c0_5": 0, "final_5": 0, "n": 0}
multi = {"c0_5": 0, "final_5": 0, "n": 0}
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    n += 1
    qi = IDX[qa]
    order_c = np.argsort(-C0[qi])
    rank_c = np.empty(NR, dtype=np.int32)
    rank_c[order_c] = np.arange(NR)
    order_f = np.argsort(-FINAL[qi])
    rank_f = np.empty(NR, dtype=np.int32)
    rank_f[order_f] = np.arange(NR)
    granks_c = sorted(rank_c[i] + 1 for i in G)
    granks_f = sorted(rank_f[i] + 1 for i in G)
    for k in KS:
        if granks_c[-1] <= k:
            cnt["c0"][("all", k)] += 1
        if granks_f[-1] <= k:
            cnt["final"][("all", k)] += 1
        if granks_c[0] <= k:
            cnt["c0"][("any", k)] += 1
        if granks_f[0] <= k:
            cnt["final"][("any", k)] += 1
    bucket = single if len(G) == 1 else multi
    bucket["n"] += 1
    if granks_c[-1] <= 5:
        bucket["c0_5"] += 1
    if granks_f[-1] <= 5:
        bucket["final_5"] += 1
    if k_i % 400 == 0:
        P("  scan %d %.0fs" % (k_i, time.time() - t0))

P("\n===== exact尺: 全部金证据turn进topK (n=%d) =====" % n)
for nm in ("c0", "final"):
    P("[%s]" % ("纯cos" if nm == "c0" else "FINAL"))
    for k in KS:
        P("  all-gold@%-3d = %.1f%%   (any-gold@%d = %.1f%%)" % (
            k, 100.0 * cnt[nm][("all", k)] / max(1, n), k, 100.0 * cnt[nm][("any", k)] / max(1, n)))
P("\n单金题: n=%d | 纯cos all@5=%.1f%% | FINAL all@5=%.1f%%" % (
    single["n"], 100.0 * single["c0_5"] / max(1, single["n"]), 100.0 * single["final_5"] / max(1, single["n"])))
P("多金题: n=%d | 纯cos all@5=%.1f%% | FINAL all@5=%.1f%%" % (
    multi["n"], 100.0 * multi["c0_5"] / max(1, multi["n"]), 100.0 * multi["final_5"] / max(1, multi["n"])))
P("FOUNDRY11_DONE %.0fs" % (time.time() - t0))
LOG.close()
