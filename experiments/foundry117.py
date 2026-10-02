# -*- coding: utf-8 -*-
"""foundry117.py — 问题驱动的动态窗口(零标签词面规则, 盲调):
问题含聚合词(how many/total/all/each/数字)→大窗 | 单点词(what/who哪一个)→小窗
验证: turn级全金对错(总分)"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry117_results.txt", "w", encoding="utf-8")
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
P("loaded %.0fs" % (time.time() - t0))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

AGG_PAT = re.compile(r"\bhow many\b|\bhow much\b|\btotal\b|\ball\b|\bevery\b|\beach\b|\bboth\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bsome\b|\blist\b", re.I)

def window_of(qtext):
    return 20 if AGG_PAT.search(qtext) else 8

# 三臂: 固定5 / 固定15 / 动态(词面)
res = {"f5": 0, "f15": 0, "dyn": 0}
n = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    n += 1
    cq = D @ X[k_i]
    order = np.argsort(-cq)
    o5 = set(order[:5].tolist())
    o15 = set(order[:15].tolist())
    kw = window_of(Q[qa]["question"])
    odyn = set(order[:kw].tolist())
    if G <= o5:
        res["f5"] += 1
    if G <= o15:
        res["f15"] += 1
    if G <= odyn:
        res["dyn"] += 1

P("===== 问题驱动动态窗口(零标签, n=%d) =====" % n)
P("固定@5=%.1f%%  固定@15=%.1f%%  动态(5/20)=%.1f%%" % (
    100.0 * res["f5"] / n, 100.0 * res["f15"] / n, 100.0 * res["dyn"] / n))
P("F117_DONE %.0fs" % (time.time() - t0))
