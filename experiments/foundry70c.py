# -*- coding: utf-8 -*-
"""foundry70c.py — 学长同尺对照: all-gold@K (K=1/3/5/10/15/35), v7(F68) vs v8_clean"""
import io, json, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

GS = {}
for qa in IDS:
    G = gold_set(qa)
    if G:
        GS[qa] = [MID[g] for g in G]
nG = len(GS)
P("nG=%d / %d (空金%d)" % (nG, len(IDS), len(IDS) - nG))
P("金片数分布: " + str({k: sum(1 for v in GS.values() if len(v) == k) for k in range(1, 9)}))

for tag, fn in (("v7(F68版)", "r41_final_order_v7.json"), ("v8_clean(零FINAL)", "r41_final_order_v8_clean.json")):
    FORD = json.load(open(HERE + "/" + fn, encoding="utf-8"))
    P("\n[%s] %s" % (tag, fn))
    for K in (1, 3, 5, 10, 15, 35):
        allk = sum(1 for qa, gm in GS.items() if all(m in FORD[qa][:K] for m in gm))
        anyk = sum(1 for qa, gm in GS.items() if any(m in FORD[qa][:K] for m in gm))
        P("  all-gold@%-2d: %5.1f%% (nG) / %5.1f%% (1382)   any@%-2d: %5.1f%%" % (
            K, 100.0 * allk / nG, 100.0 * allk / len(IDS), K, 100.0 * anyk / nG))
