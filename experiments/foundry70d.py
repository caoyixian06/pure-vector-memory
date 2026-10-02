# -*- coding: utf-8 -*-
"""foundry70d.py — 口径对齐: evidence原始结构 + 孪生折叠all-gold@K"""
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
MID2I = {m: i for i, m in enumerate(MID)}
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
TWIN = {}
for i, m in enumerate(MID):
    r = REC.get(m, {})
    tw = r.get("raw_of") or r.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]

# 1) evidence原始结构
sample = [qa for qa in IDS[:200] if Q[qa].get("evidence_messages")][0]
P("evidence样例[%s]:" % sample)
for em in Q[sample]["evidence_messages"][:3]:
    P("  keys=%s" % sorted(em.keys()))
    P("  " + json.dumps(em, ensure_ascii=False)[:220])

# 2) 金集合: 展开 / 孪生折叠 两种
def gold_groups(qa):
    """返回[组列表], 每组=set(记录索引) — 按孪生折叠; 组=一条真证据"""
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN[i])
        if not hits:
            continue
        # 孪生折叠: hits内的互为孪生成员并成一组; 无孪生关系但同文本也并成组(同证据多副本)
        merged = set(hits)
        for i in list(hits):
            t = TWIN.get(i)
            if t is not None:
                merged.add(t)
        # 若该key与已有组重叠则并入
        for g in groups:
            if g & merged:
                g |= merged
                break
        else:
            groups.append(merged)
    return groups

GS_flat = {}
GS_fold = {}
for qa in IDS:
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    flat = set()
    for k in keys:
        flat |= set(i for i in range(len(MID)) if k in RAWN[i])
    if flat:
        GS_flat[qa] = flat
    grps = gold_groups(qa)
    if grps:
        GS_fold[qa] = grps
nG = len(GS_flat)
P("\nnG=%d  flat均片=%.2f  fold均组=%.2f" % (
    nG,
    np.mean([len(v) for v in GS_flat.values()]),
    np.mean([len(v) for v in GS_fold.values()])))
P("fold组数分布: " + str({k: sum(1 for v in GS_fold.values() if len(v) == k) for k in range(1, 9)}))

for tag, fn in (("v7(F68版)", "r41_final_order_v7.json"), ("v8_clean", "r41_final_order_v8_clean.json")):
    FORD = json.load(open(HERE + "/" + fn, encoding="utf-8"))
    P("\n[%s]" % tag)
    for K in (3, 5, 10, 15, 35):
        flat_k = sum(1 for qa, G in GS_flat.items() if all(MID[g] in FORD[qa][:K] for g in G))
        fold_k = sum(1 for qa, grps in GS_fold.items()
                     if all(any(MID[g] in FORD[qa][:K] for g in grp) for grp in grps))
        P("  @%-2d: flat=%5.1f%%  孪生折叠=%5.1f%%" % (K, 100.0 * flat_k / nG, 100.0 * fold_k / nG))
