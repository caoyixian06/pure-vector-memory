# -*- coding: utf-8 -*-
"""check_delta_u2.py — 证据指纹Δ与u2(陈述-问题方向)是不是同一个东西"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50 = z1["TOP50"]
evs, nos = [], []
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if not hits:
        continue
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hits if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hits or j in tws]
    nos += [j for j in top if j not in hits and j not in tws][:8]
delta = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
delta /= np.linalg.norm(delta)
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
# u2 = stmt - qc(qc来自缓存: fusion_cache无qc; 用陈述-摘要质心近似? 不行。
# 精确版: 问题质心需重嵌入——用 diag_dims.py 的口径: stmt - qc。这里退而求其次测两个替代:
# u2a = 陈述质心 - 库全局质心; u2b = raw记录质心 - summary记录质心(库内自带,无监督)
glob = D.mean(0); glob /= np.linalg.norm(glob)
u2a = stmt - glob
u2a /= np.linalg.norm(u2a)
sum_idx = [i for i in range(N) if KIND[i] != "raw"]
summ_c = D[sum_idx].mean(0); summ_c /= np.linalg.norm(summ_c)
u2b = stmt - summ_c
u2b /= np.linalg.norm(u2b)
print("cos(Δ, u2a=陈述-全局)   = %.3f" % float(delta @ u2a))
print("cos(Δ, u2b=陈述-摘要域) = %.3f" % float(delta @ u2b))
# 以及: Δ在noise子类上的投影(问句形/寒暄/陈述)
COUR = re.compile(r"\b(thank|thanks|congrat|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
qs_, cs_, ss_ = [], [], []
for j in nos[:3000]:
    t = TEXTS[j]
    if t.rstrip().endswith("?"):
        qs_.append(j)
    elif COUR.search(t):
        cs_.append(j)
    else:
        ss_.append(j)
for tag, ids in (("问句形噪声", qs_), ("寒暄噪声", cs_), ("陈述噪声", ss_)):
    if ids:
        print("Δ投影: %s = %.3f (证据=%.3f, 噪声总体=%.3f)" % (
            tag, float((D[np.array(ids)] @ delta).mean()),
            float((D[np.array(evs)] @ delta).mean()), float((D[np.array(nos)] @ delta).mean())))
