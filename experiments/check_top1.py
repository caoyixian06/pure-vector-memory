# -*- coding: utf-8 -*-
"""check_top1.py — 精排第1名是证据还是噪声? 分EVID/TWIN/邻句/NOISE, 按对错题分层"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

from collections import Counter
stat_all, stat_ok, stat_mi = Counter(), Counter(), Counter()
ev_rank_when_noise = []
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if not hits:
        continue
    hs = set(hits)
    tws = {TWIN.get(h) for h in hits} - {None}
    adj = set()
    for h in hits:
        for d in (1, -1):
            k = h + d
            if 0 <= k < N and CONVKEY[k] == CONVKEY[h]:
                adj.add(k)
    order = TOP50[i][np.argsort(-RER[i])]
    top1 = order[0]
    if top1 in hs:
        lab = "EVID"
    elif top1 in tws:
        lab = "TWIN"
    elif top1 in adj:
        lab = "邻句"
    else:
        lab = "NOISE"
        # 噪声当第1名时,证据排第几(精排序内)
        pos = {j: p for p, j in enumerate(order)}
        rk = min((pos[h] + 1) for h in hs if h in pos) if any(h in pos for h in hs) else None
        ev_rank_when_noise.append(rk)
    stat_all[lab] += 1
    (stat_ok if r.get("llm_score") == 1 else stat_mi)[lab] += 1

tot = sum(stat_all.values())
print("第1名身份分布(全体%d):" % tot)
for k in ("EVID", "TWIN", "邻句", "NOISE"):
    print("  %-4s %4d (%.0f%%) | 对题 %d 错题 %d" % (
        k, stat_all[k], 100 * stat_all[k] / tot, stat_ok[k], stat_mi[k]))
rn = [x for x in ev_rank_when_noise if x]
print("第1名是噪声时,真证据在精排序中的名次: 中位%.0f p75=%.0f(50名外=%d题)" % (
    np.median(rn), np.percentile(rn, 75), sum(1 for x in ev_rank_when_noise if x is None)))
