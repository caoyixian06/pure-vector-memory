# -*- coding: utf-8 -*-
"""foundry70g.py — 会话分域 + rbak去重 组合效应: all-gold@K 学长尺子"""
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
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
def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR = {}
seen = {}
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR[i] = seen[k]
        PAIR[seen[k]] = i
    else:
        seen[k] = i
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]

def gold_groups(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN[i])
        if not hits:
            continue
        for i in list(hits):
            t = PAIR.get(i)
            if t is not None:
                hits.add(t)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups

GS = {}
for qa in IDS:
    grps = gold_groups(qa)
    if grps:
        GS[qa] = grps
nG = len(GS)

def dedup(order):
    out, used = [], set()
    for i in order:
        p = PAIR.get(i)
        if p is not None and p in used:
            continue
        out.append(i)
        used.add(i)
    return out

# 检查: 金是否全在本会话(分域合法性) — conv-26 <-> loco-conv-26 前缀对齐
QA2CONV = {qa: "loco-" + qa.split("#")[0] for qa in GS}
inconv = sum(1 for qa, grps in GS.items()
             if all(CONVKEY[g] == QA2CONV[qa] for grp in grps for g in grp))
P("金全部在本会话的题: %d/%d (%.1f%%)" % (inconv, nG, 100.0 * inconv / nG))

for tag, fn in (("v7(F68版)", "r41_final_order_v7.json"), ("v8_clean", "r41_final_order_v8_clean.json")):
    FORD = json.load(open(HERE + "/" + fn, encoding="utf-8"))
    SEQ = {qa: [MID2I[m] for m in FORD[qa]] for qa in FORD}
    P("\n[%s]" % tag)
    for conv_f, ded_f, dtag in ((False, False, "原序        "),
                                (False, True,  "+rbak去重    "),
                                (True,  False, "+会话分域    "),
                                (True,  True,  "+分域+去重   ")):
        row = []
        for K in (3, 5, 10, 15, 35):
            cnt = 0
            for qa, grps in GS.items():
                seq = SEQ[qa]
                if conv_f:
                    seq = [i for i in seq if CONVKEY[i] == QA2CONV[qa]]
                if ded_f:
                    seq = dedup(seq)
                if all(any(g in seq[:K] for g in grp) for grp in grps):
                    cnt += 1
            row.append("@%d=%.1f%%" % (K, 100.0 * cnt / nG))
        P("  %s: %s" % (dtag, "  ".join(row)))
