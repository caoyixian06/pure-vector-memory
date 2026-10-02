# -*- coding: utf-8 -*-
"""foundry83.py — 多终序RRF融合: v7/v8_clean/v9/v10 四套排序的互补性收割"""
import io, json, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w

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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]

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
        PAIR[i] = seen[k]; PAIR[seen[k]] = i
    else:
        seen[k] = i

def gold_groups(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN[i])
        if not hits:
            continue
        for i in list(hits):
            tt = PAIR.get(i)
            if tt is not None:
                hits.add(tt)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups

def core_group(qa, grps):
    ans = " ".join(str(x) for x in (Q[qa].get("answer") or []))
    aw = set(stem(w) for w in re.findall(r"[a-z]+", ans.lower()) if len(w) > 2)
    if not aw:
        return grps[0]
    best, bv = grps[0], -1.0
    for grp in grps:
        tw = set()
        for g in grp:
            tw |= set(stem(w) for w in re.findall(r"[a-z']+", RAW[g].lower()))
        cov = len(aw & tw) / len(aw)
        if cov > bv:
            bv, best = cov, grp
    return best

GS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
nG = len(GS)
FORDS = {}
for tag, fn in (("v7", "r41_final_order_v7.json"), ("v8c", "r41_final_order_v8_clean.json"),
                ("v9", "r41_final_order_v9.json"), ("v10", "r41_final_order_v10.json")):
    FORDS[tag] = json.load(open(HERE + "/" + fn, encoding="utf-8"))

def score(seq_dict_tags, k_core=5, k_all=15):
    all15 = core5 = 0
    for qa, grps in GS.items():
        fused = {}
        for tag in seq_dict_tags:
            seq = seq_dict_tags[tag].get(qa, [])
            for r, m in enumerate(seq):
                i = MID2I[m]
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])
        if all(any(g in order[:k_all] for g in grp) for grp in grps):
            all15 += 1
        cg = core_group(qa, grps)
        if any(g in order[:k_core] for g in cg):
            core5 += 1
    return 100.0 * all15 / nG, 100.0 * core5 / nG

for tags in (("v10",), ("v7", "v10"), ("v8c", "v10"), ("v7", "v8c", "v9", "v10"), ("v7", "v8c", "v10")):
    a15, c5 = score({t: FORDS[t] for t in tags})
    P("%-28s 全部@15=%5.1f%%  核心@5=%5.1f%%" % ("+".join(tags), a15, c5))
