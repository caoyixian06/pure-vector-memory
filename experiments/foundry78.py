# -*- coding: utf-8 -*-
"""foundry78.py — @5按金组数分层 + what题×组数交叉"""
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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v10.json", encoding="utf-8"))

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

layer_tot = {}
layer_ok = {}
cross_tot = {}
cross_ok = {}
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    ng = min(len(grps), 4)
    first = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    key = "1组" if ng == 1 else ("%d组" % ng)
    layer_tot[key] = layer_tot.get(key, 0) + 1
    cross_tot[(first, key)] = cross_tot.get((first, key), 0) + 1
    if all(any(g in seq[:5] for g in grp) for grp in grps):
        layer_ok[key] = layer_ok.get(key, 0) + 1
        cross_ok[(first, key)] = cross_ok.get((first, key), 0) + 1

P("按金组数分层@5:")
for k in sorted(layer_tot, key=lambda x: int(x[0])):
    P("  %-3s %5.1f%%  (%d/%d)" % (k, 100.0 * layer_ok.get(k, 0) / layer_tot[k], layer_ok.get(k, 0), layer_tot[k]))
P("\nwhat/which/when × 组数交叉@5:")
for first in ("what", "which", "when", "how"):
    row = []
    for k in ("1组", "2组", "3组", "4组"):
        t = cross_tot.get((first, k), 0)
        o = cross_ok.get((first, k), 0)
        row.append("%s=%5.1f%%(%d)" % (k, 100.0 * o / t if t else 0, t))
    P("  %-6s %s" % (first, "  ".join(row)))
