# -*- coding: utf-8 -*-
import io, json, re
import numpy as np
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
MID2I = {m: i for i, m in enumerate(MID)}
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
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
FORD = json.load(open(HERE + "/r41_final_order_v7.json", encoding="utf-8"))
OUT = {}
for qa, seq in FORD.items():
    myconv = "loco-" + qa.split("#")[0]
    idxs = [MID2I[m] for m in seq if CONVKEY[MID2I[m]] == myconv]
    out, used = [], set()
    for i in idxs:
        p = PAIR.get(i)
        if p is not None and p in used:
            continue
        out.append(i); used.add(i)
    OUT[qa] = [MID[i] for i in out]
json.dump(OUT, open(HERE + "/r41_final_order_v9.json", "w", encoding="utf-8"), ensure_ascii=False)
lens = [len(v) for v in OUT.values()]
print("v9落盘: 题数=%d 均长=%.1f min=%d max=%d" % (len(OUT), np.mean(lens), min(lens), max(lens)))
