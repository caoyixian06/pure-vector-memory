# -*- coding: utf-8 -*-
import json, io
MID = [json.loads(l)["mid"] for l in io.open(r"C:\locomo_refined\memsys\mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(r"C:\locomo_refined\memsys\mem.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            REC[json.loads(l).get("memory_id")] = json.loads(l)
        except Exception:
            pass

def rr(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""

RL = [rr(m) for m in MID]
print("total", len(RL))
print("hdr_startswith", sum(1 for r in RL if r.startswith("[Session")))
print("empty", sum(1 for r in RL if not r))
print("sample0", repr(RL[0][:50]))
print("sample5", repr(RL[5][:50]))
print("sess_in_15", sum(1 for r in RL if "Session" in r[:15]))
