# -*- coding: utf-8 -*-
"""r22_backfill.py - 给 memsys/mem.jsonl 存量记录补 R22 原句伴生记录(kind="raw")。
合格: 非 "[Session" 头、len>=15、尚无伴生。id 用 _rbak 前缀避免与 write 端 _m 号冲突。"""
import json, os, sys
import numpy as np

HERE = "C:/locomo_refined/memsys"
os.environ["MEM_FILE"] = HERE + "/mem.jsonl"
sys.path.insert(0, HERE)
import sessionmem
print("version:", sessionmem.SESSIONMEM_VERSION, "raw_index:", sessionmem.RAW_INDEX, flush=True)
assert sessionmem.RAW_INDEX, "MEM_RAW_INDEX=off, abort"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

path = HERE + "/mem.jsonl"
lines = [l for l in open(path, encoding="utf-8") if l.strip()]
recs, bad = [], 0
for l in lines:
    try:
        recs.append(json.loads(l))
    except Exception:
        bad += 1
has_comp = set(r.get("raw_of") for r in recs if r.get("kind") == "raw")
todo, skipped_head, skipped_short, skipped_have = [], 0, 0, 0
for r in recs:
    raw = (r.get("raw") or "").strip()
    if raw.startswith("[Session") or not raw:
        skipped_head += 1
        continue
    if len(raw) < sessionmem.RAW_MIN_CHARS:
        skipped_short += 1
        continue
    if r.get("memory_id") in has_comp:
        skipped_have += 1
        continue
    todo.append(r)
print("total=%d todo=%d (head=%d short=%d have=%d badlines=%d)" %
      (len(recs), len(todo), skipped_head, skipped_short, skipped_have, bad), flush=True)

vecs = []
B = 64
for i in range(0, len(todo), B):
    vecs.extend(sessionmem.embed([r["raw"].strip() for r in todo[i:i + B]]))
    if (i // B) % 20 == 0:
        print("embedded", min(i + B, len(todo)), flush=True)
vecs = l2n(np.asarray(vecs, dtype=np.float32))

with open(path, "a", encoding="utf-8") as f:
    for r, v in zip(todo, vecs):
        f.write(json.dumps(dict(
            memory_id=r["memory_id"].replace("_m", "_rbak", 1) + "k", session_id=r["session_id"],
            timestamp=r.get("timestamp"), project=r.get("project"),
            participants=r.get("participants") or ["用户", "AI"],
            keywords=r.get("keywords") or [], order=r.get("order") or [],
            summary="", summary_source="rawidx", raw=r["raw"],
            vector=[round(float(x), 6) for x in v], confidence=0.85, activation=1.0,
            compression_ratio=None, kind="raw", raw_of=r.get("memory_id")),
            ensure_ascii=False) + chr(10))

n_after = sum(1 for l in open(path, encoding="utf-8") if l.strip())
n_raw = sum(1 for l in open(path, encoding="utf-8") if '"kind": "raw"' in l or '"kind":"raw"' in l)
print("BACKFILL_DONE before=%d added=%d after=%d raw_kind=%d" % (len(recs), len(todo), n_after, n_raw), flush=True)
assert n_after == len(recs) + len(todo), "line count mismatch"
print("GATE2_PASS", flush=True)
