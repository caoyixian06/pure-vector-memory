# -*- coding: utf-8 -*-
"""bge_build.py - BGE-M3 重嵌入 LoCoMo 库(全库统一换域):
summary 域 → embed(summary 剥[Session]头) 的 dense(1024)+sparse
raw 域    → embed(raw 原句) 的 dense+sparse
落盘: mem_bge_dense.npz (2,N,1024 归一) + mem_bge_sparse.jsonl (每行 mid+sparse dict) + mem_bge_meta.jsonl"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("FLAGCSSL", "1")

import numpy as np
from FlagEmbedding import BGEM3FlagModel

HERE = "C:/locomo_refined/memsys"
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("bge-m3 loaded", flush=True)

recs = []
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        recs.append(json.loads(l))
    except Exception:
        continue
texts, mids, kinds = [], [], []
for r in recs:
    raw = (r.get("raw") or "").strip()
    if raw.startswith("[Session"):
        continue
    if r.get("kind") == "raw":
        txt = raw
    else:
        txt = (r.get("summary") or "").strip() or raw[:200]
    if not txt:
        continue
    texts.append(txt[:2000])
    mids.append(r.get("memory_id"))
    kinds.append(r.get("kind") or "summary")
print("to-embed:", len(texts), flush=True)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

D = []
S = []
B = 64
t0 = time.time()
for i in range(0, len(texts), B):
    out = model.encode(texts[i:i + B], return_dense=True, return_sparse=True,
                       return_colbert_vecs=False)
    D.append(l2n(np.asarray(out["dense_vecs"], dtype=np.float32)))
    S.extend(out["lexical_weights"])
    if (i // B) % 20 == 0:
        print("embedded", min(i + B, len(texts)), round(time.time() - t0, 1), "s", flush=True)
D = np.concatenate(D, axis=0)
np.savez_compressed(HERE + "/mem_bge_dense.npz", dense=D)
with open(HERE + "/mem_bge_sparse.jsonl", "w", encoding="utf-8") as f:
    for mid, k, sp in zip(mids, kinds, S):
        f.write(json.dumps(dict(mid=mid, kind=k, sparse={str(a): round(float(b), 5) for a, b in sp.items()}),
                           ensure_ascii=False) + chr(10))
with open(HERE + "/mem_bge_meta.jsonl", "w", encoding="utf-8") as f:
    for mid, k in zip(mids, kinds):
        f.write(json.dumps(dict(mid=mid, kind=k), ensure_ascii=False) + chr(10))
print("BGE_BUILD_DONE", D.shape, flush=True)
