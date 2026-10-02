# -*- coding: utf-8 -*-
"""probe256.py — 判别q0000块的生产模型: bge-m3截断256 vs qwen3-embedding"""
import json, urllib.request
import numpy as np

OUT = "C:/locomo_refined/longmemeval_mem"
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"

q0 = np.load(OUT + "/_v256_parts/q0000.npy")
print("q0000 shape:", q0.shape)

# 源json第一条会话的前几条消息文本
data = json.load(open(SRC, encoding="utf-8"))
q0s = data[0]
turns = q0s["haystack_sessions"][0]
prefix = "user: " if (turns[0].get("role") or "user") == "user" else "assistant: "
cands = [(prefix + str(turns[0].get("content") or "").strip())[:6000]]
if len(turns) > 1:
    p2 = "user: " if (turns[1].get("role") or "user") == "user" else "assistant: "
    cands.append((p2 + str(turns[1].get("content") or "").strip())[:6000])
print("cands:", len(cands))
for t in cands[:3]:
    print("  text:", t[:70].replace("\n", " "))

def oemb(model, text, dims=256):
    body = json.dumps({"model": model, "input": [text], "dimensions": dims}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return np.asarray(json.loads(r.read())["embeddings"][0], dtype=np.float32)

def l2n(x):
    return x / max(np.linalg.norm(x), 1e-9)

for t in cands[:3]:
    v_q = l2n(oemb("qwen3-embedding:latest", t))
    sims = q0 @ v_q
    print("qwen3-emb(now) max-cos vs q0000: %.4f (argmax=%d)" % (sims.max(), int(sims.argmax())))
