# -*- coding: utf-8 -*-
"""lme_qemb.py — LME 500题问题嵌入: bge1024(GPU) + qwen256(ollama) — 直迁原料"""
import io, json, os, sys, time, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
d = json.load(open(r"C:\locomo_refined\longmemeval\longmemeval_s_cleaned.json", encoding="utf-8"))
QTEXT = [q["question"] for q in d]
QIDS = [q["question_id"] for q in d]
print("questions:", len(QTEXT))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
out = bge.encode(QTEXT)
X1024 = l2n(np.asarray(out["dense_vecs"], dtype=np.float32))
np.save(r"C:\locomo_refined\longmemeval_mem\q_bge1024.npy", X1024)
print("bge1024:", X1024.shape, "%.0fs" % (time.time() - t0))

def oemb(texts, dims=256):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest", "input": texts[s:s + 64],
                           "dimensions": dims}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                break
            except Exception as e:
                print("retry", attempt + 1, str(e)[:60])
                if attempt == 3:
                    raise
                time.sleep(8 * (attempt + 1))
    return np.concatenate(out)

try:
    Q256 = l2n(oemb(QTEXT))
    np.save(r"C:\locomo_refined\longmemeval_mem\q_qwen256.npy", Q256)
    print("qwen256:", Q256.shape, "%.0fs" % (time.time() - t0))
except Exception as e:
    print("qwen256 FAILED:", str(e)[:120])
json.dump(QIDS, open(r"C:\locomo_refined\longmemeval_mem\q_ids.json", "w"))
print("DONE %.0fs" % (time.time() - t0))
