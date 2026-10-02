# -*- coding: utf-8 -*-
"""foundry17a.py — M3完全体索引构建: 11753条turn的colbert多向量(fp16分块)+sparse权重
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
OUT = HERE + "/foundry17a_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
SENTS = [rec_raw(m) for m in MID]
P("turns=%d %.0fs" % (len(SENTS), time.time() - t0))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

os.makedirs(HERE + "/colbert_parts", exist_ok=True)
CH = 1000
nch = (len(SENTS) + CH - 1) // CH
done = set()
if os.path.exists(HERE + "/_colbert_done.txt"):
    done = {int(x) for x in open(HERE + "/_colbert_done.txt").read().split() if x}
ALLT = []
for ci in range(nch):
    if ci in done:
        # 已完成的块: 读回token数用于偏移表
        _p = np.load(HERE + "/colbert_parts/c%04d.npz" % ci)
        ALLT.extend([int(x) for x in _p["counts"]])
        continue
    chunk = SENTS[ci * CH:(ci + 1) * CH]
    enc = bge.encode(chunk, return_dense=False, return_sparse=True, return_colbert_vecs=True)
    cvecs = enc["colbert_vecs"]
    sp = enc["lexical_weights"]
    counts = [cv.shape[0] for cv in cvecs]
    mat = np.zeros((sum(counts), cvecs[0].shape[1]), dtype=np.float16)
    off = 0
    for cv in cvecs:
        mat[off:off + cv.shape[0]] = cv.astype(np.float16)
        off += cv.shape[0]
    np.savez_compressed(HERE + "/colbert_parts/c%04d.npz" % ci, mat=mat, counts=np.array(counts))
    with open(HERE + "/colbert_parts/s%04d.pkl" % ci, "wb") as f:
        pickle.dump(sp, f)
    done.add(ci)
    ALLT.extend(counts)
    open(HERE + "/_colbert_done.txt", "w").write(" ".join(map(str, sorted(done))))
    P("chunk %d/%d tokens=%d %.0fs" % (ci + 1, nch, sum(counts), time.time() - t0))
P("COLBERT_INDEX_DONE chunks=%d total_tokens=%d %.0fs" % (nch, sum(ALLT), time.time() - t0))
LOG.close()
