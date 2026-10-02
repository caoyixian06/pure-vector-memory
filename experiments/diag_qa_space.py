# -*- coding: utf-8 -*-
"""diag_qa_space.py — 问题/答案/证据三角在两套向量空间的几何对比
每个空间算四个余弦:
  floor  = cos(问题, 随机别人的证据)   <- 本底
  q_ev   = cos(问题, 金证据句)
  q_a    = cos(问题, 金答案串)
  a_ev   = cos(金答案串, 金证据句)      <- 锚点(答案本就从证据抽出,应最高)
信号 = 各项相对本底的抬升; 分对题/错题看。
"""
import io, json, os, sys, time, random, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
print("rows with q/a/ev:", len(rows), flush=True)

Q  = [r["question"] for r in rows]
A  = [str(r["answer"][0]) for r in rows]
EV = [r["evidence_messages"][0]["text"] for r in rows]
n = len(rows)
rng = random.Random(42)
perm = list(range(n)); rng.shuffle(perm)
EV_OTHER = [EV[perm[i]] for i in range(n)]  # 本底: 别人的证据

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def embed_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

def embed_qwen(texts):
    out = []
    for s in range(0, len(texts), 32):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 32], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for att in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                out.append(np.asarray(d["embeddings"], dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        print("qwen", s, flush=True)
    return l2n(np.concatenate(out))

t0 = time.time()
Vb_q, Vb_a, Vb_ev, Vb_evo = embed_bge(Q), embed_bge(A), embed_bge(EV), embed_bge(EV_OTHER)
print("bge done", round(time.time() - t0, 1), flush=True)
t0 = time.time()
Vw_q, Vw_a, Vw_ev, Vw_evo = embed_qwen(Q), embed_qwen(A), embed_qwen(EV), embed_qwen(EV_OTHER)
print("qwen done", round(time.time() - t0, 1), flush=True)

def cos_rows(X, Y):
    return np.einsum("ij,ij->i", X, Y)

res = {}
for tag, (vq, va, vev, vevo) in {
        "BGE1024": (Vb_q, Vb_a, Vb_ev, Vb_evo),
        "qwen256": (Vw_q, Vw_a, Vw_ev, Vw_evo)}.items():
    res[tag] = dict(
        floor=cos_rows(vq, vevo),
        q_ev=cos_rows(vq, vev),
        q_a=cos_rows(vq, va),
        a_ev=cos_rows(va, vev))

ok = np.array([r.get("llm_score") == 1 for r in rows])

def stat(x, mask=None):
    x = x if mask is None else x[mask]
    return "mean %.3f / med %.3f" % (x.mean(), np.median(x))

for tag in res:
    print("=====", tag, "=====")
    for k in ("floor", "q_ev", "q_a", "a_ev"):
        v = res[tag][k]
        print("  %-5s 全体[%s] 对题[%s] 错题[%s]" % (k, stat(v), stat(v, ok), stat(v, ~ok)))
    for k in ("q_ev", "q_a", "a_ev"):
        lift = res[tag][k] - res[tag]["floor"]
        print("  %s-本底抬升: 对题[%s] 错题[%s]" % (k, stat(lift, ok), stat(lift, ~ok)))

# 例子: 错题里 q_a vs q_ev 谁更近
print()
print("=== 错题例(q_a与q_ev同题对比) ===")
idx = np.where(~ok)[0][:6]
for i in idx:
    print("Q:", Q[i][:70])
    print("   A[%s]: q_a=%.3f  BGE / %.3f qwen | q_ev=%.3f BGE / %.3f qwen" % (
        A[i][:40],
        res["BGE1024"]["q_a"][i], res["qwen256"]["q_a"][i],
        res["BGE1024"]["q_ev"][i], res["qwen256"]["q_ev"][i]))

json.dump({t: {k: v.tolist() for k, v in d.items()} for t, d in res.items()},
          open(HERE + "/qa_space_cos.json", "w"), indent=1)
print("SAVED qa_space_cos.json", flush=True)
