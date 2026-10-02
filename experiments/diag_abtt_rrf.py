# -*- coding: utf-8 -*-
"""diag_abtt_rrf.py — 降噪两招实测:
① ABTT/主成分扣除(All-but-the-Top): 库向量与查询同变换, 扣均值+前k主成分, k=0/1/2/4/8
② RRF替代z融合: score=Σ 1/(60+rank)
指标同前: 错题进25 / 对题进5 / 均名次
"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
out = []
for s in range(0, n, 64):
    out.append(np.asarray(bge.encode(Q[s:s + 64])["dense_vecs"], dtype=np.float32))
bQ = l2n(np.concatenate(out))

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        rr = json.loads(l)
        REC[rr.get("memory_id")] = rr
    except Exception:
        pass
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

targets = []
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        targets.append((i, hits))
print("matched:", len(targets), flush=True)

# ABTT 变换
mu = D.mean(0)
Xc = D - mu
# 主成分(库分布的前k方向)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
PC = Vt[:8]  # 前8主成分方向

def abtt(M, k):
    Z = M - mu
    if k > 0:
        Z = Z - (Z @ PC[:k].T) @ PC[:k]
    return l2n(Z)

z = np.load(HERE + "/fusion_cache.npz")
SQ, VEX = z["SQ"], z["VEX"]

def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def ev(SCf, tag):
    m25 = o5 = cnt = 0
    rsum = 0
    for i, hits in targets:
        s = SCf(i)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        rsum += rk; cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-30s 错题进25 %3d  对题进5 %3d  均名次%.0f" % (tag, m25, o5, rsum / cnt), flush=True)
    return m25, o5, rsum / cnt

print("== ① ABTT 扣主成分(纯BGE通道) ==", flush=True)
for k in (0, 1, 2, 4, 8):
    Dk = abtt(D, k)
    Qk = abtt(bQ, k)
    SBk = (Qk @ Dk.T).astype(np.float32)
    ev(lambda i, S=SBk: S[i], "ABTT k=%d(扣均值+%d主成分)" % (k, k))

print("== ② ABTT(k最优附近) + qwen + 词票 三路z融合 ==", flush=True)
for k in (1, 2, 4):
    Dk = abtt(D, k)
    Qk = abtt(bQ, k)
    SBk = (Qk @ Dk.T).astype(np.float32)
    ev(lambda i, S=SBk: zs(S[i]) + zs(SQ[i]) + 0.5 * zs(VEX[i]), "ABTT%d + a1 + b0.5ex" % k)

print("== ③ RRF 替代 z融合(1/(60+rank)) ==", flush=True)
z0 = np.load(HERE + "/fusion_cache.npz")
SB = z0["SB"]
def rrf_fuse(i, k_rrf=60, ws=(1.0, 1.0, 0.5)):
    sc = np.zeros(D.shape[0], dtype=np.float32)
    for w, S in zip(ws, (SB, SQ, VEX)):
        rk = np.empty(D.shape[0], dtype=np.float32)
        rk[np.argsort(-S[i])] = np.arange(1, D.shape[0] + 1)
        sc += w / (k_rrf + rk)
    return sc
ev(lambda i: rrf_fuse(i), "RRF 三路(1/1/0.5)")
ev(lambda i: rrf_fuse(i, ws=(1.0, 1.0, 1.0)), "RRF 三路(1/1/1)")
ev(lambda i: rrf_fuse(i, ws=(1.0, 1.0, 0.0)), "RRF 两路(BGE+qwen)")
