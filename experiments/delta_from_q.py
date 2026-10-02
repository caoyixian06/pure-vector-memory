# -*- coding: utf-8 -*-
"""delta_from_q.py — 从问题推导Δ方向(零答案在线信号)
方法: 新题的Δ̂ = k个最相似训练题的Δ_i均值(局部平均, 绕开Ridge全局映射)
验证: Δ̂投影 判别 新题的证据/噪声 AUC, 对照 全局Δ0.802/随机0.5
k = 1/3/5/10
"""
import io, json, os, sys, re, random
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
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
MID2I = {m: i for i, m in enumerate(MID)}
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
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

rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits:
        targets[i] = hits
keys = sorted(targets.keys())
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
bQ = emb([r["question"] for r in rows])
bQ = l2n(bQ)
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQu = l2n(bQ + u2)
SBS = (bQu @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# 逐题Δ_i(质心差, 拆半: 前半=训练库, 后半=测试)
train_deltas = {}   # i -> delta_i (归一)
for i in keys[:len(keys) // 2]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:6]
    if not ev or len(no) < 3:
        continue
    di = D[ev].mean(0) - D[no].mean(0)
    n2 = np.linalg.norm(di) + 1e-9
    train_deltas[i] = di / n2
print("训练Δ库:", len(train_deltas), flush=True)

train_ids = list(train_deltas.keys())
TRM = np.stack([bQu[i] for i in train_ids])       # 训练题向量(用于近邻)
TRD = np.stack([train_deltas[i] for i in train_ids])

def auc(x, plab):
    x = np.asarray(x, dtype=np.float64)
    plab = np.asarray(plab, dtype=bool)
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

# 全局Δ对照
evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
DELTA_G = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
DELTA_G /= np.linalg.norm(DELTA_G)

test_ids = keys[len(keys) // 2:]
print("测试题:", len(test_ids), flush=True)
for k in (1, 3, 5, 10):
    aucs_dn = []
    aucs_dg = []
    for i in test_ids:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        top = TOP50[i][np.argsort(-RER[i])]
        ev = [j for j in top if j in hs or j in tws]
        no = [j for j in top if j not in hs and j not in tws][:6]
        if not ev or len(no) < 3:
            continue
        # 近邻Δ
        sims = TRM @ bQu[i]
        nn = np.argsort(-sims)[:k]
        dhat = TRD[nn].mean(0)
        dhat /= np.linalg.norm(dhat) + 1e-9
        se = D[ev] @ dhat
        sn = D[no] @ dhat
        x = np.concatenate([se, sn]); lb = np.zeros(len(x), dtype=bool); lb[:len(se)] = True
        aucs_dn.append(auc(x, lb))
        se2 = D[ev] @ DELTA_G
        sn2 = D[no] @ DELTA_G
        x2 = np.concatenate([se2, sn2]); lb2 = np.zeros(len(x2), dtype=bool); lb2[:len(se2)] = True
        aucs_dg.append(auc(x2, lb2))
    print("k=%2d: 近邻Δ̂ AUC=%.3f | 全局Δ AUC=%.3f" % (k, np.mean(aucs_dn), np.mean(aucs_dg)), flush=True)
print("DFQ_DONE", flush=True)
