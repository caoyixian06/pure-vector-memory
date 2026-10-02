# -*- coding: utf-8 -*-
"""solve_x.py — x已知视角: 逆解方程 x = f(a, M) 的最小M
Q1 x与a的差向量 d = x - a: 它的长度/方向——"从问题到答案要走多远、往哪走"
Q2 d能否由'小M'(少量库向量)线性表出? 试 M = {问题质心qc, 陈述质心stmt, Δ, 该题证据c, 该题噪声h}
   解 x ≈ a + Σ w_i * m_i, 看需要几个m、残差多少
Q3 最小M搜索: 从库字典(主成分/词锚轴/质心族)里贪心选, 使 R²→1, 报告M的大小
Q4 d在会话内的特异性: d_本题 减 全局平均d̄, 剩余部分与本题证据c的关系
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

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
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
MID2I = {m: i for i, m in enumerate(MID)}
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

X_vec = emb([str(r["answer"][0]) for r in rows])   # x 已知
A_vec = emb([r["question"] for r in rows])          # a
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
print("样本:", len(keys), flush=True)

# Q1: d = x - a
Dv = X_vec[keys] - A_vec[keys]           # 未归一
dn = l2n(Dv)
print("Q1 |x-a| 均值%.3f 中位%.3f (单位球上)" % (
    np.linalg.norm(Dv, axis=1).mean(), np.median(np.linalg.norm(Dv, axis=1))), flush=True)
dbar = Dv.mean(0)
dbar_n = l2n(dbar[None])[0]
own = np.einsum("ij,ij->i", l2n(Dv), np.tile(dbar_n, (len(keys), 1)))
print("Q1b 方向一致性 cos(d_i, d̄)=%.3f (0.178问题=方向乱; x视角再看一次)" % own.mean(), flush=True)

# Q2: 小M线性表出. 字典 = 全局质心/陈述质心/问题质心/Δ/叙事轴 + 本题c与h
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = l2n(D[raw_idx[:3000]].mean(0)[None])[0]
qc_all = l2n(A_vec.mean(0)[None])[0]
sum_idx = [i for i in range(N) if KIND[i] != "raw"]
summ_c = l2n(D[sum_idx].mean(0)[None])[0]
# Δ
evA, noA = [], []
halfk = len(keys) // 2
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50 = z1["TOP50"]
for i in keys[:halfk]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evA += [j for j in top if j in hs or j in tws]
    noA += [j for j in top if j not in hs and j not in tws][:8]
DELTA = l2n((D[np.array(evA)].mean(0) - D[np.array(noA)].mean(0))[None])[0]
# 叙事轴
narr_w = ["last", "week", "ago", "yesterday", "took", "went", "made", "started", "finished", "visited", "moved", "bought", "played", "watched", "read", "graduated", "worked", "lived", "traveled", "adopted", "won", "signed", "joined", "planned"]
fluf_w = ["awesome", "great", "wow", "thanks", "thank", "support", "glad", "proud", "amazing", "cool", "nice", "love", "happy", "excited", "sorry", "congrats", "enjoy", "fun", "best", "sweet"]
narr = l2n((emb(["the " + w for w in narr_w]).mean(0) - emb(["the " + w for w in fluf_w]).mean(0))[None])[0]

DICT = {"qc全局问题质心": qc_all, "stmt陈述质心": stmt, "summ摘要质心": summ_c,
        "Δ证据指纹": DELTA, "叙事轴": narr}
BASE = np.stack(list(DICT.values()))
# 每题加入自己的 c 与 h
res_list = []
for kk, i in enumerate(keys):
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    no = [j for j in TOP50[i] if j not in hs and j not in tws]
    c_v = D[ev[0]] if ev else stmt
    h_v = D[no[0]] if no else stmt
    Mmat = np.stack([c_v, h_v])                      # 本题专属M
    target_d = X_vec[i] - A_vec[i]
    # 最小二乘: target_d ≈ BASE^T w_g + Mmat^T w_l
    Amat = np.concatenate([BASE.T, Mmat.T], axis=1)  # (1024, 5+2)
    w, *_ = np.linalg.lstsq(Amat, target_d, rcond=None)
    pred = Amat @ w
    r2 = 1 - np.sum((target_d - pred) ** 2) / max(1e-9, np.sum((target_d - target_d.mean()) ** 2))
    # 只用全局字典(无本题c/h)
    pred_g = BASE.T @ w[:BASE.shape[0]]
    r2g = 1 - np.sum((target_d - pred_g) ** 2) / max(1e-9, np.sum((target_d - target_d.mean()) ** 2))
    # 只用本题c/h
    pred_l = Mmat.T @ w[BASE.shape[0]:]
    r2l = 1 - np.sum((target_d - pred_l) ** 2) / max(1e-9, np.sum((target_d - target_d.mean()) ** 2))
    res_list.append((r2, r2g, r2l, w[BASE.shape[0]:][0], w[BASE.shape[0]:][1]))
res = np.array(res_list)
print("Q2 方程 x-a = f(M) 的拟合(留出前不管, 先看结构):", flush=True)
print("  M=全局字典5轴+本题{c,h}: R²=%.3f" % res[:, 0].mean(), flush=True)
print("  M=仅全局5轴:            R²=%.3f" % res[:, 1].mean(), flush=True)
print("  M=仅本题{c,h}:          R²=%.3f" % res[:, 2].mean(), flush=True)
print("  本题系数: c权重%.3f h权重%.3f (c主导→答案≈问题+证据方向的函数)" % (
    res[:, 3].mean(), res[:, 4].mean()), flush=True)

# Q3: 权重落在哪个字典轴上
wnames = list(DICT.keys())
wglob = np.array([np.linalg.lstsq(BASE.T, X_vec[i] - A_vec[i], rcond=None)[0] for i in keys[:400]])
print("Q3 全局轴权重均值:", {wnames[k]: round(float(wglob[:, k].mean()), 3) for k in range(len(wnames))}, flush=True)

# Q4: 剥掉全局d̄后的残差 与 c 的关系
resid = Dv - dbar[None]
cos_rc = np.einsum("ij,ij->i", l2n(resid), np.stack([D[[j for j in TOP50[i] if j in targets[i]] [0]] for i in keys]))
print("Q4 剥全局d̄后, 残差方向与本题c的cos=%.3f (高→每题剩余确实由证据决定)" % cos_rc.mean(), flush=True)
print("SOLVEX_DONE", flush=True)
