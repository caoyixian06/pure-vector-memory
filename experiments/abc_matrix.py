# -*- coding: utf-8 -*-
"""abc_matrix.py — 九变量关系网: a问题1024/b答案1024/c证据1024/e问题256/f答案256/g证据256/h噪声1024/i噪声256
P1 全对关系矩阵: 9×9 里所有跨空间组合(同空间余弦+跨空间回归系数)
P2 三元组结构: (a,c,h)与(a,g,i)的三点几何——问题在证据-噪声连线上的投影位置
P3 排序律推导: 从关系数字推出 score公式, 留出验证AUC
P4 组合规律: c与h的差异在哪个空间大(判别信息在哪个空间) — 前面已答, 这里给逐对完整表
"""
import io, json, os, sys, re, time, urllib.request
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
print("bge loaded", flush=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def emb_qwen(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
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
QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
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
TOP50 = z1["TOP50"]

# 每题固定一个证据c与一个噪声h(与该题最配对的: 证据=排名第一的证据, 噪声=排名第一的噪声)
A, B, C, E, F, G, H, I = [], [], [], [], [], [], [], []
sel_keys = []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    if not ev or not no:
        continue
    C.append(D[ev[0]]); H.append(D[no[0]])
    G.append(QW[ev[0]]); I.append(QW[no[0]])
    sel_keys.append(i)
C, H, G, I = map(np.array, (C, H, G, I))
sel_rows = [rows[i] for i in sel_keys]
A = emb_bge([r["question"] for r in sel_rows])
B = emb_bge([str(r["answer"][0]) for r in sel_rows])
E = emb_qwen([r["question"] for r in sel_rows])
F = emb_qwen([str(r["answer"][0]) for r in sel_rows])
print("样本:", len(sel_keys), flush=True)

# P1 同空间余弦
def cm(X, Y):
    return np.einsum("ij,ij->i", X, Y)
print("== P1 同空间九元关系(均值cos) ==", flush=True)
pairs_1024 = {"a-b": (A, B), "a-c": (A, C), "a-h": (A, H), "b-c": (B, C), "b-h": (B, H), "c-h": (C, H)}
for nm, (X, Y) in pairs_1024.items():
    print("  1024 %s: %.3f" % (nm, cm(X, Y).mean()), flush=True)
pairs_256 = {"e-f": (E, F), "e-g": (E, G), "e-i": (E, I), "f-g": (F, G), "f-i": (F, I), "g-i": (G, I)}
for nm, (X, Y) in pairs_256.items():
    print("  256  %s: %.3f" % (nm, cm(X, Y).mean()), flush=True)

# 跨空间: 维度不同不能逐维相关(概念错误已修正), 改测跨空间的"排序一致性":
# c的1024向量与g的256向量, 对同一批问题的排序是否一致 —— 留待P3
print("== P1b 跨空间同一文本 ==  维度不同(1024vs256)不能逐维相关, 改测排序一致性(见P3)", flush=True)

# P2 三点几何: 问题a在c-h连线上的投影位置 t
# t = ((a-h)·(c-h)) / |c-h|²   t=0→在h处, t=1→在c处
print("== P2 问题在 证据c—噪声h 连线上的位置 ==", flush=True)
ts = []
for k in range(len(C)):
    ch = C[k] - H[k]
    t = float((A[k] - H[k]) @ ch / (ch @ ch + 1e-9))
    ts.append(t)
ts = np.array(ts)
print("  t分布: 均值%.3f 中位%.3f | t<0(线外h侧)%.0f%% t>1(线外c侧)%.0f%%" % (
    ts.mean(), np.median(ts), 100 * (ts < 0).mean(), 100 * (ts > 1).mean()), flush=True)
# 256空间同样
ts2 = []
for k in range(len(G)):
    gi = G[k] - I[k]
    t = float((E[k] - I[k]) @ gi / (gi @ gi + 1e-9))
    ts2.append(t)
ts2 = np.array(ts2)
print("  256空间: 均值%.3f 中位%.3f" % (ts2.mean(), np.median(ts2)), flush=True)
# b(答案)在同连线上的位置
tbs = []
for k in range(len(C)):
    ch = C[k] - H[k]
    t = float((B[k] - H[k]) @ ch / (ch @ ch + 1e-9))
    tbs.append(t)
tbs = np.array(tbs)
print("  答案b在线上位置: 均值%.3f (a=%.3f) → 答案与问题谁更靠近证据?" % (tbs.mean(), ts.mean()), flush=True)

# P3 排序律: score(a,c) = w1*(a·c) + w2*(c在线上的位置代理) 留出验证
print("== P3 排序律留出验证(证据vs噪声二分) ==", flush=True)
def auc(x, plab):
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())
half = len(sel_keys) // 2
# 训练A半: 线性组合系数(两特征: a·x 与 x的norm差的符号信息不用; 直接解1D最优阈值无意义, 用logistic单变量)
def fit_w(feat, lab):
    # 简单: AUC最优的单调变换就是排序本身, 权重= Fisher比
    m1, m0 = feat[lab == 1].mean(), feat[lab == 0].mean()
    s = np.sqrt(feat[lab == 1].var() + feat[lab == 0].var()) + 1e-9
    return (m1 - m0) / s, m1, m0, s
feats = {
    "a·x(1024)": np.concatenate([cm(A[:half], C[:half]), cm(A[half:], C[half:])]),
    "a·x(256)": np.concatenate([cm(E[:half], G[:half]), cm(E[half:], G[half:])]),
    "线位置t(1024)": np.concatenate([ts[:half], ts[half:]]),
}
labA = np.ones(half, dtype=bool)
labA = np.concatenate([np.ones(half, dtype=bool), np.zeros(half, dtype=bool)])
for nm, f in feats.items():
    # A半求标准化, B半测
    fA, fB = f[:half], f[half:]
    d, m1, m0, s = fit_w(fA, labA[:half])
    zB = (fB - m0) / s
    a = auc(zB, labA[half:])
    print("  %-14s 单特征B半AUC=%.3f" % (nm, a), flush=True)
# 组合: a·c + a·h 的差(相对分) 与 t
reldiff = cm(A, C) - cm(A, H)
fA, fB = reldiff[:half], reldiff[half:]
d, m1, m0, s = fit_w(fA, labA[:half])
print("  相对分(a·c−a·h) B半AUC=%.3f" % auc((fB - m0) / s, labA[half:]), flush=True)
# 256相对分
rd2 = cm(E, G) - cm(E, I)
fA, fB = rd2[:half], rd2[half:]
d, m1, m0, s = fit_w(fA, labA[:half])
print("  相对分256 B半AUC=%.3f" % auc((fB - m0) / s, labA[half:]), flush=True)
# 双空间和
both = (cm(A, C) - cm(A, H)) + (cm(E, G) - cm(E, I))
fA, fB = both[:half], both[half:]
d, m1, m0, s = fit_w(fA, labA[:half])
print("  双空间和 B半AUC=%.3f" % auc((fB - m0) / s, labA[half:]), flush=True)
print("ABCMATRIX_DONE", flush=True)
