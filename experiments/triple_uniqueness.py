# -*- coding: utf-8 -*-
"""triple_uniqueness.py — 三元组独特性验证
真三元组: (a题目, c证据, x答案)   伪三元组: (a题目, h噪声, x答案)
对每组测全部已知关联, 真伪对照:
U1 关联谱: a-c, a-x, c-x 的余弦 (vs a-h, a-x, h-x)
U2 词覆盖: x独有词被c覆盖 vs 被h覆盖
U3 对齐质量: a词×c词 对齐 vs a词×h词 对齐
U4 Δ投影: c的Δ分 vs h的Δ分
U5 三元组"闭合度": x到a-c连线的距离(真三元组应更共线)
判据: 所有指标真>伪 且差异显著 → 关联独特性成立
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
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def toks(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

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

# Δ方向(全局)
evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
DELTA /= np.linalg.norm(DELTA)

A = emb([r["question"] for r in rows])
X = emb([str(r["answer"][0]) for r in rows])

def cos(u, v):
    return float(u @ v)

# 配对收集: 真三元组 vs 伪三元组(a, 随机噪声h, x)
rng = np.random.default_rng(11)
metrics = {k: ([], []) for k in ("U1_ac", "U1_ax", "U1_cx", "U2覆盖", "U3对齐", "U4Δ分", "U5闭合差")}
count = 0
for i in keys:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    if not ev or len(no) < 2:
        continue
    c = ev[0]
    h = no[0]
    a, x = A[i], X[i]
    av, xv, cv, hv = a, x, D[c], D[h]
    # 伪噪声: 换一个随机题的噪声(跨题噪声更严)
    # 真组
    metrics["U1_ac"][0].append(cos(av, cv))
    metrics["U1_ax"][0].append(cos(av, xv))
    metrics["U1_cx"][0].append(cos(cv, xv))
    # 词覆盖: x独有词被c覆盖
    xt = toks(x)
    at = toks(a)
    xonly = xt - at
    ct = toks(TEXTS[c])
    ht = toks(TEXTS[h])
    if xonly:
        metrics["U2覆盖"][0].append(len(xonly & ct) / len(xonly))
        metrics["U2覆盖"][1].append(len(xonly & ht) / len(xonly))
    # 对齐: a词×c词(需要词向量) — 简化用toks重合的连续版: 用emb文本
    # U3用词对齐近似: xonly在c中的词数 vs 在h中
    metrics["U3对齐"][0].append(len(xonly & ct))
    metrics["U3对齐"][1].append(len(xonly & ht))
    # U4 Δ分
    metrics["U4Δ分"][0].append(float(cv @ DELTA))
    metrics["U4Δ分"][1].append(float(hv @ DELTA))
    # U5 闭合差: |x - (a到c连线最近点)| 即 x到a-c直线的距离
    ch = cv - av
    t = float((xv - av) @ ch / (ch @ ch + 1e-9))
    foot = av + t * ch
    dline_true = float(np.linalg.norm(xv - foot))
    hh = hv - av
    t2 = float((xv - av) @ hh / (hh @ hh + 1e-9))
    foot2 = av + t2 * hh
    dline_false = float(np.linalg.norm(xv - foot2))
    metrics["U5闭合差"][0].append(-dline_true)   # 负号: 越小越好→取负后AUC>0.5
    metrics["U5闭合差"][1].append(-dline_false)
    count += 1
print("配对样本:", count, flush=True)

def auc(x, plab):
    x = np.asarray(x, dtype=np.float64)
    plab = np.asarray(plab, dtype=bool)
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

print("== 三元组独特性: 真(c证据) vs 伪(h噪声) ==", flush=True)
all_true = []
all_false = []
for k in metrics:
    t_, f_ = metrics[k]
    t_, f_ = np.array(t_, dtype=np.float64), np.array(f_, dtype=np.float64)
    if len(t_) < 10:
        print("  %s: 样本不足" % k, flush=True)
        continue
    x = np.concatenate([t_, f_])
    lb = np.zeros(len(x), dtype=bool); lb[:len(t_)] = True
    a = auc(x, lb)
    print("  %-10s 真均值%.3f 伪均值%.3f  AUC=%.3f" % (k, t_.mean(), f_.mean(), a), flush=True)
    all_true.append(t_ / (np.abs(t_).std() + 1e-9))
    all_false.append(f_ / (np.abs(f_).std() + 1e-9))
# 组合(全部指标z和)
ct = np.mean(np.stack(all_true), axis=0)
cf = np.mean(np.stack(all_false), axis=0)
x = np.concatenate([ct, cf]); lb = np.zeros(len(x), dtype=bool); lb[:len(ct)] = True
print("  组合(全部指标): AUC=%.3f" % auc(x, lb), flush=True)
print("UNIQ_DONE", flush=True)
