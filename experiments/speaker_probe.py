# -*- coding: utf-8 -*-
"""speaker_probe.py — BGE句向量能否区分说话人? (严格对照实验)
实验: 取两个人对话库, 抽两人各自的句子向量, 用"留一最近邻分类"测:
  P1 分类准确率: 只用句向量, 能否说出这句是谁说的?
  P2 对照1: 内容匹配但说话人互换的假句(内容归一化对照)
  P3 对照2: 随机基线
  P4 结构分析: 两人的句子质心距离 vs 同人句间距离 — 说话人信息占多大空间
  P5 关键追问: 把内容信息剔除(减去话题质心)后, 说话人信号还剩多少?
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
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

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

# 取conv-26(Caroline vs Melanie, 前面解剖过的会话)两人各自的raw句
sents = {"A": [], "B": []}   # A=Caroline, B=Melanie
names = {"A": "Caroline", "B": "Melanie"}
for k in range(N):
    if KIND[k] != "raw" or not MID[k].startswith("loco-conv-26"):
        continue
    t = rec_text(REC.get(MID[k], {}))
    for g, nm in names.items():
        if t.startswith(nm + ":"):
            body = t[len(nm) + 1:].strip()
            if 30 < len(body) < 300 and "?" not in body:
                sents[g].append((k, body))
print("Caroline句: %d  Melanie句: %d" % (len(sents["A"]), len(sents["B"])), flush=True)

# P1 留一最近邻分类(余弦, 库向量)
idxA = [k for k, _ in sents["A"]]
idxB = [k for k, _ in sents["B"]]
correct = tot = 0
for g in ("A", "B"):
    own = sents[g]
    other = sents["B" if g == "A" else "A"]
    for k, body in own:
        # 最近邻(除自己)的身份 = 预测
        best_d, best_g = -9, None
        for g2 in ("A", "B"):
            for k2, b2 in (own if g2 == g else other):
                if g2 == g and k2 == k:
                    continue
                d2 = float(D[k] @ D[k2])
                if d2 > best_d:
                    best_d, best_g = d2, g2
        tot += 1
        if best_g == g:
            correct += 1
print("P1 留一最近邻身份分类: %d/%d = %.0f%% (随机50%%)" % (correct, tot, 100 * correct / tot), flush=True)

# P2 内容归一化对照: 换内容保身份——用两人说过的"同话题"句子
# 方法: 找两人都聊过的话题词(top重叠词), 各抽含该词的句子, 再分类
话题 = None
tokA = [toks(b) for _, b in sents["A"]]
tokB = [toks(b) for _, b in sents["B"]]
from collections import Counter
ca, cb = Counter(), Counter()
for s2 in tokA:
    ca.update(s2)
for s2 in tokB:
    cb.update(s2)
shared = [(w, min(ca[w], cb[w])) for w in set(ca) & set(cb) if ca[w] >= 5 and cb[w] >= 5]
shared.sort(key=lambda x: -x[1])
topics = [w for w, _ in shared[:15]]
print("共同话题词:", topics[:10], flush=True)
correct2 = tot2 = 0
for g in ("A", "B"):
    own = sents[g]
    other = sents["B" if g == "A" else "A"]
    for k, body in own:
        bt = toks(body)
        if not any(t2 in bt for t2 in topics):
            continue
        best_d, best_g = -9, None
        for g2 in ("A", "B"):
            for k2, b2 in (own if g2 == g else other):
                if g2 == g and k2 == k:
                    continue
                b2t = toks(b2)
                if not any(t3 in b2t for t3 in topics):
                    continue
                d2 = float(D[k] @ D[k2])
                if d2 > best_d:
                    best_d, best_g = d2, g2
        if best_g:
            tot2 += 1
            if best_g == g:
                correct2 += 1
if tot2:
    print("P2 同话题约束下分类: %d/%d = %.0f%%" % (correct2, tot2, 100 * correct2 / tot2), flush=True)

# P4 结构: 质心距离分析
cA = l2n(np.mean([D[k] for k, _ in sents["A"]], axis=0)[None])[0]
cB = l2n(np.mean([D[k] for k, _ in sents["B"]], axis=0)[None])[0]
print("P4 说话人质心cos(A,B)=%.4f" % float(cA @ cB), flush=True)
wa = np.mean([float(D[k] @ cA) for k, _ in sents["A"]])
wb = np.mean([float(D[k] @ cA) for k, _ in sents["B"]])
print("   A句到A质心%.4f vs B句到A质心%.4f" % (wa, wb), flush=True)

# P5 内容剔除后: 减去话题质心(所有句均值)再分类
allmean = l2n(np.mean(np.stack([D[k] for g in ("A", "B") for k, _ in sents[g]]), axis=0)[None])[0]
correct5 = tot5 = 0
for g in ("A", "B"):
    own = sents[g]
    other = sents["B" if g == "A" else "A"]
    for k, body in own:
        v = l2n((D[k] - allmean)[None])[0]
        best_d, best_g = -9, None
        for g2 in ("A", "B"):
            for k2, b2 in (own if g2 == g else other):
                if g2 == g and k2 == k:
                    continue
                v2 = l2n((D[k2] - allmean)[None])[0]
                d2 = float(v @ v2)
                if d2 > best_d:
                    best_d, best_g = d2, g2
        tot5 += 1
        if best_g == g:
            correct5 += 1
print("P5 去话题均值后分类: %d/%d = %.0f%%" % (correct5, tot5, 100 * correct5 / tot5), flush=True)
print("SPKPROBE_DONE", flush=True)
