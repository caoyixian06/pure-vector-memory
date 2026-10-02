# -*- coding: utf-8 -*-
"""vec_numbers.py — 直接看数字: qwen词向量的逐维解剖
N1 "last month" vs "July"/"January"/"apple" 的逐维乘积分布——0.815由哪些维贡献
N2 相对词组 vs 月份组 vs 随机组: 每组的均值向量逐维对比, 找出"时间维"
N3 月份内部: 12个月的向量互相之间——月份族是否真实成簇
N4 "last month"和"July"的高贡献维上, 数值符号与幅度——它懂的是"时间"还是"最近"还是别的
"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

import urllib.request
def emb(texts):
    out = []
    for s in range(0, len(texts), 32):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 32], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
    return l2n(np.concatenate(out))

rel_words = ["yesterday", "last week", "last month", "next week", "tomorrow", "recently", "ago", "last year"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
rand_words = ["apple", "democracy", "guitar", "cloud", "pencil", "justice", "river", "mountain", "bread", "shadow", "lamp", "engine"]
other_time = ["Monday", "Tuesday", "morning", "evening", "noon", "midnight", "weekend", "weekday", "dawn", "dusk"]

allw = rel_words + months + rand_words + other_time
EMB = l2n(emb(allw))
nR, nM, nX, nT = len(rel_words), len(months), len(rand_words), len(other_time)
iR, iM = 0, nR
iX, iT = iM + nM, iM + nM + nX
R_ = EMB[iR:iR+nR]
M_ = EMB[iM:iM+nM]
X_ = EMB[iX:iX+nX]
T_ = EMB[iT:iT+nT]

print("== N1 'last month'与各词的逐维乘积分布 ==", flush=True)
lm = EMB[2]  # last month
july = EMB[iM + 6]
jan = EMB[iM + 0]
apple = EMB[iX + 0]
pm = lm * july
pj = lm * jan
pa = lm * apple
top_pm = np.argsort(-np.abs(pm))[:8]
print("  last-month×July top8维:", top_pm.tolist(), flush=True)
print("    乘积值:", np.round(pm[top_pm], 3).tolist(), flush=True)
print("    July原值:", np.round(july[top_pm], 3).tolist(), flush=True)
print("    lm原值:  ", np.round(lm[top_pm], 3).tolist(), flush=True)
print("  last-month×apple top维(对照):", np.argsort(-np.abs(pa))[:5].tolist(),
      "乘积:", np.round(pa[np.argsort(-np.abs(pa))[:5]], 3).tolist(), flush=True)

print("== N2 三组的均值向量对比 ==", flush=True)
mR, mM, mX, mT = R_.mean(0), M_.mean(0), X_.mean(0), T_.mean(0)
print("  cos(R组,月份组)=%.3f cos(R组,随机组)=%.3f cos(月份,时间词)=%.3f" % (
    float(mR @ mM), float(mR @ mX), float(mM @ mT)), flush=True)
# 哪些维度区分 R组 vs 随机组
dd = mR - mX
topd = np.argsort(-np.abs(dd))[:10]
print("  R-vs-随机 top10区分维:", topd.tolist(), flush=True)
print("  R组均值:", np.round(mR[topd], 3).tolist(), flush=True)
print("  随机组值:", np.round(mX[topd], 3).tolist(), flush=True)
print("  月份组值:", np.round(mM[topd], 3).tolist(), flush=True)

print("== N3 月份族内部结构 ==", flush=True)
MM = M_ @ M_.T
iu = np.triu_indices(nM, 1)
print("  月份两两cos: 均值%.3f min%.3f max%.3f" % (MM[iu].mean(), MM[iu].min(), MM[iu].max()), flush=True)
# 相邻月份(时间相邻) vs 不相邻
adjc, farc = [], []
for a2 in range(nM):
    for b2 in range(nM):
        if a2 == b2:
            continue
        dd2 = min((a2 - b2) % 12, (b2 - a2) % 12)
        if dd2 == 1:
            adjc.append(MM[a2, b2])
        elif dd2 >= 5:
            farc.append(MM[a2, b2])
print("  时间相邻月份cos=%.3f vs 远隔月份=%.3f (循环排列感知?)" % (
    np.mean(adjc), np.mean(farc)), flush=True)

print("== N4 R组与时间词组的对齐 ==", flush=True)
RT = R_ @ T_.T
for ridx, rw in enumerate(rel_words):
    j = int(np.argmax(RT[ridx]))
    print("  %-11s 最像的时间词: %s (cos=%.3f)" % (rw, other_time[j], RT[ridx, j]), flush=True)
print("NUMBERS_DONE", flush=True)
