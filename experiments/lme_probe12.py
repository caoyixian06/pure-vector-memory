# -*- coding: utf-8 -*-
"""lme_probe12.py — 纯向量问句分离验证(用户宪法: 零标点/零词表/零标注):
问句方向=问题集质心方向(问题=系统天然输入) → 句子投影=问句性
对照: 标点版(?结尾自标注)AUC=0.988"""
import io, json, os, re, sys, time, hashlib
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
rng = np.random.RandomState(0)
pick = rng.choice(len(RAWS), size=30000, replace=False)
sents = []
for i in pick:
    for m in re.finditer(r"[^.!?]*[.?]?", RAWS[i]):
        p = m.group(0).strip()
        if 4 <= len(p.split()) <= 45:
            sents.append(p)
sents = [sents[j] for j in rng.choice(len(sents), size=20000, replace=False)]
is_q = np.array([s.rstrip().endswith("?") for s in sents])
P("句子=%d 问句=%d(%.1f%%) %.0fs" % (len(sents), is_q.sum(), 100.0 * is_q.mean(), time.time() - t0))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
SV = []
for st in range(0, len(sents), 64):
    SV.append(np.asarray(bge.encode(sents[st:st + 64])["dense_vecs"], dtype=np.float32))
SV = l2n(np.concatenate(SV))
P("句子嵌入 %.0fs" % (time.time() - t0))

# 问题集质心方向(零标注: 问题是系统输入)
XQ = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
qcentroid = l2n(XQ.mean(0, keepdims=True))[0]
# 库句均值(对照方向: 减去库的共模,留下问句的特异方向)
lib_centroid = l2n(SV.mean(0, keepdims=True))[0]
ask_dir = qcentroid - lib_centroid
ask_dir = ask_dir / max(np.linalg.norm(ask_dir), 1e-9)

proj = SV @ ask_dir
from bisect import bisect_left
def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    return (sum(bisect_left(allv, v) + 1 for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

P("\n===== 纯向量问句分离(问题质心法) =====")
P("问题质心方向投影: 问句=%.4f 陈述=%.4f AUC=%.3f" % (
    proj[is_q].mean(), proj[~is_q].mean(), auc(proj[is_q], proj[~is_q])))
P("对照: 标点自标注版AUC=0.988")
P("done %.0fs" % (time.time() - t0))
