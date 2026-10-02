# -*- coding: utf-8 -*-
"""verify_rules.py — 验证"问题规则"是否真实存在于向量空间
V1 人称失配: LongMemEval是第一人称对话("I graduated with X")。
   测: 第一人称查询("What degree did I get") vs 第三人称查询("What degree did she get")
       对同一批第一人称句子的检索命中率差。
   数据: 从LME的session第一人称句子里采样, 构造两类查询。
V2 相对时间失配: 测 "When did X happen last month" vs "When did X happen in March 2023"
   对含绝对日期句子的检索命中率差。
V3 时间锚定的必要性: 相对词("yesterday")的向量 与 它实际指的绝对日期("2023-05-03")的向量
   距离 vs 随机日期——如果距离不比随机近, 说明嵌入模型根本不懂相对时间, 锚定必要。
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

# 从LoCoMo raw库采样第一人称句子(数据已嵌入, 用D+MID)
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

# 第一人称事实句样本(含I/my/me的事实陈述)
fp_sents = []
for k in range(N):
    if KIND[k] != "raw":
        continue
    t = rec_text(REC.get(MID[k], {}))
    if re.match(r"^[A-Z][a-z]+:\s*(I|My|We)\b", t) and 30 < len(t) < 200 and "?" not in t:
        if re.search(r"\b(have|has|had|am|was|went|got|like|love|work|live|study|bought|moved|started|finished)\b", t, re.I):
            fp_sents.append((k, t))
print("第一人称事实句:", len(fp_sents), flush=True)
rng = random.Random(9)
sample = rng.sample(fp_sents, min(120, len(fp_sents)))

# V1: 人称视角检索失配
# 对每个第一人称句子, 构造第一/第三人称查询, 各检索全库, 看原句排名
win_fp = win_tp = tie = 0
fp_ranks, tp_ranks = [], []
import numpy as _np
for k, t in sample[:80]:
    spk, body = t.split(":", 1)
    body = body.strip()
    # 第一人称查询: "What did I {do}"? 直接从句子反推问题形态(第一人称保留)
    q_fp = "What did I " + re.sub(r"^(have|had|like|love|want|need|start)\b", r"\1", body.lower(), flags=re.I)[:90]
    q_tp = "What did " + spk + " " + re.sub(r"^(have|had|like|love|want|need|start)\b", r"\1", body.lower(), flags=re.I)[:90]
    qv_fp = emb([q_fp])[0]
    qv_tp = emb([q_tp])[0]
    # 检索全库: 原句的排名
    s_fp = D @ qv_fp
    s_tp = D @ qv_tp
    r_fp = int((s_fp > s_fp[k]).sum()) + 1
    r_tp = int((s_tp > s_tp[k]).sum()) + 1
    fp_ranks.append(r_fp)
    tp_ranks.append(r_tp)
    if r_fp < r_tp:
        win_fp += 1
    elif r_tp < r_fp:
        win_tp += 1
    else:
        tie += 1
fp_ranks, tp_ranks = np.array(fp_ranks), np.array(tp_ranks)
print("== V1 人称失配(第一人称句库) ==", flush=True)
print("  第一人称查询名次: 中位%.0f 均值%.0f" % (np.median(fp_ranks), fp_ranks.mean()), flush=True)
print("  第三人称查询名次: 中位%.0f 均值%.0f" % (np.median(tp_ranks), tp_ranks.mean()), flush=True)
print("  胜率: 第一%.0f%% 第三%.0f%% 平%.0f%%" % (
    100 * win_fp / len(sample[:80]), 100 * win_tp / len(sample[:80]), 100 * tie / len(sample[:80])), flush=True)

# V2: 相对时间失配
print("== V2 相对时间失配 ==", flush=True)
# 构造: 含具体日期的事实句, 分别用相对查询和绝对查询
DATE_SENTS = []
for k in range(N):
    if KIND[k] != "raw":
        continue
    t = rec_text(REC.get(MID[k], {}))
    m = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,4}\b", t)
    if m and "?" not in t and 40 < len(t) < 220:
        DATE_SENTS.append((k, t, m.group(0)))
print("含绝对日期句:", len(DATE_SENTS), flush=True)
sample2 = rng.sample(DATE_SENTS, min(60, len(DATE_SENTS)))
rel_ranks, abs_ranks = [], []
wr = wa = tie2 = 0
for k, t, dstr in sample2[:50]:
    # 相对查询: "When did that happen last month/year" 泛化形态
    q_rel = "When did " + re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,4}\b", "last month", t, flags=re.I)[:100]
    q_abs = "When did " + t[:90]
    rv = emb([q_rel])[0]
    av = emb([q_abs])[0]
    s_rel = D @ rv
    s_abs = D @ av
    r_rel = int((s_rel > s_rel[k]).sum()) + 1
    r_abs = int((s_abs > s_abs[k]).sum()) + 1
    rel_ranks.append(r_rel)
    abs_ranks.append(r_abs)
    if r_rel < r_abs:
        wr += 1
    elif r_abs < r_rel:
        wa += 1
    else:
        tie2 += 1
rel_ranks, abs_ranks = np.array(rel_ranks), np.array(abs_ranks)
print("  相对词查询名次: 中位%.0f 均值%.0f" % (np.median(rel_ranks), rel_ranks.mean()), flush=True)
print("  绝对词查询名次: 中位%.0f 均值%.0f" % (np.median(abs_ranks), abs_ranks.mean()), flush=True)
print("  胜率: 相对%.0f%% 绝对%.0f%% 平%.0f%%" % (
    100 * wr / max(1, wr + wa + tie2), 100 * wa / max(1, wr + wa + tie2), 100 * tie2 / max(1, wr + wa + tie2)), flush=True)

# V3: 相对词向量 vs 绝对日期向量
print("== V3 相对词与绝对日期的向量距离 ==", flush=True)
rel_words = ["yesterday", "last week", "last month", "next week", "tomorrow", "recently"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
RV = emb(rel_words)
MV = emb(months)
# 每个相对词与12个月的平均max-cos vs 与随机词的平均max-cos
rw_list = ["apple", "democracy", "guitar", "cloud", "pencil", "justice"]
RX = emb(rw_list)
for ridx, rw in enumerate(rel_words):
    d_month = float(np.max(RV[ridx] @ MV.T))
    d_rand = float(np.max(RV[ridx] @ RX[ridx % len(RX)]))
    print("  %-11s vs月份max-cos=%.3f vs随机词=%.3f" % (rw, d_month, d_rand), flush=True)
print("VERIFY_DONE", flush=True)
