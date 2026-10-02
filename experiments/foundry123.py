# -*- coding: utf-8 -*-
"""foundry123.py — 独特联系证明(用户令): 上下文词向量层的 问题↔金证据 匹配
假说: 上下文化消除歧义→软对齐翻正(静态版0.399反向)
证: 金vs同会话噪声的AUC+均值+逐题成立率(所有题里找到)"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry123_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]

# 上下文token向量库(F116导出)
CTX = np.load("C:/locomo_refined/ctx_token_vecs/ctx_vecs.npy")  # (T,1024) fp16
meta = [json.loads(l) for l in io.open("C:/locomo_refined/ctx_token_vecs/tokens_meta.jsonl", encoding="utf-8")]
P("ctx库: %s tokens=%d %.0fs" % (str(CTX.shape), len(meta), time.time() - t0))
# 记录→token行号
REC2TOK = {}
for ti, m in enumerate(meta):
    REC2TOK.setdefault(m["r"], []).append(ti)

# 问题的上下文token向量(现场嵌入, GPU一次)
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()

def ctx_tokens(text):
    with torch.no_grad():
        enc = tok([text[:1500]], truncation=True, max_length=64, return_tensors="pt").to("cuda")
        out = model(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        v = out[mask].cpu().numpy().astype(np.float32)
    return l2n(v)

def align_score(qctx, ridx):
    """上下文软对齐: 问题token与记录token的cross-max均值"""
    tidx = REC2TOK.get(ridx)
    if not tidx or len(qctx) == 0:
        return None
    R = l2n(CTX[tidx].astype(np.float32))
    S = qctx @ R.T
    return float(S.max(axis=1).mean())

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    from bisect import bisect_left
    rp = sum(bisect_left(allv, v) + 1 for v in pos)
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
pos_scores = []
neg_scores = []
per_q_win = 0
per_q_tot = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    qctx = ctx_tokens(Q[qa]["question"])
    conv = CONVKEY[0] if False else None
    myconv = "loco-" + qa.split("#")[0]
    gold_i = [i for i in G if i in REC2TOK]
    noise_pool = []
    seen = set()
    for i in range(len(MID)):
        if CONVKEY[i] == myconv and i not in G and i in REC2TOK and "_rbak" not in MID[i]:
            if i not in seen:
                noise_pool.append(i)
                seen.add(i)
    if not gold_i or len(noise_pool) < 20:
        continue
    rng = np.random.RandomState(k_i)
    noise_pick = rng.choice(noise_pool, size=min(60, len(noise_pool)), replace=False).tolist()
    gs = [align_score(qctx, i) for i in gold_i]
    ns = [align_score(qctx, i) for i in noise_pick]
    gs = [x for x in gs if x is not None]
    ns = [x for x in ns if x is not None]
    if not gs or not ns:
        continue
    pos_scores += gs
    neg_scores += ns
    per_q_tot += 1
    if np.mean(gs) > np.mean(ns):
        per_q_win += 1
    if k_i % 200 == 0:
        P("  %d/%d 逐题胜率=%.1f%% %.0fs" % (
            k_i, len(IDS), 100.0 * per_q_win / max(1, per_q_tot), time.time() - t0))

P("\n===== 独特联系证明: 上下文软对齐(金vs同会话噪声) =====")
P("样本: 金=%d 噪=%d (题数=%d)" % (len(pos_scores), len(neg_scores), per_q_tot))
P("均值: 金=%.4f 噪=%.4f  AUC=%.3f" % (
    np.mean(pos_scores), np.mean(neg_scores), auc(pos_scores, neg_scores)))
P("逐题成立率(金的均分>噪声均分): %.1f%%" % (100.0 * per_q_win / max(1, per_q_tot)))
P("对照: 静态词向量版AUC=0.399(反向) | 会话鸿沟=0.122")
P("F123_DONE %.0fs" % (time.time() - t0))
