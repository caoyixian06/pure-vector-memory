# -*- coding: utf-8 -*-
"""foundry124.py — 上下文软对齐上turn排序(独特联系的武器化):
会话定位(cos第一会话)→会话内按软对齐排 vs 按记录cos排, 总分对错"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry124_results.txt", "w", encoding="utf-8")
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
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))

CTX = np.load("C:/locomo_refined/ctx_token_vecs/ctx_vecs.npy")
meta = [json.loads(l) for l in io.open("C:/locomo_refined/ctx_token_vecs/tokens_meta.jsonl", encoding="utf-8")]
REC2TOK = {}
for ti, m in enumerate(meta):
    REC2TOK.setdefault(m["r"], []).append(ti)
CTXN = l2n(CTX.astype(np.float32))  # 归一一次(内存~1.7GB float32)
P("loaded+ctx归一 %.0fs" % (time.time() - t0))

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
    tidx = REC2TOK.get(ridx)
    if not tidx or len(qctx) == 0:
        return -1.0
    R = CTXN[tidx]
    return float((qctx @ R.T).max(axis=1).mean())

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

res = {"cos": {1: 0, 3: 0, 5: 0}, "align": {1: 0, 3: 0, 5: 0}, "fuse": {1: 0, 3: 0, 5: 0}}
n = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    cq = D @ X[k_i]
    sess_score = {}
    for cv in UCONVS:
        sess_score[cv] = float(cq[CONVKEY == cv].max())
    top_conv = max(sess_score, key=sess_score.get)
    rows = np.where((CONVKEY == top_conv) & np.array(["_rbak" not in MID[i] for i in range(len(MID))]))[0]
    if len(rows) < 5:
        continue
    qctx = ctx_tokens(Q[qa]["question"])
    scores_align = [align_score(qctx, int(r)) for r in rows]
    scores_cos = [float(D[int(r)] @ X[k_i]) for r in rows]
    zs = lambda a: (np.asarray(a) - np.mean(a)) / (np.std(a) + 1e-9)
    fuse = zs(scores_cos) + zs(scores_align)
    oa = [int(rows[i]) for i in np.argsort(-np.asarray(scores_align))]
    oc = [int(rows[i]) for i in np.argsort(-np.asarray(scores_cos))]
    of = [int(rows[i]) for i in np.argsort(-fuse)]
    n += 1
    for k2 in (1, 3, 5):
        if G & set(oa[:k2]):
            res["align"][k2] += 1
        if G & set(oc[:k2]):
            res["cos"][k2] += 1
        if G & set(of[:k2]):
            res["fuse"][k2] += 1
    if k_i % 200 == 0:
        P("  %d align@5=%.1f%% cos@5=%.1f%% %.0fs" % (
            k_i, 100.0 * res["align"][5] / max(1, n), 100.0 * res["cos"][5] / max(1, n), time.time() - t0))

P("\n===== 上下文软对齐turn排序(n=%d) =====" % n)
P("记录cos:  top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res["cos"][1] / n, 100.0 * res["cos"][3] / n, 100.0 * res["cos"][5] / n))
P("软对齐:   top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res["align"][1] / n, 100.0 * res["align"][3] / n, 100.0 * res["align"][5] / n))
P("cos+对齐融合: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res["fuse"][1] / n, 100.0 * res["fuse"][3] / n, 100.0 * res["fuse"][5] / n))
P("F124_DONE %.0fs" % (time.time() - t0))
