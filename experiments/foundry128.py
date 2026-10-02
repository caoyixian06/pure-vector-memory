# -*- coding: utf-8 -*-
"""foundry128.py — 98.5%消失之谜的定量解剖(用户令):
逐题拆开: 金有几条、金在分数序的什么位置、噪声右尾有多深、金左尾拖多远
把'统计层成立个体层消失'拆成可命名的数值成分"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry128_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

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
CTXN = l2n(CTX.astype(np.float32))
P("loaded %.0fs" % (time.time() - t0))

import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
def ctx_tokens(text):
    with torch.no_grad():
        enc = tok([text[:1500]], truncation=True, max_length=64, return_tensors="pt").to("cuda")
        out = model(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        return l2n(out[mask].cpu().numpy().astype(np.float32))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

# 逐题解剖容器
rows_stat = []
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
    if len(rows) < 10:
        continue
    qctx = ctx_tokens(Q[qa]["question"])
    al = np.full(len(rows), -1.0)
    for rr, r in enumerate(rows):
        tidx = REC2TOK.get(int(r))
        if tidx:
            al[rr] = float((qctx @ CTXN[tidx].T).max(axis=1).mean())
    gmask = np.array([int(r) in G for r in rows])
    if gmask.sum() == 0 or (al > 0).sum() < 10:
        continue
    ga = al[gmask]
    na = al[~gmask & (al > 0)]
    if len(na) < 5:
        continue
    order = np.argsort(-al)
    ranks = [int(np.where(order == rr)[0][0]) for rr in range(len(rows)) if gmask[rr]]
    best_noise = float(na.max())
    rows_stat.append({
        "n_gold": int(gmask.sum()),
        "gold_mean": float(ga.mean()),
        "noise_mean": float(na.mean()),
        "gold_min": float(ga.min()),
        "noise_max": best_noise,
        "gold_best_rank": int(min(ranks)),
        "gold_worst_rank": int(max(ranks)),
        "n_noise": int(len(na)),
        "mean_rank_of_gold": float(np.mean(ranks)),
        "n_noise_above_gold_mean": int((na > ga.mean()).sum()),
        "gap_mean": float(ga.mean() - na.mean()),
        "overlap_frac": float(((na >= ga.min()) & (na <= ga.max())).sum() / len(na)),
    })
    if k_i % 300 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))

import collections
st = rows_stat
n = len(st)
P("\n===== 98.5%%消失之谜解剖(n=%d题) =====" % n)
P("[均值层——规律确实在]")
P("  金均值-噪均值 gap: 均值=%.4f 中位=%.4f  gap>0的题=%.1f%%" % (
    np.mean([s["gap_mean"] for s in st]), np.median([s["gap_mean"] for s in st]),
    100.0 * np.mean([s["gap_mean"] > 0 for s in st])))
P("[个体层——消失的三个嫌疑犯]")
P("  嫌疑犯1·人数: 每题噪声条数中位=%d vs 金条数中位=%d (人数比=%.0f:1)" % (
    np.median([s["n_noise"] for s in st]), np.median([s["n_gold"] for s in st]),
    np.median([s["n_noise"] for s in st]) / max(1, np.median([s["n_gold"] for s in st]))))
P("  嫌疑犯2·噪声右尾: 超过金均值的噪声条数/题 中位=%.0f条 (这些条排在多数金前面)" % (
    np.median([s["n_noise_above_gold_mean"] for s in st])))
P("  嫌疑犯3·金左尾: 金的最佳名次 中位=%d / 金的最差名次 中位=%d / 金均名次 中位=%.0f" % (
    np.median([s["gold_best_rank"] for s in st]), np.median([s["gold_worst_rank"] for s in st]),
    np.median([s["mean_rank_of_gold"] for s in st])))
P("[量化: 要'抓住'需要什么]")
P("  噪声max>金min 的题占比=%.1f%% (即至少一片金被某条噪声压住)" % (
    100.0 * np.mean([s["noise_max"] > s["gold_min"] for s in st])))
P("  前5全金 的题占比=%.1f%% | 金best_rank≤5 的题占比=%.1f%%" % (
    100.0 * np.mean([s["gold_worst_rank"] < 5 for s in st]),
    100.0 * np.mean([s["gold_best_rank"] < 5 for s in st])))
P("  分布重叠: 噪声落进金[min,max]区间的比例 中位=%.1f%%" % (
    100.0 * np.median([s["overlap_frac"] for s in st])))
P("F128_DONE %.0fs" % (time.time() - t0))
