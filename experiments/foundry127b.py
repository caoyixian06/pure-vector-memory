# -*- coding: utf-8 -*-
"""foundry127.py — 集合选择法(用户洞见: 98.5%均值规律→不用排序用切分):
每题候选软对齐分布→二分(谷底/2簇)→高簇=金候选集
指标: 集合含金率(命中)+集合大小+纯度"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry127_results.txt", "w", encoding="utf-8")
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

def valley_cut(scores):
    """谷底切分: 直方图找双峰谷底; 无双峰退top20%"""
    s = np.sort(np.asarray(scores))[::-1]
    if len(s) < 10:
        return max(1, len(s) // 5)
    hist, edges = np.histogram(s, bins=20)
    # 找最高峰(高频簇=噪)与其后更低谷, 再找谷后的次峰
    i_main = int(np.argmax(hist))
    # 从主峰往后找谷
    best_valley = None
    for j in range(i_main + 1, len(hist) - 1):
        if hist[j] < hist[j - 1] and hist[j] <= hist[j + 1]:
            if best_valley is None or hist[j] < hist[best_valley]:
                best_valley = j
    if best_valley is not None and edges[best_valley] < s[0] - 1e-6:
        thr = edges[best_valley]
        k = int((s >= thr).sum())
        if 1 <= k <= max(3, len(s) // 3):
            return max(1, k)
    return max(1, len(s) // 10)

# 置信分层: 分布顶部集中度(top10%均值-中位)为零标签置信分
ALL = []
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
    al = []
    for r in rows:
        r = int(r)
        tidx = REC2TOK.get(r)
        al.append(float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1.0)
    al = np.asarray(al)
    order = np.argsort(-al)
    top5 = [int(rows[i]) for i in order[:5]]
    g_in = sum(1 for i in top5 if i in G)
    g1 = 1 if (order.size and int(rows[order[0]]) in G) else 0
    # 零标签置信分: 顶部集中度
    k10 = max(1, len(al) // 10)
    conf = float(np.sort(al)[-k10:].mean() - np.median(al))
    ALL.append((conf, g_in, g1))

ALL.sort(key=lambda x: -x[0])
n = len(ALL)
P("===== 置信分层(n=%d, 按分布顶部集中度) =====" % n)
for frac, nm in ((0.2, "高置信top20%"), (0.4, "top40%"), (0.6, "top60%"), (1.0, "全部")):
    m = int(n * frac)
    seg = ALL[:m]
    h5 = sum(x[1] for x in seg) / max(1, len(seg))
    h1 = sum(x[2] for x in seg) / max(1, len(seg))
    P("%-12s 命中@5=%.1f%%  top1=%.1f%%  (n=%d)" % (nm, 100 * h5, 100 * h1, len(seg)))
P("F127B_DONE %.0fs" % (time.time() - t0))
