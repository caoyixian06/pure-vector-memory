# -*- coding: utf-8 -*-
"""foundry131.py — 金证据vs噪声在四个空间的全面差异解剖(用户令):
1024记录向量/256记录向量/上下文词向量/静态词向量
逐维差/分布参数/逐题一致性——找出金在哪个空间和噪声真正不同"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry131_results.txt", "w", encoding="utf-8")
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
QW = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
Q256 = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WH.keys())
WV = l2n(np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
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

def auc(pos, neg):
    from bisect import bisect_left
    allv = sorted(pos + neg)
    rp = sum(bisect_left(allv, v) + 1 for v in pos)
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

# 逐题收集四个空间的金/噪向量对
G1024, N1024 = [], []   # 记录向量差
G256, N256 = [], []
Gctx, Nctx = [], []     # 上下文对齐分
Gstat, Nstat = [], []   # 静态词向量软对齐分
d1024_all = []          # 金噪逐维差的累加
d256_all = []
rng = np.random.RandomState(0)
nq = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    cq1 = D @ X[k_i]
    sm = {}
    for cv in UCONVS:
        m2 = CONVKEY == cv
        if m2.any():
            sm[cv] = float(cq1[m2].max())
    top_conv = max(sm, key=lambda c: sm[c])
    rows = np.where((CONVKEY == top_conv) & np.array(["_rbak" not in MID[i] for i in range(len(MID))]))[0]
    if len(rows) < 10:
        continue
    gset = set(i for i in rows.tolist() if i in G)
    nset = [i for i in rows.tolist() if i not in G]
    if not gset or len(nset) < 10:
        continue
    npick = rng.choice(nset, size=min(30, len(nset)), replace=False).tolist()
    qctx = ctx_tokens(Q[qa]["question"])
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    qidx = [W2I[w] for w in qstems if w in W2I]
    QV = WV[qidx] if qidx else None
    for i in gset:
        G1024.append(float(D[i] @ X[k_i]))
        G256.append(float(QW[i] @ Q256[k_i]))
        tidx = REC2TOK.get(i)
        Gctx.append(float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1)
        if QV is not None:
            sws = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()) if len(w) > 2)
            sidx = [W2I[w] for w in sws if w in W2I]
            Gstat.append(float((WV[sidx] @ QV.T).max(axis=1).mean()) if sidx else -1)
        d1024_all.append(D[i])
    for i in npick:
        N1024.append(float(D[i] @ X[k_i]))
        N256.append(float(QW[i] @ Q256[k_i]))
        tidx = REC2TOK.get(i)
        Nctx.append(float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1)
        if QV is not None:
            sws = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()) if len(w) > 2)
            sidx = [W2I[w] for w in sws if w in W2I]
            Nstat.append(float((WV[sidx] @ QV.T).max(axis=1).mean()) if sidx else -1)
    nq += 1
    if k_i % 300 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))

P("\n===== 金vs噪 四空间差异解剖(n=%d题) =====" % nq)
P("空间              金均值    噪均值    gap     AUC")
P("1024记录cos      %.4f   %.4f   %.4f   %.3f" % (
    np.mean(G1024), np.mean(N1024), np.mean(G1024) - np.mean(N1024), auc(G1024, N1024)))
P("256记录cos       %.4f   %.4f   %.4f   %.3f" % (
    np.mean(G256), np.mean(N256), np.mean(G256) - np.mean(N256), auc(G256, N256)))
P("上下文词向量对齐  %.4f   %.4f   %.4f   %.3f" % (
    np.mean(Gctx), np.mean(Nctx), np.mean(Gctx) - np.mean(Nctx), auc(Gctx, Nctx)))
if Gstat and Nstat:
    P("静态词向量对齐    %.4f   %.4f   %.4f   %.3f" % (
        np.mean(Gstat), np.mean(Nstat), np.mean(Gstat) - np.mean(Nstat), auc(Gstat, Nstat)))

# 逐维差异(哪个维度金噪差最大)
gmean = np.mean([D[i] for i in range(len(MID)) if MID[i] and i in set()], axis=0) if False else None
# 用逐题收集的d1024_all做金质心 vs 全库质心
gold_centroid = l2n(np.mean(d1024_all, axis=0, keepdims=True))[0]
lib_centroid = l2n(D.mean(0, keepdims=True))[0]
delta = gold_centroid - lib_centroid
top_dims = np.argsort(-np.abs(delta))[:15]
P("\n1024空间·金质心vs库质心的top差异维:")
P("  " + ",".join("d%d(%.3f)" % (d, delta[d]) for d in top_dims))
P("  |Δ|>0.01的维数: %d/1024" % int((np.abs(delta) > 0.01).sum()))
# 256空间
g256_rows = []
for k_i, qa in enumerate(IDS):
    pass
P("F131_DONE %.0fs" % (time.time() - t0))
