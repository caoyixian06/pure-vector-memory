# -*- coding: utf-8 -*-
"""diag_dims_evn.py — 直接算数字: top50内 证据记录 vs 噪声记录 的逐维对比
①Δ=每维(证据均值-噪声均值), top差异维, 拆半稳定性
②Δ投影判别: A半Δ→B半AUC(对比精排0.921)
③数字全景: D的方差profile/谱/有效秩
④证据指纹 vs 日期指纹重叠
⑤逐题中心化版Δ(消问题间偏移)
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
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
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
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        targets[i] = hits
print("matched:", len(targets), flush=True)

# 标签收集(逐题中心化同时做)
ev_rows, no_rows = [], []
qc_ev_sum = np.zeros(1024, dtype=np.float64)
qc_no_sum = np.zeros(1024, dtype=np.float64)
qc_cnt = 0
for i, hits in targets.items():
    hs = set(hits)
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev_idx = [j for j in top if j in hs or j in tws]
    no_idx = [j for j in top if j not in hs and j not in tws]
    if not ev_idx or len(no_idx) < 5:
        continue
    ev_rows += ev_idx
    no_rows += list(np.random.default_rng(i).choice(no_idx, size=min(10, len(no_idx)), replace=False))
    # 逐题中心化: 减去本题top50均值
    mu = D[top].mean(0)
    qc_ev_sum += (D[ev_idx] - mu).mean(0)
    qc_no_sum += (D[no_idx] - mu).mean(0)
    qc_cnt += 1
ev_rows = np.array(ev_rows); no_rows = np.array(no_rows)
print("EVID记录%d NOISE记录%d(采样)" % (len(ev_rows), len(no_rows)), flush=True)

mu_ev = D[ev_rows].mean(0)
mu_no = D[no_rows].mean(0)
delta = mu_ev - mu_no
absd = np.abs(delta)
print("①Δ(证据-噪声)逐维: |Δ|中位%.4f 最大%.4f; |Δ|>0.02的维数%d/1024" % (
    np.median(absd), absd.max(), (absd > 0.02).sum()), flush=True)
top_dims = np.argsort(-absd)[:20]
print("  top20维:", top_dims.tolist(), flush=True)
print("  对应Δ值:", [round(float(delta[d]), 4) for d in top_dims], flush=True)

# 拆半稳定性
keys = list(targets.keys())
half = len(keys) // 2
def delta_of(ks):
    evs, nos = [], []
    for i in ks:
        hs = set(targets[i])
        tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
        top = TOP50[i]
        evs += [j for j in top if j in hs or j in tws]
        nos += [j for j in top if j not in hs and j not in tws][:8]
    return D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
dA, dB = delta_of(keys[:half]), delta_of(keys[half:])
c = float(dA @ dB / (np.linalg.norm(dA) * np.linalg.norm(dB)))
ov = len(set(np.argsort(-np.abs(dA))[:200]) & set(np.argsort(-np.abs(dB))[:200]))
print("②拆半稳定性: cos(Δ_A,Δ_B)=%.3f top200维重叠=%d/200" % (c, ov), flush=True)

# Δ投影判别(A半Δ→B半记录)
dn = dA / (np.linalg.norm(dA) + 1e-9)
evB, noB = [], []
for i in keys[half:]:
    hs = set(targets[i])
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evB += [j for j in top if j in hs or j in tws]
    noB += [j for j in top if j not in hs and j not in tws][:10]
sev = D[np.array(evB)] @ dn
sno = D[np.array(noB)] @ dn
r = np.argsort(np.concatenate([sev, sno]))
ranks = np.empty_like(r, dtype=np.float64); ranks[r] = np.arange(1, len(r) + 1)
auc = (ranks[:len(sev)].sum() - len(sev) * (len(sev) + 1) / 2) / (len(sev) * len(sno))
print("③Δ投影判别(B半): EVID分%.3f NOISE分%.3f AUC=%.3f (精排=0.921)" % (
    sev.mean(), sno.mean(), auc), flush=True)

# 逐题中心化Δ
qc = (qc_ev_sum - qc_no_sum) / qc_cnt
qcn = qc / (np.linalg.norm(qc) + 1e-9)
sev2 = D[np.array(evB)] @ qcn
sno2 = D[np.array(noB)] @ qcn
r2 = np.argsort(np.concatenate([sev2, sno2]))
rk2 = np.empty_like(r2, dtype=np.float64); rk2[r2] = np.arange(1, len(r2) + 1)
auc2 = (rk2[:len(sev2)].sum() - len(sev2) * (len(sev2) + 1) / 2) / (len(sev2) * len(sno2))
print("④逐题中心化Δ判别: AUC=%.3f" % auc2, flush=True)
print("  中心化Δ与全局Δ的cos=%.3f" % float(qcn @ dn), flush=True)

# ⑤数字全景: 方差profile + 谱
var_profile = D.var(0)
print("⑤D列方差: top5维%s; 方差比=最大/中位=%.1f倍" % (
    np.argsort(-var_profile)[:5].tolist(), var_profile.max() / np.median(var_profile)), flush=True)
U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
eff = (S ** 2).sum() / (S[0] ** 2)
print("  谱: S1=%.0f S10=%.0f S100=%.0f; 有效秩(能量比)=%.0f/1024" % (
    S[0], S[9], S[99], eff), flush=True)
# 日期指纹重叠
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
ans = [str(r["answer"][0]) for r in rows if re.match(r"(?i)^when\b", r["question"])]
out = []
for s in range(0, len(ans), 64):
    out.append(np.asarray(bge.encode(ans[s:s + 64])["dense_vecs"], dtype=np.float32))
Awhen = l2n(np.concatenate(out))
prof = np.abs(Awhen - Awhen.mean(0)).mean(0)
date_dims = set(np.argsort(-prof)[:200].tolist())
ev_dims = set(np.argsort(-absd)[:200].tolist())
print("⑥证据指纹top200 ∩ 日期指纹top200 = %d 个(随机期望~39)" % len(ev_dims & date_dims), flush=True)
