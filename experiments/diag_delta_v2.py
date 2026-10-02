# -*- coding: utf-8 -*-
"""diag_delta_v2.py — Δ干净复测+方差指纹(r36底盘, 拆半留出)
K1 Δ均值指纹(0.80复现?)
K2 方差指纹(二阶矩, 新)
K3 均值+方差联合
K4 在r36底盘排序上叠Δ的代理增益
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
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
bQ = emb(Q)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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
MID2I = {m: i for i, m in enumerate(MID)}
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits:
        targets[i] = hits
keys = sorted(targets.keys())
half = len(keys) // 2
print("matched:", len(targets), flush=True)

def per_item(i):
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    # r36底盘排序: 融合+精排(重新构造: 用fusion_cache+stage1)
    return hs, tws

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQu = l2n(bQ + u2)
SBS = (bQu @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
def armB_row(i):
    fused = zs(SBS[i]) + zs(SQc[i]) + 0.5 * zs(VEXc[i])
    top = np.argsort(-fused)[:50]
    sc = RER[i]
    row = fused.copy()
    row[top] = np.linspace(50, 1, 50)  # 精排主导(臂B口径近似)
    return row
ARM = np.stack([armB_row(i) for i in range(n)])

def collect_items(sel_keys):
    """返回 (ev_ids, no_ids) 列表"""
    evs, nos = [], []
    for i in sel_keys:
        hs, tws = per_item(i)
        top = TOP50[i]
        evs += [j for j in top if j in hs or j in tws]
        nos += [j for j in top if j not in hs and j not in tws][:8]
    return np.array(evs), np.array(nos)

evA, noA = collect_items(keys[:half])
evB, noB = collect_items(keys[half:])
print("pool A: ev%d no%d | pool B: ev%d no%d" % (len(evA), len(noA), len(evB), len(noB)), flush=True)

muA_e, muA_n = D[evA].mean(0), D[noA].mean(0)
vaA_e, vaA_n = D[evA].var(0), D[noA].var(0)
delta_mu = muA_e - muA_n
delta_mu /= np.linalg.norm(delta_mu)
delta_va = vaA_e - vaA_n
delta_va /= np.linalg.norm(delta_va)

def auc(x, p, q):
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=bool)
    q = np.asarray(q, dtype=bool)
    r = np.argsort(x[p].tolist() + x[q].tolist())
    ranks = np.empty(len(r), dtype=np.float64)
    ranks[r] = np.arange(1, len(r) + 1)
    return (ranks[:p.sum()].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * q.sum())

sB_mu = D[evB] @ delta_mu
sB_no = D[noB] @ delta_mu
allx = np.concatenate([sB_mu, sB_no])
pp = np.zeros(len(allx), dtype=bool); pp[:len(sB_mu)] = True
print("K1 Δ均值指纹(留出B): EVID %.3f NOISE %.3f AUC=%.3f" % (
    sB_mu.mean(), sB_no.mean(), auc(allx, pp, ~pp)), flush=True)
sBv_e = D[evB] @ delta_va
sBv_n = D[noB] @ delta_va
allv = np.concatenate([sBv_e, sBv_n])
print("K2 Δ方差指纹(留出B): EVID %.3f NOISE %.3f AUC=%.3f" % (
    sBv_e.mean(), sBv_n.mean(), auc(allv, pp, ~pp)), flush=True)
comb_e = sB_mu / (np.abs(sB_mu).std() + 1e-9) + sBv_e / (np.abs(sBv_e).std() + 1e-9)
comb_n = sB_no / (np.abs(sB_mu).std() + 1e-9) + sBv_n / (np.abs(sB_mu).std() + 1e-9)
allc = np.concatenate([comb_e, comb_n])
print("K3 均值+方差联合: AUC=%.3f" % auc(allc, pp, ~pp), flush=True)

# K4: 在臂B排序上叠Δ(留出Δ), 代理指标
PROJ = (D @ delta_mu).astype(np.float32)
for w in (0.0, 0.3, 0.6):
    m25 = o5 = 0
    rsum = 0
    for i in keys[half:]:
        hs, tws = per_item(i)
        row = ARM[i] + w * zs(PROJ)
        order = np.argsort(-row)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in list(hs) + list(tws)) + 1
        rsum += rk
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("K4 臂B+Δw=%.1f(B半): 错题进25 %d 对题进5 %d 均名次%.0f" % (
        w, m25, o5, rsum / len(keys[half:])), flush=True)
