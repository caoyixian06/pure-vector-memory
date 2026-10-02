# -*- coding: utf-8 -*-
"""check_narr_axis.py — D1词面差异→向量叙事轴(无标签, 纯嵌入)
N1 从D1词表构造叙事轴: narr = mean(叙事词向量) − mean(反应词向量)
N2 narr与Δ的cos(Δ是它的化身吗?)
N3 narr判别证据/噪声AUC(留出) vs Δ 0.802
N4 narr+精排联合 vs Δ+精排
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

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
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
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50 = z1["TOP50"]

evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
evs, nos = np.array(evs), np.array(nos)
k = len(evs) // 2
evA, evB, noA, noB = evs[:k], evs[k:], nos[:k], nos[k:]
DELTA = D[evA].mean(0) - D[noA].mean(0)
DELTA /= np.linalg.norm(DELTA)

# N1: 叙事轴(词表来自D1的观察, 但方向是嵌入固有的, 与金标签无关)
narr_words = ["last", "week", "ago", "yesterday", "took", "went", "made", "started",
              "finished", "visited", "moved", "bought", "played", "watched", "read",
              "graduated", "worked", "lived", "traveled", "adopted", "won", "signed", "joined", "planned"]
fluff_words = ["awesome", "great", "wow", "thanks", "thank", "support", "glad", "proud",
               "amazing", "cool", "nice", "love", "happy", "excited", "sorry", "congrats",
               "enjoy", "fun", "best", "sweet"]
Vn = emb(["the " + w for w in narr_words])
Vf = emb(["the " + w for w in fluff_words])
narr = Vn.mean(0) - Vf.mean(0)
narr /= np.linalg.norm(narr)
print("N1 叙事轴已建(%d vs %d 锚词)" % (len(narr_words), len(fluff_words)), flush=True)
print("N2 cos(narr, Δ) = %.3f" % float(narr @ DELTA), flush=True)

def auc(x, plab):
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

PROJ_narr = (D @ narr).astype(np.float32)
PROJ_delta = (D @ DELTA).astype(np.float32)
sE, sN = PROJ_narr[evB], PROJ_narr[noB]
x = np.concatenate([sE, sN])
lab = np.zeros(len(x), dtype=bool); lab[:len(sE)] = True
print("N3 叙事轴判别(留出B): EVID %.3f NOISE %.3f AUC=%.3f   [Δ=0.802]" % (
    sE.mean(), sN.mean(), auc(x, lab)), flush=True)
# 各密度层(检验是否同样误伤低密度碎片)
info = ["My favorite book is The Pragmatic Programmer by Hunt.", "We visited Kyoto in April 2019 for cherry blossoms.", "She works as a pediatric nurse at the children hospital.", "The concert tickets cost 45 dollars each last Friday.", "My brother graduated from MIT with computer science degree."] * 6
fluff = ["That sounds really great and awesome!", "Thanks so much for your support friend!", "Wow I cannot believe it amazing!", "Have a wonderful day and take care!", "It was so much fun hanging out together!"] * 6
ia = emb(info).mean(0) - emb(fluff).mean(0)
ia /= np.linalg.norm(ia)
dens_e = PROJ_narr[evB]  # 用narr自己的投影分层(自洽)
q1, q2 = np.percentile(dens_e, [33, 66])
bkt = np.where(dens_e < q1, 0, np.where(dens_e < q2, 1, 2))
for b, name in ((0, "低"), (1, "中"), (2, "高")):
    se = evB[bkt == b]
    if len(se) < 30:
        continue
    x = np.concatenate([PROJ_narr[se], PROJ_narr[noB]])
    lb = np.zeros(len(x), dtype=bool); lb[:len(se)] = True
    print("   %s密度层: AUC=%.3f" % (name, auc(x, lb)), flush=True)

# N4: 排序融合对比(臂B底 + Δ vs + narr)
zc = np.load(HERE + "/fusion_cache.npz")
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ = emb([r["question"] for r in rows]).mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQ = emb([r["question"] for r in rows])
bQu = l2n(bQ + u2)
SBS = (bQu @ D.T).astype(np.float32)
RER = z1["RER"]
def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)
def base_row(i):
    fused = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-fused)[:50]
    row = fused.copy()
    row[top] = np.linspace(50, 1, 50)
    return row
BHALF = [i for i in keys[len(keys) // 2:]]
print("N4 臂B底+叙事轴/Δ(留出B半):", flush=True)
for w, tag in ((0.5, "narr"),):
    m25 = o5 = 0
    for i in BHALF:
        row = base_row(i) + w * zs(PROJ_narr)
        hs = targets[i]
        tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
        order = np.argsort(-row)
        pos = {x2: p for p, x2 in enumerate(order)}
        rk = min(pos[h2] for h2 in list(hs) + list(tws)) + 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  臂B+%s w=0.5: 错题进25 %d 对题进5 %d" % (tag, m25, o5), flush=True)
for w in (0.3, 0.6):
    m25 = o5 = 0
    for i in BHALF:
        row = base_row(i) + w * zs(PROJ_delta)
        hs = targets[i]
        tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
        order = np.argsort(-row)
        pos = {x2: p for p, x2 in enumerate(order)}
        rk = min(pos[h2] for h2 in list(hs) + list(tws)) + 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  臂B+Δ w=%.1f: 错题进25 %d 对题进5 %d" % (w, m25, o5), flush=True)
