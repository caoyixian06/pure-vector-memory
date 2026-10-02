# -*- coding: utf-8 -*-
"""check_delta_lowdens.py — 低密度证据上Δ还灵吗?
把证据按信息密度(与寒暄轴投影+长度)分层, 各层测Δ的判别力(AUC)
若低密度层AUC仍>0.7 → 密度只是成分之一, Δ还编码了别的
若低密度层AUC≈0.5 → Δ误伤枚举碎片, 需要门控(枚举/短句时降权Δ)
"""
import io, json, os, sys, re, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
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
# 拆半: 前半算Δ, 后半测
k = len(evs) // 2
evA, evB, noA, noB = evs[:k], evs[k:], nos[:k], nos[k:]
DELTA = D[evA].mean(0) - D[noA].mean(0)
DELTA /= np.linalg.norm(DELTA)

info = ["My favorite book is The Pragmatic Programmer by Hunt.", "We visited Kyoto in April 2019 for cherry blossoms.", "She works as a pediatric nurse at the children hospital.", "The concert tickets cost 45 dollars each last Friday.", "My brother graduated from MIT with computer science degree."] * 6
fluff = ["That sounds really great and awesome!", "Thanks so much for your support friend!", "Wow I cannot believe it amazing!", "Have a wonderful day and take care!", "It was so much fun hanging out together!"] * 6
ia = emb(info).mean(0) - emb(fluff).mean(0)
ia /= np.linalg.norm(ia)
PROJ_ALL = (D @ ia).astype(np.float32)

# 证据按密度分三层
dens_e = PROJ_ALL[evB]
q1, q2 = np.percentile(dens_e, [33, 66])
dens_bucket = np.where(dens_e < q1, 0, np.where(dens_e < q2, 1, 2))

def auc(x, plab):
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

for b, name in ((0, "低密度证据"), (1, "中密度证据"), (2, "高密度证据")):
    sel_e = evB[dens_bucket == b]
    if len(sel_e) < 30:
        continue
    # 对照噪声: 全体噪声(固定), 看该层证据的Δ得分
    sE = D[sel_e] @ DELTA
    sN = D[noB] @ DELTA
    x = np.concatenate([sE, sN])
    lab = np.zeros(len(x), dtype=bool)
    lab[:len(sE)] = True
    print("%s(n=%d, 均长%.0f字): Δ判别AUC=%.3f" % (
        name, len(sel_e), np.mean([len(TEXTS[j]) for j in sel_e]), auc(x, lab)), flush=True)

# 同口径下全体对照
sE = D[evB] @ DELTA
sN = D[noB] @ DELTA
x = np.concatenate([sE, sN])
lab = np.zeros(len(x), dtype=bool)
lab[:len(sE)] = True
print("全体证据对照: AUC=%.3f" % auc(x, lab), flush=True)
# 枚举题的碎片证据单独看
import collections
enum_rows = [i for i in keys if ("," in str(rows[i]["answer"][0])) or (" and " in str(rows[i]["answer"][0]).lower())]
print("枚举题数:", len(enum_rows), flush=True)
