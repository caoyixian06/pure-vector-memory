# -*- coding: utf-8 -*-
"""check_k3_leak.py — 0.855泄漏裁决重跑: 三个口径并列
①K3原版(全集std归一, 有泄漏嫌疑) ②干净版(A半std归一→B半测) ③随机方向对照(应当≈0.5)
重复5个随机拆分, 报均值±幅度
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
bQ = emb([r["question"] for r in rows])

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

def per_item(i):
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:8]
    return ev, no

def auc(x, plab):
    x = np.asarray(x, dtype=np.float64)
    plab = np.asarray(plab, dtype=bool)
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

print("5次随机拆分, 每次报告: Δ均值 | K3泄漏版 | K3干净版 | 随机方向对照", flush=True)
res = {k: [] for k in ("delta", "leak", "clean", "rand")}
for seed in range(5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    A = [keys[i] for i in perm[:len(keys) // 2]]
    B = [keys[i] for i in perm[len(keys) // 2:]]
    evA, noA, evB, noB = [], [], [], []
    for i in A:
        e, n2 = per_item(i)
        evA += e; noA += n2
    for i in B:
        e, n2 = per_item(i)
        evB += e; noB += n2
    evA, noA, evB, noB = map(np.array, (evA, noA, evB, noB))
    dmu = D[evA].mean(0) - D[noA].mean(0)
    dmu /= np.linalg.norm(dmu)
    dva = D[evA].var(0) - D[noA].var(0)
    dva /= np.linalg.norm(dva) + 1e-9
    smuE, smuN = D[evB] @ dmu, D[noB] @ dmu
    svaE, svaN = D[evB] @ dva, D[noB] @ dva
    # Δ均值
    x = np.concatenate([smuE, smuN])
    lab = np.zeros(len(x), dtype=bool)
    lab[:len(smuE)] = True
    res["delta"].append(auc(x, lab))
    # K3泄漏版: std在合并全集上算
    cE = smuE / np.abs(smuE).std() + svaE / np.abs(svaE).std()
    cN = smuN / np.abs(smuE).std() + svaN / np.abs(svaE).std()
    x = np.concatenate([cE, cN])
    res["leak"].append(auc(x, lab))
    # K3干净版: std只用A半区算
    refE, refN = D[evA] @ dmu, D[noA] @ dmu
    rE, rN = D[evA] @ dva, D[noA] @ dva
    s1 = np.abs(np.concatenate([refE, refN])).std() + 1e-9
    s2 = np.abs(np.concatenate([rE, rN])).std() + 1e-9
    cE = smuE / s1 + svaE / s2
    cN = smuN / s1 + svaN / s2
    x = np.concatenate([cE, cN])
    res["clean"].append(auc(x, lab))
    # 随机方向对照
    rnd = rng.standard_normal(1024).astype(np.float32)
    rnd /= np.linalg.norm(rnd)
    xr = np.concatenate([D[evB] @ rnd, D[noB] @ rnd])
    res["rand"].append(auc(xr, lab))
    print("  split%d: Δ=%.3f 泄漏=%.3f 干净=%.3f 随机=%.3f" % (
        seed, res["delta"][-1], res["leak"][-1], res["clean"][-1], res["rand"][-1]), flush=True)
print()
for k, v in res.items():
    v = np.array(v)
    print("%-6s 均值%.3f (min %.3f max %.3f)" % (k, v.mean(), v.min(), v.max()), flush=True)
