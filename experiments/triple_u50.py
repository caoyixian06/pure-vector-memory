# -*- coding: utf-8 -*-
"""triple_uniqueness2.py — 修复版: 全部指标等长, 组合AUC可算"""
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
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def toks(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
MID2I = {m: i for i, m in enumerate(MID)}
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
_rng = np.random.default_rng(777)
sel50 = list(_rng.choice(len(keys), size=50, replace=False))
keys = [keys[i2] for i2 in sel50]
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
DELTA /= np.linalg.norm(DELTA)

A = emb([r["question"] for r in rows])
X = emb([str(r["answer"][0]) for r in rows])

def cos(u, v):
    return float(u @ v)

def auc(t_, f_):
    t_ = np.asarray(t_, dtype=np.float64)
    f_ = np.asarray(f_, dtype=np.float64)
    x = np.concatenate([t_, f_])
    lb = np.zeros(len(x), dtype=bool); lb[:len(t_)] = True
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = len(t_)
    return (ranks[:np_].sum() - np_ * (np_ + 1) / 2) / (np_ * (len(x) - np_))

metrics = {k: ([], []) for k in ("U1_ac", "U1_ax", "U1_cx", "U2覆盖", "U3对齐词数", "U4Δ分", "U5闭合差")}
count = 0
for i in keys:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    if not ev or len(no) < 2:
        continue
    c, h = ev[0], no[0]
    a, x = A[i], X[i]
    cv, hv = D[c], D[h]
    metrics["U1_ac"][0].append(cos(a, cv)); metrics["U1_ac"][1].append(cos(a, hv))
    metrics["U1_ax"][0].append(cos(a, x)); metrics["U1_ax"][1].append(cos(a, x))  # ax与h无关, 填同值(对照无意义, 跳过AUC)
    metrics["U1_cx"][0].append(cos(cv, x)); metrics["U1_cx"][1].append(cos(hv, x))
    xt = toks(x); at = toks(a)
    xonly = xt - at
    ct = toks(TEXTS[c]); ht = toks(TEXTS[h])
    if xonly:
        u2t = len(xonly & ct) / len(xonly)
        u2f = len(xonly & ht) / len(xonly)
    else:
        u2t = u2f = 0.0
    metrics["U2覆盖"][0].append(u2t); metrics["U2覆盖"][1].append(u2f)
    metrics["U3对齐词数"][0].append(float(len(xonly & ct)))
    metrics["U3对齐词数"][1].append(float(len(xonly & ht)))
    metrics["U4Δ分"][0].append(float(cv @ DELTA)); metrics["U4Δ分"][1].append(float(hv @ DELTA))
    ch = cv - a
    t_ = float((x - a) @ ch / (ch @ ch + 1e-9))
    dline_t = float(np.linalg.norm(x - (a + t_ * ch)))
    hh = hv - a
    t2 = float((x - a) @ hh / (hh @ hh + 1e-9))
    dline_f = float(np.linalg.norm(x - (a + t2 * hh)))
    metrics["U5闭合差"][0].append(-dline_t); metrics["U5闭合差"][1].append(-dline_f)
    count += 1
print("配对(50组子样本):", count, flush=True)

print("== 三元组独特性(真c vs 伪h) ==", flush=True)
zts, zfs = [], []
for k, (t_, f_) in metrics.items():
    if k == "U1_ax":
        continue
    t_ = np.array(t_, dtype=np.float64)
    f_ = np.array(f_, dtype=np.float64)
    a = auc(t_, f_)
    sd = np.concatenate([t_, f_]).std() + 1e-9
    zt = (t_ - np.concatenate([t_, f_]).mean()) / sd
    zf = (f_ - np.concatenate([t_, f_]).mean()) / sd
    zts.append(zt); zfs.append(zf)
    print("  %-10s 真%.3f 伪%.3f AUC=%.3f" % (k, t_.mean(), f_.mean(), a), flush=True)
ct = np.mean(np.stack(zts), axis=0)
cf = np.mean(np.stack(zfs), axis=0)
print("  组合: AUC=%.3f" % auc(ct, cf), flush=True)
print("UNIQ2_DONE", flush=True)
