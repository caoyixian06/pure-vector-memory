# -*- coding: utf-8 -*-
"""diag_fusion_split.py — 最优组合的拆半验证: 组合在两个半区各自是否仍胜基线"""
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
out = []
for s in range(0, n, 64):
    out.append(np.asarray(bge.encode(Q[s:s + 64])["dense_vecs"], dtype=np.float32))
bQ = l2n(np.concatenate(out))

z = np.load(HERE + "/fusion_cache.npz")
SB, SQ, VEX = z["SB"], z["SQ"], z["VEX"]
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        rr = json.loads(l)
        REC[rr.get("memory_id")] = rr
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

targets = []
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
        targets.append((i, hits))

raw_idx = [i for i in range(len(MID)) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQs = l2n(bQ + u2)
SBS = (bQs @ D.T).astype(np.float32)

def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def half_of(i):
    sid = rows[i].get("sample_id") or ""
    try:
        return int(re.sub(r"\D", "", sid)) % 2
    except Exception:
        return 0

def ev(SC_builder, mask, tag):
    m25 = o5 = cnt = 0
    rsum = 0
    for i, hits in targets:
        if half_of(i) != mask:
            continue
        s = SC_builder(i)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        rsum += rk; cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-24s 错题进25 %3d  对题进5 %3d  均名次%.0f" % (tag, m25, o5, rsum / cnt), flush=True)

for mask in (0, 1):
    print("== 半区%d(会话号%%2==%d) ==" % (mask, mask), flush=True)
    ev(lambda i: SB[i], mask, "基线纯BGE")
    ev(lambda i: zs(SB[i]) + zs(SQ[i]) + 0.5 * zs(VEX[i]), mask, "组合(无手术)")
    ev(lambda i: zs(SBS[i]) + zs(SQ[i]) + 0.5 * zs(VEX[i]), mask, "组合(u2+a1+b0.5ex)")
    ev(lambda i: zs(SBS[i]) + 0.5 * zs(SQ[i]) + 0.5 * zs(VEX[i]), mask, "组合(u2+a0.5+b0.5ex)")
