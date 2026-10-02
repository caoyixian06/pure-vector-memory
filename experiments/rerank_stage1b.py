# -*- coding: utf-8 -*-
"""rerank_stage1b.py — 阶段1补充: ①指针型证据假象量化 ②隔壁turn记分(扩展命中) ③残余错题分型"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)
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

def next_turn(j):
    if j + 1 < N and KIND[j + 1] == "raw" and re.sub(r"_(rbak\d+k|m\d+)$", "", MID[j + 1]) == re.sub(r"_(rbak\d+k|m\d+)$", "", MID[j]):
        return j + 1
    return None

COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
def cls(t):
    if t.rstrip().endswith("?"):
        return "问句"
    if COUR.search(t):
        return "寒暄"
    return "陈述"

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
ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])

# 重建融合分与精排名次
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
out = []
for s in range(0, n, 64):
    out.append(np.asarray(bge.encode(Q[s:s + 64])["dense_vecs"], dtype=np.float32))
bQ = l2n(np.concatenate(out))
del bge
import torch
torch.cuda.empty_cache()
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
zc = np.load(HERE + "/fusion_cache.npz")
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
FUSED = np.stack([zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i]) for i in range(n)])

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
def rerank_score(i):
    s = FUSED[i].copy()
    top = TOP50[i]
    order_inside = np.argsort(-RER[i])
    s[top[order_inside]] = np.linspace(50, 1, 50)
    return s

def ranks(score_fn, ext):
    rk = []
    for k, (i, hits) in enumerate(targets):
        hs = list(hits)
        if ext:
            for h in hits:
                nx = next_turn(h)
                if nx is not None:
                    hs.append(nx)
        s = score_fn(i)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk.append(min(pos[h] for h in hs) + 1)
    return np.array(rk)

rkF = ranks(lambda i: FUSED[i], False)
rkR = ranks(rerank_score, False)
rkFx = ranks(lambda i: FUSED[i], True)
rkRx = ranks(rerank_score, True)
def m(rk):
    return ((rk[~ok] <= 25).sum(), (rk[ok] <= 5).sum(), rk.mean())
print("标准口径   融合%d/%d/%.0f → 精排%d/%d/%.0f" % (m(rkF) + m(rkR)), flush=True)
print("扩展口径(+隔壁turn) 融合%d/%d/%.0f → 精排%d/%d/%.0f" % (m(rkFx) + m(rkRx)), flush=True)

# 残余错题(精排后仍>25)的证据分型
from collections import Counter
c = Counter()
for k in range(len(targets)):
    if rkR[k] > 25 and not ok[k]:
        c[cls(TEXTS[targets[k][1][0]])] += 1
print("精排后仍错题的证据句分型:", dict(c), flush=True)

# 指针型证据(问句)占比: 全体与残余
allc = Counter(cls(TEXTS[t[1][0]]) for t in targets)
print("全体1376证据句分型:", dict(allc), flush=True)

# 精排把隔壁turn抬进top5的例子
print("=== 精排抬隔壁turn例 ===", flush=True)
shown = 0
for k in range(len(targets)):
    if shown >= 3:
        break
    i, hits = targets[k]
    nx = next_turn(hits[0])
    if nx is None:
        continue
    posR = {x: p for p, x in enumerate(np.argsort(-rerank_score(i)))}
    posF = {x: p for p, x in enumerate(np.argsort(-FUSED[i]))}
    if posR.get(nx, 999) <= 5 and posF.get(nx, 999) > 5:
        print("Q:", Q[i][:75], flush=True)
        print("  证据指针:", TEXTS[hits[0]][:70], flush=True)
        print("  隔壁答案: 名次%d→%d |" % (posF.get(nx, -1) + 1, posR.get(nx, -1) + 1), TEXTS[nx][:80], flush=True)
        shown += 1
if shown == 0:
    print("(无此类例)", flush=True)
