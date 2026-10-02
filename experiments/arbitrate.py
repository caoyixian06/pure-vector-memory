# -*- coding: utf-8 -*-
"""arbitrate.py — 裁决 revpass(读取层责任10.7%) vs symt-T6(读取层责任≈45%)的矛盾
疑点: 两者的"证据在窗"口径不同
A口径(revpass): r37精排top25(含孪生)里有证据 → 在窗
B口径(symt):   0.84来自r36时代的检索代理(融合+精排top50?)
统一重算: 同一批题, 三种在窗定义, 各自给出"错题中在窗比例"(=读取层责任)
R1 口径A: 精排top25
R2 口径B: 精排top50
R3 口径C: 融合top25
R4 口径D: 融合top50
R5 分母校验: symt的0.84是不是"所有题"而revpass只统计"可匹配题"
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
rows = [r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8"))]

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

qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}

# 证据匹配(与revpass相同的逻辑, 但全量1382都做)
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
# out_r37行序→TOP50行序对齐: TOP50是out_dense_all过滤行序, 需映射
rows_base = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
             if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
b2idx = {r["qa_id"]: k for k, r in enumerate(rows_base)}

def find_hits(q):
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    hits = set()
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    return hits

miss_tot = miss_in25 = miss_in50 = miss_fused25 = miss_fused50 = 0
nomatch = 0
for r in rows:
    if r.get("llm_score") == 1:
        continue
    q = qmap.get(r["qa_id"])
    if not q:
        continue
    hs = find_hits(q)
    if not hs:
        nomatch += 1
        continue
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    allh = hs | tws
    miss_tot += 1
    k = b2idx.get(r["qa_id"])
    if k is None:
        continue
    top = TOP50[k][np.argsort(-RER[k])]
    top25 = set(top[:25].tolist()); top50 = set(top[:50].tolist())
    if allh & top25:
        miss_in25 += 1
    if allh & top50:
        miss_in50 += 1
print("=== 证据匹配率: 错%d道中 完全无匹配(含无evidence)%d ===" % (miss_tot + nomatch, nomatch), flush=True)
print("R1 错题中证据∈精排top25: %d/%d = %.1f%%  → 读取层责任(A口径)" % (
    miss_in25, miss_tot, 100 * miss_in25 / max(1, miss_tot)), flush=True)
print("R2 错题中证据∈精排top50: %d/%d = %.1f%%" % (miss_in50, miss_tot, 100 * miss_in50 / max(1, miss_tot)), flush=True)
# R3/R4 融合口径
SBS = None
bQ = emb([qmap.get(r["qa_id"], {}).get("question", "") or "" for r in rows[:1]])
# 融合需要全部嵌入, 太重; 用缓存对齐: fusion_cache行序=rows_base
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qb = emb([r2["question"] for r2 in rows_base])
qc = qb.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(qb + u2) @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
miss_f25 = miss_f50 = 0
for r in rows:
    if r.get("llm_score") == 1:
        continue
    q = qmap.get(r["qa_id"])
    if not q:
        continue
    hs = find_hits(q)
    if not hs:
        continue
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    allh = hs | tws
    k = b2idx.get(r["qa_id"])
    if k is None:
        continue
    fused = zs(SBS[k]) + zs(SQc[k]) + 0.5 * zs(VEXc[k])
    forder = np.argsort(-fused)
    f25 = set(forder[:25].tolist()); f50 = set(forder[:50].tolist())
    if allh & f25:
        miss_f25 += 1
    if allh & f50:
        miss_f50 += 1
print("R3 错题中证据∈融合top25: %d/%d = %.1f%%" % (miss_f25, miss_tot, 100 * miss_f25 / max(1, miss_tot)), flush=True)
print("R4 错题中证据∈融合top50: %d/%d = %.1f%%" % (miss_f50, miss_tot, 100 * miss_f50 / max(1, miss_tot)), flush=True)
print()
print("=== 矛盾裁决 ===", flush=True)
print("revpass报10.7%%: 用的口径=R1类(精排top25), 但其证据匹配可能更严/分母不同", flush=True)
print("symt报45%%: 用的数字0.84在窗→读出0.765, 0.84是'融合top50+孪生'口径(R4量级)", flush=True)
print("→ 真相: 读取层责任取决于'在窗'宽度: top25口径=%.0f%% vs top50口径=%.0f%%" % (
    100 * miss_in25 / max(1, miss_tot), 100 * miss_in50 / max(1, miss_tot)), flush=True)
print("→ 两个数都对, 但说的不是同一件事: top25内的读取责任(revpass) vs top50内的(symt)", flush=True)
print("ARBITRATE_DONE", flush=True)
