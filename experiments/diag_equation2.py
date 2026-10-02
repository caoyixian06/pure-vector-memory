# -*- coding: utf-8 -*-
"""diag_equation2.py — 方程解法实验(防过拟合版)
纪律: PRF超参在A半区选, B半区独立验证; Ridge只留低秩当监督参考
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
out = []
for s in range(0, n, 64):
    out.append(np.asarray(bge.encode(Q[s:s + 64])["dense_vecs"], dtype=np.float32))
bQ = l2n(np.concatenate(out))

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
ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])
def half_of(i):
    sid = rows[i].get("sample_id") or ""
    try:
        return int(re.sub(r"\D", "", sid)) % 2
    except Exception:
        return 0
half = np.array([half_of(i) for i, _ in targets])
print("matched:", len(targets), flush=True)

raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
def fused_of(i, sb_row):
    return zs(sb_row) + zs(SQc[i]) + 0.5 * zs(VEXc[i])

def ev_halves(SCf, tag):
    res = {}
    for h in (0, 1):
        sel = np.where(half == h)[0]
        m25 = o5 = cnt = 0
        rsum = 0
        for k in sel:
            i, hits = targets[k]
            s = SCf(i)
            order = np.argsort(-s)
            pos = {x: p for p, x in enumerate(order)}
            rk = min(pos[h2] for h2 in hits) + 1
            rsum += rk; cnt += 1
            if ok[k]:
                o5 += rk <= 5
            else:
                m25 += rk <= 25
        res[h] = (m25, o5, rsum / max(cnt, 1), int((~ok[sel]).sum()), int(ok[sel].sum()))
    print("  %-24s A半: 进25 %3d/%d 进5 %3d/%d 名次%.0f | B半: 进25 %3d/%d 进5 %3d/%d 名次%.0f" % (
        tag, res[0][0], res[0][3], res[0][1], res[0][4], res[0][2],
        res[1][0], res[1][3], res[1][1], res[1][4], res[1][2]), flush=True)
    return res

SBS0 = (l2n(bQ + u2) @ D.T).astype(np.float32)
ev_halves(lambda i: fused_of(i, SBS0[i]), "基线: 融合u2")

# ---- PRF不动点: 全部配置跑双半区, A半选参B半验证 ----
print("== PRF不动点(无监督, A选B验) ==", flush=True)
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
results = {}
for gamma in (0.3, 0.5):
    for k in (3, 5):
        qq = l2n(bQ + u2).copy()
        exp = np.zeros_like(qq)
        for i in range(n):
            top = TOP50[i][np.argsort(-RER[i])[:k]]
            c = D[top].mean(0)
            exp[i] = c / (np.linalg.norm(c) + 1e-9)
        qq = l2n(qq + gamma * exp)
        SBS = (qq @ D.T).astype(np.float32)
        results[(gamma, k)] = ev_halves(lambda i, S=SBS: fused_of(i, S[i]),
                                        "PRF γ=%.1f k=%d" % (gamma, k))
best_A = max(results, key=lambda g: results[g][0][0] + results[g][0][1])
print("A半区最优配置: γ=%.1f k=%d → B半区独立验证见上行" % best_A, flush=True)

# ---- Ridge低秩(监督参考, 不当主力) ----
print("== Ridge低秩(监督参考) ==", flush=True)
def ridge_map(train, test_h, rank, lam=1.0, tag=""):
    X = np.stack([bQ[i] for i, _ in train])
    Y = np.stack([l2n(D[hits].mean(0)[None])[0] for _, hits in train])
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc = X - Xm
    Yc = Y - Ym
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:rank].T @ np.diag(S[:rank] / (S[:rank] ** 2 + lam)) @ (U[:, :rank].T @ Yc)
    sel = np.where(half == test_h)[0]
    m25 = o5 = cnt = 0
    rsum = 0
    for kk in sel:
        i, hits = targets[kk]
        pred = (bQ[i] - Xm) @ W + Ym
        pred /= np.linalg.norm(pred) + 1e-9
        s = D @ pred
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h2] for h2 in hits) + 1
        rsum += rk; cnt += 1
        if ok[kk]:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-24s 进25 %3d 进5 %3d 名次%.0f (n=%d)" % (tag, m25, o5, rsum / max(cnt, 1), cnt), flush=True)

trA = [(i, h) for i, h in targets if half_of(i) == 0]
trB = [(i, h) for i, h in targets if half_of(i) == 1]
for rank in (8, 32):
    ridge_map(trA, 1, rank, tag="rank=%d 训A测B" % rank)
    ridge_map(trB, 0, rank, tag="rank=%d 训B测A" % rank)
