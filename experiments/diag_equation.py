# -*- coding: utf-8 -*-
"""diag_equation.py — 把检索看作方程: 未知数x=答案/证据向量
已知: 问题向量q, 库矩阵D, 实测几何(q·e=0.49, a·e=0.56, 半球-0.454)
解法1(无监督不动点): q_{t+1} = normalize(q_t + γ·normalize(top-k记录质心)) — 答案必须由库张成
解法2(低秩回归): ê = W q + b, W用一半会话的问题→证据向量拟合, 另一半验证
指标: 错题进25 / 对题进5 / 均名次 (基线: 纯BGE 210/342/97, 融合u2 378/634/45)
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
print("matched:", len(targets), flush=True)

raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
zc = np.load(HERE + "/fusion_cache.npz")
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
def fused_of(i, sb_row):
    return zs(sb_row) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])

def ev(SCf, tag):
    m25 = o5 = cnt = 0
    rsum = 0
    for k, (i, hits) in enumerate(targets):
        s = SCf(i, k)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        rsum += rk; cnt += 1
        if ok[k]:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-30s 错题进25 %3d  对题进5 %3d  均名次%.0f" % (tag, m25, o5, rsum / cnt), flush=True)
    return m25, o5

SBS0 = (l2n(bQ + u2) @ D.T).astype(np.float32)
ev(lambda i, k: fused_of(i, SBS0[i]), "基线: 融合u2")

# ---- 解法1: 不动点迭代(答案向量=问题+库的反复自洽) ----
print("== 解法1: PRF不动点(无监督) ==", flush=True)
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
QROW = {rows[i]["qa_id"]: i for i in range(n)}
def prf_iter(gamma, k, iters, use_rer):
    qq = l2n(bQ + u2).copy()
    for _ in range(iters):
        exp = np.zeros_like(qq)
        for i in range(n):
            if use_rer:
                top = TOP50[i][np.argsort(-RER[i])[:k]]
            else:
                sc = fused_of(i, D @ qq[i])
                top = np.argsort(-sc)[:k]
            c = D[top].mean(0)
            exp[i] = c / (np.linalg.norm(c) + 1e-9)
        qq = l2n(qq + gamma * exp)
    SBS = (qq @ D.T).astype(np.float32)
    return SBS
for gamma in (0.3, 0.5):
    for k in (3, 5):
        SBS = prf_iter(gamma, k, 1, True)
        ev(lambda i, kk, S=SBS: fused_of(i, S[i]), "PRF reranktop γ=%.1f k=%d" % (gamma, k))

# ---- 解法2: 低秩回归 q→证据向量(拆半) ----
print("== 解法2: 低秩ridge回归(拆半交叉) ==", flush=True)
def half_of(i):
    sid = rows[i].get("sample_id") or ""
    try:
        return int(re.sub(r"\D", "", sid)) % 2
    except Exception:
        return 0

def ridge_map(train, test, rank, lam=1.0, tag=""):
    X = np.stack([bQ[i] for i, _ in train])
    Y = np.stack([l2n(D[hits].mean(0)[None])[0] for _, hits in train])
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - Xm, Y - Ym
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = (U[:, :rank] * (S[:rank] / (S[:rank] ** 2 + lam))) @ (U[:, :rank].T @ Yc)
    def score(i):
        pred = W.T @ (bQ[i] - Xm) + Ym
        pred /= np.linalg.norm(pred) + 1e-9
        return D @ pred
    m25 = o5 = cnt = 0
    rsum = 0
    for k, (i, hits) in enumerate(targets):
        if half_of(i) != test:
            continue
        s = score(i)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        rsum += rk; cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-30s 错题进25 %3d  对题进5 %3d  均名次%.0f (n=%d)" % (tag, m25, o5, rsum / max(cnt, 1), cnt), flush=True)

trA = [(i, h) for i, h in targets if half_of(i) == 0]
trB = [(i, h) for i, h in targets if half_of(i) == 1]
for rank in (8, 32, 128):
    ridge_map(trA, 1, rank, tag="rank=%d 训A测B" % rank)
    ridge_map(trB, 0, rank, tag="rank=%d 训B测A" % rank)
