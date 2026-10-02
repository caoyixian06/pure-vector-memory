# -*- coding: utf-8 -*-
"""diag_fulleq.py — 全约束方程: 把所有已测已知条件全部叠进一个得分
C1 检索先验: 融合(BGE-u2+PRF松弛 + qwen256 + 0.5词票)
C2 精排读对: rerank z分(top50外置底)
C3 邻turn指针: top5里的问句 -> 下一turn加分(答案在隔壁)
C4 孪生互证: raw与summary twin互相投票(84% twin排名更高)
纪律: 权重在A半区扫, B半区验证
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
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

raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# C1: PRF松弛后的融合
qq = l2n(bQ + u2).copy()
exp = np.zeros_like(qq)
for i in range(n):
    top = TOP50[i][np.argsort(-RER[i])[:3]]
    c = D[top].mean(0)
    exp[i] = c / (np.linalg.norm(c) + 1e-9)
qq = l2n(qq + 0.5 * exp)
SBS = (qq @ D.T).astype(np.float32)
BASE = np.stack([zs(SBS[i]) + zs(SQc[i]) + 0.5 * zs(VEXc[i]) for i in range(n)])

# C2: 精排z分
ZRER = np.zeros((n, N), dtype=np.float32)
for i in range(n):
    v = np.full(N, float(RER[i].min()) - 1.0, dtype=np.float32)
    v[TOP50[i]] = RER[i]
    ZRER[i] = zs(v)

# C3: 邻turn票(top5问句→下一turn)
ADJ = np.zeros((n, N), dtype=np.float32)
for i in range(n):
    order = np.argsort(-BASE[i])[:5]
    for j in order:
        t = TEXTS[j].rstrip()
        if t.endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
            ADJ[i][j + 1] += 1.0

# C4: 孪生互证票
TWINV = np.zeros((n, N), dtype=np.float32)
for i in range(n):
    order = np.argsort(-BASE[i])[:50]
    for r0, j in enumerate(order):
        tw = TWIN.get(j)
        if tw is not None:
            TWINV[i][tw] += max(0.0, 1.0 - r0 / 50.0)

def ev(SC, sel):
    m25 = o5 = cnt = 0
    rsum = 0
    for k in sel:
        i, hits = targets[k]
        s = SC[i]
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h2] for h2 in hits) + 1
        rsum += rk; cnt += 1
        if ok[k]:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    return m25, o5, rsum / max(cnt, 1)

selA = np.where(half == 0)[0]
selB = np.where(half == 1)[0]
bA = ev(BASE, selA); bB = ev(BASE, selB)
print("C1基线(PRF融合): A %d/%d/%.0f | B %d/%d/%.0f" % (bA + bB), flush=True)

best = None
for wr in (0.5, 1.0):
    for wa in (0.25, 0.5):
        for wt in (0.1, 0.25):
            SC = BASE + wr * ZRER + wa * ADJ + wt * TWINV
            a = ev(SC, selA)
            score = a[0] + a[1]
            if best is None or score > best[0]:
                best = (score, wr, wa, wt, a, ev(SC, selB))
            b = ev(SC, selB)
            print("  wr=%.1f wa=%.2f wt=%.2f | A %d/%d/%.0f | B %d/%d/%.0f" % (
                (wr, wa, wt) + a + b), flush=True)
_, wr, wa, wt, aR, bR = best
print("A半最优: wr=%.1f wa=%.2f wt=%.2f → B半独立验证: %d/%d/%.0f" % (wr, wa, wt, bR), flush=True)
print("全量(A+B): 错题进25 %d 对题进5 %d 均名次%.0f" % (
    aR[0] + bR[0], aR[1] + bR[1], (aR[2] + bR[2]) / 2), flush=True)
print("(参照: 融合基线378/634/45, 融合+精排405/720/42, +PRF 390/667/41)", flush=True)
