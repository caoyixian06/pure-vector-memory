# -*- coding: utf-8 -*-
"""rerank_stage1.py — 阶段1: 融合top50 → 精排重排 → 闸门G1判定 + 占座者位移 + 拆半 + 退化例"""
import io, json, os, sys, re, time
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

# ---- 问题BGE嵌入(先做, 完事释放显存) ----
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
print("q embedded", flush=True)

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
print("matched:", len(targets), flush=True)

# ---- 融合分 ----
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
zc = np.load(HERE + "/fusion_cache.npz")
SQ, VEX = zc["SQ"], zc["VEX"]
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
FUSED = np.stack([zs(SBS[i]) + zs(SQ[i]) + 0.5 * zs(VEX[i]) for i in range(n)])

def ev_ranks(score_fn):
    rk = []
    for i, hits in targets:
        s = score_fn(i)
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk.append(min(pos[h] for h in hits) + 1)
    return np.array(rk)

def metrics(rk):
    ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])
    return int((rk[~ok] <= 25).sum()), int((rk[ok] <= 5).sum()), float(rk.mean())

rk_fused = ev_ranks(lambda i: FUSED[i])
m0 = metrics(rk_fused)
print("纯融合: 错题进25 %d 对题进5 %d 均名次%.0f" % m0, flush=True)

# ---- 精排 top50 ----
from FlagEmbedding import FlagReranker
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
t0 = time.time()
RER = np.zeros((n, 50), dtype=np.float32)
TOP50 = np.zeros((n, 50), dtype=np.int64)
for i in range(n):
    top = np.argsort(-FUSED[i])[:50]
    TOP50[i] = top
    pairs = [[Q[i], TEXTS[j]] for j in top]
    sc = rer.compute_score(pairs, batch_size=50)
    RER[i] = np.asarray(sc, dtype=np.float32)
    if i % 200 == 0:
        print("rerank", i, "%.0fs" % (time.time() - t0), flush=True)
print("rerank all done %.0fs" % (time.time() - t0), flush=True)

def fused_rerank_score(i):
    s = FUSED[i].copy()
    top = TOP50[i]
    rr = RER[i]
    # 精排分z归一后整体替换top50内部顺序: top50按精排排, 其余按融合排
    order_inside = np.argsort(-rr)
    newvals = np.linspace(50, 1, 50)  # 50..1 的名次分
    s[top[order_inside]] = newvals
    rest = np.argsort(-s)
    return s

rk_rr = ev_ranks(fused_rerank_score)
m1 = metrics(rk_rr)
print("融合+精排: 错题进25 %d 对题进5 %d 均名次%.0f" % m1, flush=True)

# 融合变体: 精排当第四路通道
def fused_rer_channel(i, c=1.0):
    base = FUSED[i]
    zr = np.zeros_like(base)
    zr[TOP50[i]] = (RER[i] - RER[i].mean()) / (RER[i].std() + 1e-9) / 8
    return base + c * zr
for c in (1.0, 2.0):
    m2 = metrics(ev_ranks(lambda i, cc=c: fused_rer_channel(i, cc)))
    print("融合+精排第四路 c=%.1f: 错题进25 %d 对题进5 %d 均名次%.0f" % ((c,) + m2), flush=True)

# ---- 占座者位移(融合后仍错题的题) ----
COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
def cls(t):
    if t.rstrip().endswith("?"):
        return "问句"
    if COUR.search(t):
        return "寒暄"
    return "陈述"
ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])
fused_miss = [k for k in range(len(targets)) if rk_fused[k] > 25]
from collections import Counter
before = Counter()
after = Counter()
disp = []
for k in fused_miss:
    i = targets[k][0]
    top5 = TOP50[i][:5]
    for j in top5:
        before[cls(TEXTS[j])] += 1
    s = fused_rerank_score(i)
    new_top5 = np.argsort(-s)[:5]
    for j in new_top5:
        after[cls(TEXTS[j])] += 1
    # 旧占座者位移
    pos_new = {x: p for p, x in enumerate(np.argsort(-s))}
    disp += [pos_new[j] + 1 - (r0 + 1) for r0, j in enumerate(top5)]
print("融合仍错%d题: 精排前top5构成%s" % (len(fused_miss), dict(before)), flush=True)
print("               精排后top5构成%s 占座者位移中位%+.0f名" % (dict(after), float(np.median(disp))), flush=True)

# ---- 拆半 ----
def half_of(i):
    sid = rows[i].get("sample_id") or ""
    try:
        return int(re.sub(r"\D", "", sid)) % 2
    except Exception:
        return 0
for h in (0, 1):
    sel = [k for k in range(len(targets)) if half_of(targets[k][0]) == h]
    o = ok[sel]
    a = ((rk_fused[sel][~o] <= 25).sum(), (rk_fused[sel][o] <= 5).sum())
    b = ((rk_rr[sel][~o] <= 25).sum(), (rk_rr[sel][o] <= 5).sum())
    print("半区%d: 融合%d/%d → 精排%d/%d (错题进25/对题进5)" % (h, a[0], a[1], b[0], b[1]), flush=True)

# ---- 退化例(精排把证据排名弄差的题, 找原因) ----
print("=== 精排致退化例(融合≤25被挤出) ===", flush=True)
shown = 0
for k in range(len(targets)):
    if rk_fused[k] <= 25 and rk_rr[k] > 25 and shown < 5:
        i = targets[k][0]
        print("Q:", Q[i][:80], flush=True)
        print("  gold:", str(rows[i]["answer"])[:60], "| 融合名次%d→精排名次%d" % (rk_fused[k], rk_rr[k]), flush=True)
        j = [h for h in targets[k][1] if h in set(TOP50[i].tolist())]
        if j:
            print("  证据句:", TEXTS[j[0]][:100], "| 精排分%.2f" % RER[i][list(TOP50[i]).index(j[0])], flush=True)
        top1 = TOP50[i][np.argmax(RER[i])]
        print("  精排第1名(精排分%.2f):" % float(np.max(RER[i])), TEXTS[top1][:100], flush=True)
        shown += 1
if shown == 0:
    print("(无融合≤25被精排挤出的题)", flush=True)

g1 = (m1[0] >= 420 and m1[1] >= 615)
print("G1_%s miss25=%d ok5=%d" % ("PASS" if g1 else "FAIL", m1[0], m1[1]), flush=True)
np.savez_compressed(HERE + "/rerank_stage1.npz", TOP50=TOP50, RER=RER)
print("SAVED rerank_stage1.npz", flush=True)
