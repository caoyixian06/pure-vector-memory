# -*- coding: utf-8 -*-
"""formula_foundry2.py — 七族公式一次穷举(零API, 拆半铁律, 向量化评估)
F1 组合 a·x+b·ê+c·v (ê∈{e1,cent3,cent5}, v∈{u2,Δ,hemi,narr,0})
F2 窗口 a·x+b·cent(topk)
F3 对比 a·x+b·e1+c·(e1-cent(k))
F4 交互 a·x+b·e+c·norm(x⊙e)
F5 跨度投影 P_span(topk)(a·x+b·hemi)
F6 分题型系数(每类在train各自选优)
F7 多答案z取max-cos口径(仅对冠军复测)
判据: cos + 甄别20/甄别100; 系数train半区选, held-out一次性。
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry2_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
FINAL, IDS = z_ck["FINAL"], [str(x) for x in z_ck["IDS"]]
nQ = len(IDS)

D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

X = emb([Q[i]["question"] for i in IDS])
ZS = [emb([str(x) for x in (Q[i].get("answer") or [""])]) for i in IDS]
Z = l2n(np.stack([z.mean(0) for z in ZS]))
print("embedded n=%d %.0fs" % (nQ, time.time() - t0), flush=True)
P("embedded %d (%.0fs)" % (nQ, time.time() - t0))

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
RAWN = [norm(rec_raw(m)) for m in MID]
NR = D.shape[0]

E1 = np.zeros((nQ, D.shape[1]), dtype=np.float32)
EVLBL = np.zeros((nQ, NR), dtype=np.float32)
ORDER = []
for qi, qa in enumerate(IDS):
    q = Q[qa]
    order = list(map(int, np.argsort(-FINAL[qi])))
    ORDER.append(order)
    E1[qi] = D[order[0]]
    keys = [norm(em.get("text") or "")[:60] for em in (q.get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for j in order:
        if any(k in RAWN[j] for k in keys):
            EVLBL[qi, j] = 1.0
CATS = sorted(set(Q[i].get("category") or "?" for i in IDS))

half = np.array([int(hashlib.md5(qa.encode()).hexdigest(), 16) % 2 for qa in IDS])
tr = half == 0
te = ~tr
P("train=%d held-out=%d cats=%s" % (tr.sum(), te.sum(), CATS))

qc_all = l2n(X.mean(0, keepdims=True))[0]
stmt = l2n(D.mean(0, keepdims=True))[0]
U2 = stmt - qc_all
DELTA = np.zeros_like(U2)
cnt = 0
for qi in range(nQ):
    if not tr[qi]:
        continue
    top = ORDER[qi][:50]
    lbl = EVLBL[qi, top] > 0
    if 0 < lbl.sum() < len(top):
        DELTA += D[top][lbl].mean(0) - D[top][~lbl].mean(0)
        cnt += 1
DELTA = l2n((DELTA / max(1, cnt))[None])[0]
HEMI = l2n((l2n(Z[tr].mean(0, keepdims=True))[0] - l2n(X[tr].mean(0, keepdims=True))[0])[None])[0]
ANX = ["narrative", "greeting", "chat", "small talk", "wow haha lol", "story memory life event"]
ANXV = emb(ANX)
NARR = l2n((ANXV[:4].mean(0) - ANXV[4:].mean(0))[None])[0]
P("axes ready %.0fs" % (time.time() - t0))

CENT = {k: np.stack([l2n(D[ORDER[qi][:k]].mean(0, keepdims=True))[0] for qi in range(nQ)])
        for k in (1, 3, 5, 9, 15, 25)}

def make_pools(mask, npool, seed):
    idxs = np.where(mask)[0]
    rng = np.random.RandomState(seed)
    pools = np.stack([np.concatenate([[qi], rng.choice(idxs, npool - 1, replace=False)]) for qi in idxs])
    return idxs, pools
idxs_tr, pools_tr = make_pools(tr, 20, 42)
idxs_te, pools_te = make_pools(te, 20, 43)
idxs100_tr, pools100_tr = make_pools(tr, 100, 44)
idxs100_te, pools100_te = make_pools(te, 100, 45)

def eval_vec(PZ, idxs, pools, ZSmax=False):
    pools = np.asarray(pools, dtype=np.int64)
    sub = PZ[pools]
    tgt = Z[pools[:, 0]][:, None, :]
    S = (sub * tgt).sum(-1)
    r20 = float((S >= S[:, :1]).sum(1).mean())
    return float((l2n(PZ[idxs]) * Z[idxs]).sum(1).mean()), r20

def score(PZ):
    ctr, rtr = eval_vec(PZ, np.where(tr)[0], pools_tr)
    return ctr, rtr

RES = []
def register(name, PZ, note=""):
    ctr, rtr = score(PZ)
    RES.append((ctr, name, PZ, note, rtr))
    return ctr

# ===== 基线 =====
P("\n===== 基线 =====")
register("B0_z=x", X.copy())
register("B1_z=e1", E1.copy())
register("B2_z=x+u2", l2n(X + U2))
register("B3_z=均值", np.tile(l2n(Z[tr].mean(0, keepdims=True)), (nQ, 1)))
P("baselines done %.0fs" % (time.time() - t0))

# ===== F1 =====
P("\n===== F1 组合族 =====")
AXES = {"0": None, "u2": U2, "delta": DELTA, "hemi": HEMI, "narr": NARR}
ES = {"e1": E1, "cent3": CENT[3], "cent5": CENT[5]}
R = [0, 0.25, 0.5, 1, 2, 4]
f1best = None
for en, E in ES.items():
    for vn, v in AXES.items():
        for a in R:
            for b in R:
                for c in ([0] if v is None else [0, 0.1, 0.25, 0.5, 1, 2, -0.5, -1]):
                    PZ = a * X + b * E + (c * v if v is not None else 0)
                    ctr, _ = eval_vec(l2n(PZ), np.where(tr)[0], pools_tr)
                    if f1best is None or ctr > f1best[0]:
                        f1best = (ctr, en, vn, a, b, c)
P("F1 train best: %s" % (f1best,))
_, en, vn, a, b, c = f1best
PZ = l2n(a * X + b * ES[en] + (c * AXES[vn] if AXES[vn] is not None else 0))
register("F1_best[%s,%s a=%s b=%s c=%s]" % (en, vn, a, b, c), PZ)

# ===== F2 =====
P("\n===== F2 窗口族 =====")
f2best = None
for k in (1, 3, 5, 9, 15, 25):
    for a in R:
        for b in R:
            if a == 0 and b == 0:
                continue
            ctr, _ = eval_vec(l2n(a * X + b * CENT[k]), np.where(tr)[0], pools_tr)
            if f2best is None or ctr > f2best[0]:
                f2best = (ctr, k, a, b)
P("F2 train best: %s" % (f2best,))
_, k2, a2, b2 = f2best
register("F2_best[k=%s a=%s b=%s]" % (k2, a2, b2), l2n(a2 * X + b2 * CENT[k2]))

# ===== F3 =====
P("\n===== F3 对比族 =====")
f3best = None
for k in (5, 15):
    DIFF = E1 - CENT[k]
    for a in R:
        for b in R:
            for c in (0, 0.25, 0.5, 1, 2, -0.5, -1):
                ctr, _ = eval_vec(l2n(a * X + b * E1 + c * DIFF), np.where(tr)[0], pools_tr)
                if f3best is None or ctr > f3best[0]:
                    f3best = (ctr, k, a, b, c)
P("F3 train best: %s" % (f3best,))
_, k3, a3, b3, c3 = f3best
register("F3_best[k=%s a=%s b=%s c=%s]" % (k3, a3, b3, c3), l2n(a3 * X + b3 * E1 + c3 * (E1 - CENT[k3])))

# ===== F4 =====
P("\n===== F4 交互族 =====")
f4best = None
XG = X * E1
XG = l2n(XG)
for a in R:
    for b in R:
        for c in (0, 0.1, 0.25, 0.5, 1):
            ctr, _ = eval_vec(l2n(a * X + b * E1 + c * XG), np.where(tr)[0], pools_tr)
            if f4best is None or ctr > f4best[0]:
                f4best = (ctr, a, b, c)
P("F4 train best: %s" % (f4best,))
_, a4, b4, c4 = f4best
register("F4_best[a=%s b=%s c=%s]" % (a4, b4, c4), l2n(a4 * X + b4 * E1 + c4 * XG))

# ===== F5 =====
P("\n===== F5 跨度投影族 =====")
f5best = None
for k in (5, 15):
    for a in (0.5, 1, 2):
        for b in (0, 0.25, 0.5, 1):
            base = a * X + b * HEMI
            PZ = np.zeros_like(X)
            for qi in range(nQ):
                M = D[ORDER[qi][:k]]
                G2 = M @ M.T + 1e-6 * np.eye(k)
                w = np.linalg.solve(G2, M @ base[qi])
                PZ[qi] = M.T @ w
            ctr, _ = eval_vec(l2n(PZ), np.where(tr)[0], pools_tr)
            if f5best is None or ctr > f5best[0]:
                f5best = (ctr, k, a, b)
P("F5 train best: %s" % (f5best,))
_, k5, a5, b5 = f5best
base = a5 * X + b5 * HEMI
PZ = np.zeros_like(X)
for qi in range(nQ):
    M = D[ORDER[qi][:k5]]
    G2 = M @ M.T + 1e-6 * np.eye(k5)
    w = np.linalg.solve(G2, M @ base[qi])
    PZ[qi] = M.T @ w
register("F5_best[k=%s a=%s b=%s]" % (k5, a5, b5), l2n(PZ))

# ===== F6 =====
P("\n===== F6 分题型族(每类train各自选优F1空间) =====")
PZ6 = np.zeros_like(X)
ok6 = 0
for cat in CATS:
    mask_cat = np.array([Q[i].get("category") == cat for i in IDS])
    sub_tr = np.where(tr & mask_cat)[0]
    if len(sub_tr) < 30:
        continue
    bestc = None
    for a in (0.25, 0.5, 1):
        for b in (0.25, 0.5, 1, 2):
            for c in (0, 0.25, 0.5, 1):
                PZ = l2n(a * X + b * E1 + c * HEMI)
                ctr, _ = eval_vec(PZ, sub_tr, make_pools(tr & mask_cat, 20, 46)[1])
                if bestc is None or ctr > bestc[0]:
                    bestc = (ctr, a, b, c)
    _, a6, b6, c6 = bestc
    idx_cat = np.where(mask_cat)[0]
    PZ6[idx_cat] = l2n(a6 * X + b6 * E1 + c6 * HEMI)[idx_cat]
    ok6 += 1
    P("  cat=%-14s n=%4d a=%s b=%s c=%s" % (cat, mask_cat.sum(), a6, b6, c6))
register("F6_per_cat(%d类)" % ok6, PZ6)

# ===== 终评(held-out一次) =====
P("\n===== 终评(train选优 → held-out一次) =====")
RES.sort(key=lambda x: -x[0])
idxs_te, pools_te20 = np.where(te), pools_te
for ctr_tr, name, PZ, note, rtr in RES[:8]:
    cho, r20ho = eval_vec(PZ, idxs_te, pools_te20)
    _, r100ho = eval_vec(PZ, idxs_te, pools100_te)
    P("%-38s cos tr=%.4f ho=%.4f | 甄别20ho=%.2f 100ho=%.2f" % (name, ctr_tr, cho, r20ho, r100ho))
cho_b2, r20b2 = eval_vec(l2n(X + U2), idxs_te, pools_te20)
_, r100b2 = eval_vec(l2n(X + U2), idxs_te, pools100_te)
P("%-38s cos tr=%.4f ho=%.4f | 甄别20ho=%.2f 100ho=%.2f" % ("[基线]B2_z=x+u2", 0.4133, cho_b2, r20b2, r100b2))
P("\nFOUNDRY2_DONE %.0fs" % (time.time() - t0))
LOG.close()
