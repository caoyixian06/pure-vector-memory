# -*- coding: utf-8 -*-
"""foundry4.py — 公式存在性证明器(零API)
①逐题可解性: 5维系数空间(x/cent5/u2/hemi/e1)里每题是否存在系数使金证据argmax第1
②系数团覆盖: 解系数k-means聚类, k个团覆盖多少可解题
③GBDT天花板: 通道特征柔性学习者 vs 线性冠军(held-out同池对比)
预注册: 可解率>=70% 且 (天花板-线性)>=5pp → 存在,继续找; 否则 → 表征墙,停止
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry5_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2)

t0 = time.time()
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
FINAL, IDS = z_ck["FINAL"], [str(x) for x in z_ck["IDS"]]
nQ = len(IDS)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
NR = D.shape[0]

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

if os.path.exists(HERE + "/xz_cache.npz"):
    _c = np.load(HERE + "/xz_cache.npz")
    X, Z = _c["X"], _c["Z"]
else:
    X = emb([Q[i]["question"] for i in IDS])
    Z = emb(["; ".join(str(x) for x in (Q[i].get("answer") or [])) for i in IDS])
    np.savez(HERE + "/xz_cache.npz", X=X, Z=Z)
P("embedded %d %.0fs" % (nQ, time.time() - t0))

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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
KINDARR = [1 if (json.loads(l).get("kind") or "summary") == "raw" else 0
           for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]

GOLD = np.full(nQ, -1, dtype=np.int64)
for qi, qa in enumerate(IDS):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for j, nj in enumerate(RAWN):
        if any(k in nj for k in keys):
            GOLD[qi] = j
            break
OK = GOLD >= 0
P("evidence locatable %d/%d" % (int(OK.sum()), nQ))

# 排名表(RANKS[qi] = 该题ORDER下每个记录的名次)
RANKS = np.zeros((nQ, NR), dtype=np.int32)
for qi in range(nQ):
    RANKS[qi, np.argsort(-FINAL[qi])] = np.arange(NR)

# 轴
qc_all = l2n(X.mean(0, keepdims=True))[0]
stmt = l2n(D.mean(0, keepdims=True))[0]
U2 = l2n((stmt - qc_all)[None])[0]
half = np.array([int(hashlib.md5(qa.encode()).hexdigest(), 16) % 2 for qa in IDS])
tr = half == 0
te = ~tr
HEMI = l2n((l2n(Z[tr].mean(0, keepdims=True))[0] - l2n(X[tr].mean(0, keepdims=True))[0])[None])[0]
E1 = D[np.array([int(np.argmax(FINAL[qi])) for qi in range(nQ)])]
C0 = l2n(X @ D.T)   # 每题对全库cos列缓存 (nQ, NR)
CENT5 = np.stack([l2n(D[np.argsort(-FINAL[qi])[:5]].mean(0, keepdims=True))[0] for qi in range(nQ)])
P("axes ready %.0fs" % (time.time() - t0))

# ===== ①逐题可解性 =====
VALS = [0.0, 0.25, 0.5, 1.0, 2.0]
C = np.array([[a, b, cu, ch, ce] for a in VALS for b in VALS for cu in VALS
              for ch in VALS for ce in VALS], dtype=np.float32).T   # (5,3125)
P("grid coefs: %d" % (C.shape[1],))
POOL = 200
solvable = np.zeros(nQ, dtype=bool)
sol_coef = np.zeros((nQ, 5), dtype=np.float32)
gold_rank_in_pool = np.full(nQ, -1, dtype=np.int64)
colU2 = D @ U2
colHEMI = D @ HEMI
for qi in range(nQ):
    if not OK[qi]:
        continue
    c0 = C0[qi]
    pool = np.argsort(-c0)[:POOL]
    if GOLD[qi] not in pool:
        pool = np.concatenate([pool, [GOLD[qi]]])
    W = np.stack([c0[pool], (D[pool] @ CENT5[qi]), colU2[pool], colHEMI[pool],
                  (D[pool] @ E1[qi])], axis=1)          # (pool,5)
    gpos = int(np.where(pool == GOLD[qi])[0][0])
    gold_rank_in_pool[qi] = gpos
    scores = W @ C                                        # (pool,3125)
    am = np.argmax(scores, axis=0)
    hit = am == gpos
    solvable[qi] = hit.any()
    if hit.any():
        # 取裕量最大的解系数
        golds = scores[gpos]
        others = scores.max(axis=0)
        margin = golds - others
        margin[~hit] = -1e9
        sol_coef[qi] = C[:, int(np.argmax(margin))]
    if qi % 300 == 0:
        P("  sol scan %d %.0fs" % (qi, time.time() - t0))

P("\n===== ①逐题可解性 =====")
P("全部题: %d/%d = %.1f%%" % (solvable.sum(), OK.sum(), 100.0 * solvable.sum() / max(1, OK.sum())))
errset = set()
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m and m.group(2) == "0":
        errset.add(m.group(1))
errmask = np.array([qa in errset for qa in IDS])
em = errmask & OK
P("错题:   %d/%d = %.1f%%" % (solvable[em].sum(), em.sum(), 100.0 * solvable[em].sum() / max(1, em.sum())))
cm = (~errmask) & OK
P("对题:   %d/%d = %.1f%%" % (solvable[cm].sum(), cm.sum(), 100.0 * solvable[cm].sum() / max(1, cm.sum())))
for cat in sorted(set(Q[i].get("category") or "?" for i in IDS)):
    cmask = np.array([Q[i].get("category") == cat for i in IDS]) & OK
    P("  cat%s: %d/%d = %.1f%%" % (cat, solvable[cmask].sum(), cmask.sum(),
                                   100.0 * solvable[cmask].sum() / max(1, cmask.sum())))

# ===== ②系数团覆盖 =====
P("\n===== ②系数团覆盖 =====")
from sklearn.cluster import KMeans
sidx = np.where(solvable)[0]
SC = l2n(sol_coef[sidx])
for k in (1, 2, 4, 6, 8, 12):
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(SC)
    centers = l2n(km.cluster_centers_.T)          # (5,k)
    cov = 0
    for qi in sidx:
        c0 = C0[qi]
        pool = np.argsort(-c0)[:POOL]
        if GOLD[qi] not in pool:
            pool = np.concatenate([pool, [GOLD[qi]]])
        W = np.stack([c0[pool], (D[pool] @ CENT5[qi]), colU2[pool], colHEMI[pool],
                      (D[pool] @ E1[qi])], axis=1)
        am = np.argmax(W @ centers, axis=0)
        gpos = int(np.where(pool == GOLD[qi])[0][0])
        if (am == gpos).any():
            cov += 1
    P("k=%2d 团覆盖: %d/%d = %.1f%%" % (k, cov, len(sidx), 100.0 * cov / len(sidx)))

# ===== ③回到向量本身: 三模型同池对决 =====
P("== ③三模型对决: 旧8 / 纯原始2048 / 合并 ==")
from sklearn.ensemble import HistGradientBoostingClassifier
gsplit = np.array([int(hashlib.md5((qa + "gbdt").encode()).hexdigest(), 16) % 10 < 7 for qa in IDS])
QTOK = [toks(Q[qa]["question"]) for qa in IDS]
rows_old, rows_raw, rows_y, rows_q = [], [], [], []
for qi in range(nQ):
    if not OK[qi]:
        continue
    c0 = C0[qi]
    pool = np.argsort(-c0)[:50]
    if GOLD[qi] not in pool:
        pool = np.concatenate([pool, [GOLD[qi]]])
    W = np.stack([c0[pool], (D[pool] @ CENT5[qi]), colU2[pool], colHEMI[pool],
                  (D[pool] @ E1[qi])], axis=1)
    qr = RANKS[qi, pool]
    xi = X[qi]
    for rr, j in enumerate(pool):
        ov = len(QTOK[qi] & toks(RAW[j])) / max(1, len(toks(RAW[j])))
        rj = D[j]
        rows_old.append([W[rr, 0], W[rr, 1], W[rr, 2], W[rr, 3], W[rr, 4],
                         float(qr[rr]), ov, float(KINDARR[j])])
        prod = xi * rj
        dif = np.abs(xi - rj)
        rows_raw.append(np.concatenate([prod, dif]))
        rows_y.append(1 if j == GOLD[qi] else 0)
        rows_q.append(qi)
    if qi % 300 == 0:
        P("  feat %d %.0fs" % (qi, time.time() - t0))
FOLD = np.array(rows_old, dtype=np.float32)
FRAW = np.array(rows_raw, dtype=np.float32)
FY = np.array(rows_y, dtype=np.int8)
FQ = np.array(rows_q)
P("rows=%d raw_dims=%d %.0fs" % (len(FY), FRAW.shape[1], time.time() - t0))
gtr = np.isin(FQ, np.where(gsplit)[0])
gte = np.isin(FQ, np.where(~gsplit)[0])
LIN = np.zeros(8, dtype=np.float32)
LIN[:5] = [0.25, 1.0, 0.0, 0.5, 0.0]
def pool_eval(score_by_q):
    h1 = h5 = n = 0
    for qi in np.unique(FQ[gte]):
        m = FQ == qi
        Yq, Sq = FY[m], score_by_q[m]
        if Yq.sum() == 0:
            continue
        g = int(np.argmax(Yq))
        n += 1
        r = int((Sq >= Sq[g]).sum())
        h1 += (r == 1)
        h5 += (r <= 5)
    return 100.0 * h1 / max(1, n), 100.0 * h5 / max(1, n), n
# 线性冠军
sc_lin = np.full(len(FY), -1e9)
sl = FOLD @ LIN
sc_lin[gte] = sl[gte]
a1, a5, nte = pool_eval(sc_lin)
P("线性冠军:   hit@1=%.1f%% hit@5=%.1f%% (n=%d)" % (a1, a5, nte))
for nm, FF in (("A_旧8特征", FOLD), ("B_纯原始2048", FRAW), ("C_合并", np.hstack([FOLD, FRAW]))):
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(FF[gtr], FY[gtr])
    sg = np.full(len(FY), -1e9)
    sg[gte] = clf.predict_proba(FF[gte])[:, 1]
    b1, b5, _ = pool_eval(sg)
    P("%s: hit@1=%.1f%% hit@5=%.1f%%" % (nm, b1, b5))
P("FOUNDRY5_DONE %.0fs" % (time.time() - t0))
LOG.close()
