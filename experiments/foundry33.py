# -*- coding: utf-8 -*-
"""foundry33.py — 独特相似性选座: 阈值准入+互选确认(0.864判别力落地)
seat score = γ·GBDT分 + λ·max(与已入座成员的组合相似度-θ, 0)
θ/λ/γ 网格, LOCO。靶: 多片题all-gold@5/8/15/30。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry33_results.txt"
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
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
CONV_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(CONV_of.values()))
TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]
P("loaded %.0fs" % (time.time() - t0))

# ===== 池 + 特征(带相似度矩阵缓存) =====
from sklearn.ensemble import HistGradientBoostingClassifier
POOLSZ = 650
SIMCACHE = {}
def sim_pair(a, b):
    key = (a, b) if a < b else (b, a)
    if key in SIMCACHE:
        return SIMCACHE[key]
    ta, tb = ctoks(a), ctoks(b)
    jac = len(ta & tb) / max(1, len(ta | tb))
    v = float(QW[a] @ QW[b])
    d = float(D[a] @ D[b])
    s = 0.4 * jac + 0.3 * (v - 0.5) + 0.3 * (d - 0.6)
    SIMCACHE[key] = s
    return s

FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    for i in list(np.argsort(-FINAL[qi]))[:50]:
        pool.add(max(0, i - 1))
        pool.add(min(NR - 1, i + 1))
    pool = sorted(pool)[:POOLSZ]
    POOLA[k_i] = pool
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    Fm = np.zeros((len(pool), 321), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = ctoks(c)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lr = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
        vqc = float(qv256 @ QW[c])
        dqc = float(D[c] @ qv1024)
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
    FEATS[k_i] = Fm
    POOLA[k_i] = pool
    if k_i % 200 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

K = (5, 8, 15, 30)
GRID = [(th, lam, gam) for th in (0.2, 0.35) for lam in (1.0, 2.0) for gam in (0.5, 1.0)]
V = {("GBDT纯分", 0): [0]*4}
V.update({("阈值准入θ=%.2f λ=%.1f γ=%.1f" % g, i): [0]*4 for g in GRID for i in [0]})
V.update({(nm, i): [0]*4 for nm in ("GBDT纯分",) for i in [0]})
n_multi = 0
best_key = None
for hold in FOLDS:
    trF, trY = [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOLA[k_i]
        G = GSETS[k_i]
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
        for rr in gold_rows:
            trF.append(FEATS[k_i][rr]); trY.append(1)
        for rr in noisepick:
            trF.append(FEATS[k_i][rr]); trY.append(0)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8))
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if len(G) < 2:
            continue
        n_multi += 1
        pool = POOLA[k_i]
        gb = clf.decision_function(FEATS[k_i])
        gb_order = [pool[i] for i in np.argsort(-gb)]
        gbnorm = gb / (np.abs(gb).max() + 1e-9)
        # GBDT纯分基线
        for k_j, k in enumerate(K):
            if all(r in set(gb_order[:k]) for r in G):
                V[("GBDT纯分", 0)][k_j] += 1
        # 阈值准入+互选确认
        for th, lam, gam in GRID:
            key = ("阈值准入θ=%.2f λ=%.1f γ=%.1f" % (th, lam, gam), 0)
            seats = [gb_order[0]]
            seatset = set(seats)
            cand = gb_order[1:]
            pos = {c: i for i, c in enumerate(gb_order)}
            while len(seats) < 30 and cand:
                best, bv = None, -1e18
                for c in cand:
                    sim_max = max((sim_pair(c, s2) for s2 in seats), default=-1)
                    bonus = lam * max(sim_max - th, 0.0)
                    v = gam * gbnorm[pos[c]] / 3.0 + bonus
                    if v > bv:
                        bv, best = v, c
                seats.append(best)
                seatset.add(best)
                cand.remove(best)
            for k_j, k in enumerate(K):
                if all(r in set(seats[:k]) for r in G):
                    V[key][k_j] += 1
P("\n===== 独特相似性选座 (多片题n=%d, LOCO) =====" % n_multi)
for (nm, _), v in V.items():
    P("%-28s " % nm + "  ".join("all@%d=%.1f%%" % (k, 100.0*v[i]/max(1,n_multi)) for i, k in enumerate(K)))
P("FOUNDRY33_DONE %.0fs" % (time.time() - t0))
LOG.close()
