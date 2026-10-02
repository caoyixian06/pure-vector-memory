# -*- coding: utf-8 -*-
"""foundry29.py — 两层叠加: R5规则底盘 + GBDT残差修正
最终分 = z(R5) + γ·GBDT(全特征, 训练目标=金标签残差)。γ∈{0.3,0.6,1.0}扫描。
LOCO协议。对照: R5单飞 / GBDT单飞 / FINAL。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry29_results.txt"
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
ISRAW = np.array([(json.loads(l).get("kind") or "summary") == "raw"
                  for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()])
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
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

# ===== 特征+R5分一次构建 =====
from sklearn.ensemble import HistGradientBoostingRegressor
POOLSZ = 400
FEATS = [None] * len(IDS)
R5S = [None] * len(IDS)
POOLA = [None] * len(IDS)
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    pool = set(np.argsort(-C0[qi])[:300]) | set(np.argsort(-FINAL[qi])[:250])
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
    r5 = np.zeros(len(pool), dtype=np.float32)
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
        r5[rr] = 0.5 * qcov + 0.2 * min(lr, 4) / 4 - 1.5 * qmark - iecho + 0.3 * vqc + 0.3 * dqc
    FEATS[k_i] = Fm
    R5S[k_i] = r5
    if k_i % 200 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

def zsd(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)

K = (5, 15, 30)
V = {"R5": [0]*3, "GBDT": [0]*3, "叠0.3": [0]*3, "叠0.6": [0]*3, "叠1.0": [0]*3, "FINAL": [0]*3}
n = 0
for hold in FOLDS:
    trX, trY, trR = [], [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        G = GSETS[k_i]
        pool = POOLA[k_i]
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
        for rr in gold_rows:
            trX.append(FEATS[k_i][rr]); trY.append(1.0); trR.append(R5S[k_i][rr])
        for rr in noisepick:
            trX.append(FEATS[k_i][rr]); trY.append(0.0); trR.append(R5S[k_i][rr])
    Xtr = np.array(trX, dtype=np.float32)
    Ytr = np.array(trY, dtype=np.float32)
    Rtr = zsd(np.array(trR, dtype=np.float64))
    resid = Ytr - 0.5 * Rtr
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    reg.fit(Xtr, resid)
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        r5z = zsd(R5S[k_i])
        gbz = zsd(reg.predict(FEATS[k_i]))
        finals = zsd(FINAL[IDX[qa]])
        variants = {"R5": r5z, "GBDT": gbz, "叠0.3": r5z + 0.3 * gbz,
                    "叠0.6": r5z + 0.6 * gbz, "叠1.0": r5z + 1.0 * gbz}
        pool = POOLA[k_i]
        for nm, sc in variants.items():
            o = [pool[i] for i in np.argsort(-sc)]
            rk = {c: r + 1 for r, c in enumerate(o)}
            rks = [rk.get(i, 10**9) for i in G]
            for k_j, k in enumerate(K):
                if max(rks) <= k:
                    V[nm][k_j] += 1
        o = list(np.argsort(-finals))
        rkf = {c: r + 1 for r, c in enumerate(o)}
        rks = [rkf.get(i, 10**9) for i in G]
        for k_j, k in enumerate(K):
            if max(rks) <= k:
                V["FINAL"][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 两层叠加计分板 (n=%d) =====" % n)
for nm in V:
    a, b, c = V[nm]
    P("%-8s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n)))
P("FOUNDRY29_DONE %.0fs" % (time.time() - t0))
LOG.close()
