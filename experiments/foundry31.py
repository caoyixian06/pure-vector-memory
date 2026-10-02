# -*- coding: utf-8 -*-
"""foundry31.py — 多片金证据选座: 锚点-兄弟扩展选座 vs 纯分数top-K
 seats: ①FINAL纯top-K(基线) ②GBDT纯top-K ③锚显著词兄弟扩展选座(GBDT锚+词扩展)
靶: 多片题all-gold@K(5/8/15/30), LOCO。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry31_results.txt"
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
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]
P("loaded+DF %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

POOLSZ = 650
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

from sklearn.ensemble import HistGradientBoostingClassifier
K = (5, 8, 15, 30)
V = {"FINAL纯分": [0]*4, "GBDT纯分": [0]*4, "锚兄弟选座": [0]*4, "混合选座": [0]*4}
n_multi = 0
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
        qi = IDX[qa]
        pool = POOLA[k_i]
        gb = clf.decision_function(FEATS[k_i])
        gb_order = [pool[i] for i in np.argsort(-gb)]
        f_order = list(np.argsort(-FINAL[qi]))
        qtok = toks(Q[qa]["question"])
        # 锚=GBDT第1名
        anchor = gb_order[0]
        sal = [w for w in RTOK[anchor] if w not in qtok and DF.get(w, 999) <= 50]
        salset = set(sal)
        # 锚兄弟扩展选座: 第一座=锚; 之后每座=池内(GBDT分 + 0.5·锚显著词覆盖)最高且未入座
        seats = [anchor]
        seatset = set(seats)
        pos_gb = {c: i for i, c in enumerate(gb_order)}
        cand = list(gb_order[1:60])
        cov = {c: (len(salset & RTOK[c]) / max(1, len(salset)) if salset else 0.0) for c in cand}
        while len(seats) < 30 and cand:
            best, bv = None, -1e18
            for c in cand:
                if c in seatset:
                    continue
                v = 0.12 * (60 - pos_gb.get(c, 60)) + 2.0 * cov[c]
                if v > bv:
                    bv, best = v, c
            seats.append(best)
            seatset.add(best)
            cand.remove(best)
        for k_j, k in enumerate(K):
            if all(r in set(gb_order[:k]) for r in G):
                V["GBDT纯分"][k_j] += 1
            if all(r in set(f_order[:k]) for r in G):
                V["FINAL纯分"][k_j] += 1
            if all(r in set(seats[:k]) for r in G):
                V["锚兄弟选座"][k_j] += 1
        # 混合: 前2座=GBDT前2, 其余=锚兄弟
        mix = list(gb_order[:2]) + [c for c in seats if c not in set(gb_order[:2])]
        for k_j, k in enumerate(K):
            if all(r in set(mix[:k]) for r in G):
                V["混合选座"][k_j] += 1
P("\n===== 多片题选座对比 (n=%d, LOCO) =====" % n_multi)
for nm in V:
    P("%-10s " % nm + "  ".join("all@%d=%.1f%%" % (k, 100.0*V[nm][i]/max(1,n_multi)) for i, k in enumerate(K)))
P("FOUNDRY31_DONE %.0fs" % (time.time() - t0))
LOG.close()
