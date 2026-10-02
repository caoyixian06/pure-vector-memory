# -*- coding: utf-8 -*-
"""foundry56.py — top5全真金的天花板测量
对每题池650: GBDT分 + 12信号组合分(LOCO logistic)
测:
  A. all-gold@5: GBDT序 / 组合序 / 混合序
  B. 分离性oracle: max(非金分) < min(真金分) 的题占比 — 信号集的理论天花板
  C. 不可分题的假金来源分析(头部假金是什么)
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry56_results.txt"
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
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w

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
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

def sig12(qa, qi, c):
    """12条普适信号(F52验证)"""
    q = Q[qa]
    ql = q["question"].lower()
    qwords = [w for w in re.findall(r"[a-z']+", ql) if w not in QSTOP and len(w) > 2]
    qstems = set(stem(w) for w in qwords)
    ct = ctoks(c)
    qtok = toks(q["question"])
    inter = qtok & ct
    jac = len(inter) / max(1, len(qtok | ct))
    qcov = len(inter) / max(1, len(qtok))
    vqc = float(Q256[qi] @ QW[c])
    dqc = float(D[c] @ X[qi])
    avg = (vqc + dqc) / 2
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
    g1 = len(qstems & rstems) / max(1, len(qstems))
    rcws = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
    g2 = sum(1 for w in rcws if stem(w) not in qstems) / max(1, len(rcws)) if rcws else 0.0
    dif256 = np.abs(Q256[qi] - QW[c])
    dif1024 = np.abs(X[qi] - D[c])
    m256, s256 = float(dif256.mean()), float(dif256.std())
    m1024, s1024 = float(dif1024.mean()), float(dif1024.std())
    p90256 = float(np.quantile(dif256, 0.9))
    p901024 = float(np.quantile(dif1024, 0.9))
    neg = float((X[qi] * D[c] < 0).mean())
    return [g1, avg, vqc, m256, p90256, s256, dqc, s1024, m1024, p901024, neg, g2]

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
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
res = {"gbdt": 0, "comb": 0, "mix": 0, "oracle_gbdt": 0, "oracle_comb": 0}
n = 0
infeasible5 = 0
overlap_cases = []
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
    # 组合器: 12信号logistic, LOCO训练(池内金vs噪)
    trX2, trY2 = [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOLA[k_i]
        G = GSETS[k_i]
        golds = [c for c in pool if c in G][:6]
        noises = [c for c in pool if c not in G][:20]
        for c in golds:
            trX2.append(sig12(qa, qi, c)); trY2.append(1)
        for c in noises:
            trX2.append(sig12(qa, qi, c)); trY2.append(0)
    sc2 = StandardScaler().fit(np.array(trX2, dtype=np.float64))
    lr2 = LogisticRegression(max_iter=3000).fit(sc2.transform(np.array(trX2, dtype=np.float64)), np.array(trY2))
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        if len(G) > 5:
            infeasible5 += 1
            continue
        qi = IDX[qa]
        pool = POOLA[k_i]
        pos_of = {c: i for i, c in enumerate(pool)}
        s = clf.decision_function(FEATS[k_i])
        csig = np.array([sig12(qa, qi, c) for c in pool], dtype=np.float64)
        comb = lr2.decision_function(sc2.transform(csig))
        # A. all-gold@5
        og = [pool[i] for i in np.argsort(-s)]
        rk_g = {c: r + 1 for r, c in enumerate(og)}
        oc = [pool[i] for i in np.argsort(-comb)]
        rk_c = {c: r + 1 for r, c in enumerate(oc)}
        # 混合: z(gbdt)+z(comb)
        zs = (s - s.mean()) / (s.std() + 1e-9)
        zc = (comb - comb.mean()) / (comb.std() + 1e-9)
        om = [pool[i] for i in np.argsort(-(zs + zc))]
        rk_m = {c: r + 1 for r, c in enumerate(om)}
        if all(rk_g.get(g, 999) <= 5 for g in G):
            res["gbdt"] += 1
        if all(rk_c.get(g, 999) <= 5 for g in G):
            res["comb"] += 1
        if all(rk_m.get(g, 999) <= 5 for g in G):
            res["mix"] += 1
        # B. 分离性oracle: 池内非金最高分 < 真金最低分
        gold_s = [s[pos_of[g]] for g in G if g in pos_of]
        noise_s = [s[i] for i, c in enumerate(pool) if c not in G]
        if noise_s and gold_s and max(noise_s) < min(gold_s):
            res["oracle_gbdt"] += 1
        gold_c = [comb[pos_of[g]] for g in G if g in pos_of]
        noise_c = [comb[i] for i, c in enumerate(pool) if c not in G]
        if noise_c and gold_c and max(noise_c) < min(gold_c):
            res["oracle_comb"] += 1
        # C. 不可分案例: 记录挤掉真金的假金是什么
        if gold_c and noise_c and not (max(noise_c) < min(gold_c)) and len(overlap_cases) < 5:
            # 找挤位假金: 分数高于某真金的非金记录
            for g in G:
                higher = [c for i, c in enumerate(pool) if c not in G and comb[i] > comb[pos_of[g]]]
                if higher:
                    overlap_cases.append((qa, RAW[g][:60], RAW[higher[0]][:60]))
                    break
    P("  fold %s done" % hold)

P("\n===== top5全真金天花板测量 (n=%d) =====" % n)
P("片数>5(数学不可行): %d题 (%.1f%%)" % (infeasible5, 100.0*infeasible5/max(1,n)))
P("all-gold@5: GBDT=%.1f%% 组合器=%.1f%% 混合=%.1f%%" % (
    100.0*res["gbdt"]/max(1,n), 100.0*res["comb"]/max(1,n), 100.0*res["mix"]/max(1,n)))
P("分离性oracle(信号集天花板): GBDT分=%.1f%% 组合分=%.1f%%" % (
    100.0*res["oracle_gbdt"]/max(1,n), 100.0*res["oracle_comb"]/max(1,n)))
P("\n不可分案例(挤位假金 vs 被挤真金):")
for qa, gold_txt, fake_txt in overlap_cases:
    P("  [%s]\n    真金: %s\n    假金: %s" % (qa, gold_txt, fake_txt))
P("FOUNDRY56_DONE %.0fs" % (time.time() - t0))
LOG.close()
