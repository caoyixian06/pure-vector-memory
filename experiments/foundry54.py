# -*- coding: utf-8 -*-
"""foundry54.py — 动态对比(pairwise差分)仲裁: 分数由竞争者动态决定
对头部10名两两差分: feat_diff(c,d) = 特征(c) - 特征(d)
组合分(c) = Σ_d w·sign(差分) 的logistic胜率 → 重排头部
特征: 12条普适信号(F52) + 团连接度
对照: GBDT纯分
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry54_results.txt"
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

NF = 14
def dyn_feats(qa, qi, c, peers):
    """动态特征: c的12普适信号 + 与peers的差分摘要"""
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
    # 团连接度: 与peers的双空间均值相似
    conn = np.mean([0.5 * float(D[c] @ D[p]) + 0.5 * float(QW[c] @ QW[p]) for p in peers]) if peers else 0.0
    base = [jac, qcov, vqc, dqc, avg, g1, g2, conn]
    # 差分摘要: 与每个peer的g1/vqc/avg差分的均值
    dg1 = dv = da = 0.0
    if peers:
        for p in peers:
            if p == c:
                continue
            pstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[p].lower()))
            pvqc = float(Q256[qi] @ QW[p])
            pdqc = float(D[p] @ X[qi])
            dg1 += g1 - len(qstems & pstems) / max(1, len(qstems))
            dv += vqc - pvqc
            da += avg - (pvqc + pdqc) / 2
        k = max(1, len(peers))
        dg1 /= k; dv /= k; da /= k
    return base + [dg1, dv, da]

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
from sklearn.metrics import roc_auc_score
K = (5, 15, 30)
res = {"gbdt": [0]*3, "gbdt+动态对比": [0]*3}
n = 0
auc_pairs = []
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
    # 训练动态对比器: 同题头部10名内 两两差分, 学"谁该在前"
    pairX, pairY = [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOLA[k_i]
        qi = IDX[qa]
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)][:10]
        head = o
        peers_all = head
        feats_of = {c: dyn_feats(qa, qi, c, head) for c in head}
        for a in head:
            for b in head:
                if a == b:
                    continue
                la, lb = (1 if a in G else 0), (1 if b in G else 0)
                if la == lb:
                    continue
                fa, fb = feats_of[a], feats_of[b]
                pairX.append([x - y for x, y in zip(fa, fb)])
                pairY.append(1 if la > lb else 0)
    PX = np.array(pairX, dtype=np.float32)
    PY = np.array(pairY, dtype=np.int8)
    if len(PY) < 50 or PY.min() == PY.max():
        continue
    lr = LogisticRegression(max_iter=3000, C=1.0)
    lr.fit(PX, PY)
    # 测试折应用
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        pool = POOLA[k_i]
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)]
        for k_j, k in enumerate(K):
            if all(r in set(o[:k]) for r in G):
                res["gbdt"][k_j] += 1
        head = o[:10]
        feats_of = {c: dyn_feats(qa, qi, c, head) for c in head}
        # listwise: 每个候选的动态分 = 对其余9个的胜率总和
        scores = {}
        for a in head:
            win = 0.0
            for b in head:
                if a == b:
                    continue
                diff = [x - y for x, y in zip(feats_of[a], feats_of[b])]
                p = float(lr.predict_proba([diff])[0][1])
                win += p
            scores[a] = win
        o_new = sorted(head, key=scores.get, reverse=True) + [c for c in o if c not in set(head)]
        rkn = {c: r + 1 for r, c in enumerate(o_new)}
        for k_j, k in enumerate(K):
            if all(r in set(o_new[:k]) for r in G):
                res["gbdt+动态对比"][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 动态对比(pairwise listwise)计分板 (n=%d) =====" % n)
for nm in ("gbdt", "gbdt+动态对比"):
    a, b, c = res[nm]
    P("%-14s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n)))
P("FOUNDRY54_DONE %.0fs" % (time.time() - t0))
LOG.close()
