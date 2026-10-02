# -*- coding: utf-8 -*-
"""foundry60.py — 题内优势维自适应放大算法(用户设计)
每题: 找出该题头部内"真金候选"占优的维度集(题内自适应) — 但无金标签时如何找?
用户洞察: 真金占优维是分布性的(20/256)。新算法:
  SCORE(c) = Σ_d w_d · (x_r[d]·r_c[d])
  其中 w_d = 该维度在"题内候选集"上的判别度权重 — 用"区分度高的维度"自加权:
  维度方差大 + 与GBDT top1的维度模式一致 → 权重高
三个变体:
  V1 方差加权: w_d ∝ 该维度在题内候选的std(方差大的维度携带更多信息)
  V2 锚对齐: w_d ∝ |锚(GBDT top1)的维度值|(锚是金概率最高, 全库对齐它)
  V3 互积放大: w_d ∝ 候选间该维度的平均互相关(共识维加权)
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry60_results.txt"
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
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

def g1_of(qa, c):
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
    return len(qstems & rstems) / max(1, len(qstems))

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
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
CONV_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(CONV_of.values()))
K = (5, 15, 30)
res = {"gbdt": [0]*3, "V1方差": [0]*3, "V2锚对齐": [0]*3, "V3互积": [0]*3}
n = 0
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
        if not G:
            continue
        n += 1
        qi = IDX[qa]
        pool = POOLA[k_i]
        pos_of = {c: i for i, c in enumerate(pool)}
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)]
        for k_j, kk in enumerate(K):
            if all(r in set(o[:kk]) for r in G):
                res["gbdt"][k_j] += 1
        # 题内头部10
        head = o[:10]
        qv = Q256[k_i]
        PRODM = np.array([qv * QW[c] for c in head])   # (10, 256) 交互积矩阵
        # V1 方差加权: 题内候选交互积的std
        w1 = PRODM.std(axis=0)
        w1 = w1 / (w1.sum() + 1e-9)
        # V2 锚对齐: GBDT top1的积模式
        w2 = np.abs(PRODM[0])
        w2 = w2 / (w2.sum() + 1e-9)
        # V3 互积共识: 维度在候选间的均值积(共识方向)
        w3 = np.abs(PRODM.mean(axis=0))
        w3 = w3 / (w3.sum() + 1e-9)
        gb_norm = np.array([s[pos_of[c]] for c in head])
        gb_norm = (gb_norm - gb_norm.min()) / (gb_norm.max() - gb_norm.min() + 1e-9)
        for nm, w in (("V1方差", w1), ("V2锚对齐", w2), ("V3互积", w3)):
            boost = PRODM @ (w * 256.0)   # 加权交互积和(放大优势维)
            boost = (boost - boost.min()) / (boost.max() - boost.min() + 1e-9)
            mix = 0.6 * gb_norm + 0.4 * boost
            o_v = [head[i] for i in np.argsort(-mix)] + [c for c in o if c not in set(head)]
            for k_j, kk in enumerate(K):
                if all(r in set(o_v[:kk]) for r in G):
                    res[nm][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 题内优势维放大算法 (n=%d) =====" % n)
for nm in ("gbdt", "V1方差", "V2锚对齐", "V3互积"):
    a, b, c = res[nm]
    P("%-10s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%%" % (nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n)))
P("F60_DONE %.0fs" % (time.time() - t0))
LOG.close()
