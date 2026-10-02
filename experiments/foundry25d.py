# -*- coding: utf-8 -*-
"""foundry25.py — 锚点扩展+词干化桥接 接入LOCO重排器: 严格all-gold全测
池: C0-top200 ∪ 锚点扩展50; 特征: 12衍生+4形态+100量化+100差+2词干化 = 218维
协议: 会话留一(LOCO) 10折。对照: FINAL同题。
"""
import io, json, os, re, sys, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry25d_results.txt"
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
def stoks(s):
    return set(stem(w) for w in re.findall(r"[a-z]{3,}", str(s).lower()))

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
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]
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
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
CONV = [qa.split("#")[0] for qa in IDS]
CONVS = sorted(set(CONV))

DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
P("loaded+DF %d词 %.0fs" % (len(DF), time.time() - t0))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

Q256I = {qa: i for i, qa in enumerate(IDS)}
FORMD = [616, 538, 218, 408]
RNG = np.random.RandomState(7)
STAB = RNG.permutation(NR)[:100] if False else np.arange(100)

# ===== 逐题构建: 池(C0-200∪扩展50) + 特征(218) =====
FDATA = []
POOLALL = []
GALL = []
QTOK = [stoks(Q[qa]["question"]) for qa in IDS]
QLEN = [max(1, len(t)) for t in QTOK]
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = gold_set(qa)
    GALL.append(G)
    order = list(np.argsort(-C0[qi]))
    pool = order[:200]
    anchor = None
    for i in order:
        if ISRAW[i]:
            anchor = i
            break
    exp = []
    if anchor is not None:
        qtok = toks(Q[qa]["question"])
        ws = [w for w in re.findall(r"[a-z]{4,}", RAW[anchor].lower()) if w not in qtok]
        ws = sorted(set(ws), key=lambda w: -DF.get(norm(w), 0))[:12]
        salset = set(ws)
        if salset:
            sc = []
            inpool = set(pool)
            for i in range(NR):
                if ISRAW[i] and i not in inpool:
                    rn = RTOK[i]
                    cov = len(salset & rn) / len(salset)
                    if cov >= 0.5:
                        sc.append((cov, i))
            sc.sort(reverse=True)
            exp = [i for _, i in sc[:50]]
    cands = pool + [i for i in exp if i not in set(pool)]
    POOLALL.append(cands)
    qtokS = QTOK[k_i]
    qlen = QLEN[k_i]
    qv256 = Q256[Q256I[qa]]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    Fm = np.zeros((len(cands), 212), dtype=np.float32)
    XDq = X[qi][STAB]
    for rr, c in enumerate(cands):
        ct = stoks(RAW[c])
        inter = qtokS & ct
        jac = len(inter) / max(1, len(qtokS | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lenratio = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
        v_qc = float(qv256 @ QW[c])
        d_qc = float(D[c] @ qv1024)
        Fm[rr, :12] = [jac, qcov, ccov, qmark, iecho, lenratio, v_qc, v_qc - d_qc,
                       d_qc, d_qc - v_qc, d_qc - qcov, qcov - d_qc]
        Fm[rr, 12:112] = D[c][STAB]
        Fm[rr, 112:212] = np.abs(XDq - D[c][STAB])
    FDATA.append(Fm)
    if k_i % 200 == 0:
        P("  build %d %.0fs" % (k_i, time.time() - t0))
P("built %d题 (池均%.0f条) %.0fs" % (len(FDATA), np.mean([len(f) for f in FDATA]), time.time() - t0))

# ===== LOCO 10折 =====
from sklearn.ensemble import HistGradientBoostingClassifier
K = (5, 10, 30)
res = {"gbdt": [0]*3, "final": [0]*3}
n = 0
per5 = []
for hold in CONVS:
    trF, trY, gr = [], [], []
    for k_i, qa in enumerate(IDS):
        if CONV[k_i] == hold:
            continue
        G = GALL[k_i]
        glist = [rr for rr, c in enumerate(POOLALL[k_i]) if c in G and rr < 200][:8]
        noises = [rr for rr, c in enumerate(POOLALL[k_i]) if c not in G and rr < 200][:12]
        for rr in glist:
            trF.append(FDATA[k_i][rr]); trY.append(1)
        for rr in noises:
            trF.append(FDATA[k_i][rr]); trY.append(0)
    Ftr = np.array(trF, dtype=np.float32)
    Ytr = np.array(trY, dtype=np.int8)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(Ftr, Ytr)
    f5 = f10 = f30 = cnt = 0
    b5 = b10 = b30 = 0
    for k_i, qa in enumerate(IDS):
        if CONV[k_i] != hold or not GALL[k_i]:
            continue
        n += 1
        cnt += 1
        s200 = clf.decision_function(FDATA[k_i][:200])
        o200 = [POOLALL[k_i][i] for i in np.argsort(-s200)]
        head = o200[:5]
        exp_recs = POOLALL[k_i][200:]
        final_order = head + list(exp_recs) + [r for r in o200[5:] if r not in set(exp_recs)]
        G = GALL[k_i]
        o = final_order
        forder = list(np.argsort(-FINAL[IDX[qa]]))
        for k_j, k in enumerate(K):
            if all(r in o[:k] for r in G):
                if k_j == 0: f5 += 1
                if k_j == 1: f10 += 1
                if k_j == 2: f30 += 1
            if all(r in forder[:k] for r in G):
                if k_j == 0: b5 += 1
                if k_j == 1: b10 += 1
                if k_j == 2: b30 += 1
    per5.append((hold, f5, b5, cnt))
    res["gbdt"] = [res["gbdt"][i] + v for i, v in enumerate([f5, f10, f30])]
    res["final"] = [res["final"][i] + v for i, v in enumerate([b5, b10, b30])]
    P("hold=%-8s GBDT@5=%.1f%% FINAL@5=%.1f%% (n=%d)" % (hold, 100.0*f5/max(1,cnt), 100.0*b5/max(1,cnt), cnt))
P("\n===== LOCO合并(扩展池250) =====")
for nm in ("gbdt", "final"):
    P("%-8s " % nm + "  ".join("all@%d=%.1f%%" % (k, 100.0*res[nm][i]/max(1,n)) for i, k in enumerate(K)))
P("FOUNDRY25_DONE %.0fs" % (time.time() - t0))
LOG.close()
