# -*- coding: utf-8 -*-
"""foundry40.py — 终极管线实测: 检索层召回 + 金证据进窗率 + 窗内排名分布
全发现合编: 池(F250∪C0250∪邻句) → GBDT压缩器(321维全特征, LOCO) → 窗口30
输出: 池召回 / 进窗率(all/any) / 窗内名次分布(1/2/3/5/8/15/30) vs FINAL对照
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry40_results.txt"
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
P("loaded %.0fs" % (time.time() - t0))

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
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
KW = (1, 2, 3, 5, 8, 10, 15, 20, 30)
res_any = {"new": [0]*len(KW), "final": [0]*len(KW)}
res_all = {"new": [0]*len(KW), "final": [0]*len(KW)}
pool_recall = {"new": 0, "final": 0}
top1 = {"new": 0, "final": 0}
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
        s = clf.decision_function(FEATS[k_i])
        o_new = [pool[i] for i in np.argsort(-s)]
        o_fin = list(np.argsort(-FINAL[qi]))
        if G <= set(pool):
            pool_recall["new"] += 1
        if G <= set(o_fin[:650]):
            pool_recall["final"] += 1
        rkg = {c: r + 1 for r, c in enumerate(o_new)}
        rkf = {c: r + 1 for r, c in enumerate(o_fin)}
        ag = [rkg.get(i, 10**9) for i in G]
        af = [rkf.get(i, 10**9) for i in G]
        for k_j, k in enumerate(KW):
            if min(ag) <= k:
                res_any["new"][k_j] += 1
            if min(af) <= k:
                res_any["final"][k_j] += 1
            if max(ag) <= k:
                res_all["new"][k_j] += 1
            if max(af) <= k:
                res_all["final"][k_j] += 1
        if min(ag) == 1:
            top1["new"] += 1
        if min(af) == 1:
            top1["final"] += 1
    P("  fold %s done" % hold)

P("\n===== 终极管线实测 (n=%d, 池650, LOCO) =====" % n)
P("① 检索层: 全金在池率  新管线=%.1f%%  FINAL-650=%.1f%%" % (
    100.0*pool_recall["new"]/max(1,n), 100.0*pool_recall["final"]/max(1,n)))
P("\n② 金证据进窗率(窗口=K行):")
P("%-6s " % "K" + "".join("%-7d" % k for k in KW))
P("%-6s " % "任一" + "".join("%6.1f%% " % (100.0*res_any["new"][i]/max(1,n)) for i in range(len(KW))))
P("%-6s " % "(FINAL)" + "".join("%6.1f%% " % (100.0*res_any["final"][i]/max(1,n)) for i in range(len(KW))))
P("%-6s " % "全部" + "".join("%6.1f%% " % (100.0*res_all["new"][i]/max(1,n)) for i in range(len(KW))))
P("%-6s " % "(FINAL)" + "".join("%6.1f%% " % (100.0*res_all["final"][i]/max(1,n)) for i in range(len(KW))))
P("\n③ 金证据登上第1名: 新=%.1f%% FINAL=%.1f%%" % (
    100.0*top1["new"]/max(1,n), 100.0*top1["final"]/max(1,n)))
P("FOUNDRY40_DONE %.0fs" % (time.time() - t0))
LOG.close()
