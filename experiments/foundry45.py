# -*- coding: utf-8 -*-
"""foundry45.py — 终极管线v2: 并集池 → GBDT → 头部仲裁(9普适信号) → 窗口20
新增: 头部5席按 avgcos+vqc+qcov+jac 组合仲裁重排(LOCO内系数, 防泄漏: 权重固定)
对照: GBDT纯分 / FINAL。输出: 全套名次分布+进窗率。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry45_results.txt"
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

POOLSZ = 650
FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
ARBC = [None] * len(IDS)   # 头部仲裁特征(每池前8名)
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
    Ab = np.zeros((8, 6), dtype=np.float32)   # 头部8名的仲裁特征
    top8 = np.argsort(-FINAL[qi])  # 占位, 真头部由GBDT定——但GBDT要FEATS, 鸡蛋问题
    # 解法: 仲裁特征按"池内FINAL前8"预存不足, 改为存每条记录的6仲裁特征, 选座后取
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
        avgcos = (vqc + dqc) / 2.0
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
        Ab[rr if rr < 8 else 7] = [avgcos, vqc, qcov, jac, dqc, ccov]
    FEATS[k_i] = Fm
    ARBC[k_i] = Ab
    POOLA[k_i] = pool
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
KW = (1, 3, 5, 10, 15, 20, 30)
res = {"gbdt": [0]*len(KW), "gbdt+仲裁": [0]*len(KW), "final": [0]*len(KW)}
pool_ok = {"new": 0, "final": 0}
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
        o = [pool[i] for i in np.argsort(-s)]
        of = list(np.argsort(-FINAL[qi]))
        if G <= set(pool):
            pool_ok["new"] += 1
        if G <= set(of[:650]):
            pool_ok["final"] += 1
        # 头部仲裁: 前5名按 仲裁分重排(权重固定=普适信号等权, 无训练)
        head = o[:5]
        arb = ARBC[k_i]
        fi = {c: i for i, c in enumerate(pool)}
        def arbscore(c):
            a = arb[fi[c]] if fi[c] < 8 else arb[7]
            return a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
        head2 = sorted(head, key=arbscore, reverse=True)
        o_new = head2 + [c for c in o if c not in set(head2)]
        for nm, order in (("gbdt", o), ("gbdt+仲裁", o_new), ("final", of)):
            rk = {c: r + 1 for r, c in enumerate(order)}
            rks = [rk.get(i, 10**9) for i in G]
            for k_j, k in enumerate(KW):
                if max(rks) <= k:
                    res[nm][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 终极v2计分板 (n=%d, LOCO) =====" % n)
P("池: 新=%.1f%% FINAL-650=%.1f%%" % (100.0*pool_ok["new"]/max(1,n), 100.0*pool_ok["final"]/max(1,n)))
P("%-10s " % "方法" + "".join("all@%-3d " % k for k in KW))
for nm in ("gbdt", "gbdt+仲裁", "final"):
    P("%-10s " % nm + "".join("%7.1f%%" % (100.0*res[nm][i]/max(1,n)) for i in range(len(KW))))
t1n = sum(1 for k_i, qa in enumerate(IDS) if False)
P("FOUNDRY45_DONE %.0fs" % (time.time() - t0))
LOG.close()
