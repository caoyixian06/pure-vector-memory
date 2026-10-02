# -*- coding: utf-8 -*-
"""foundry46.py — 团内结构定位: 问题引力在团内的顶点假设
①逐团测: 真金在"团内问题cos排名"的分布(应该是顶点=第1名富集)
②对照: 假金在各自团的团内排名(应该均匀)
③落地: 团内问题排名第1的成员获得"团代表"加成, 重排头部
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry46_results.txt"
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
def sim(a, b):
    return 0.5 * float(D[a] @ D[b]) + 0.5 * float(QW[a] @ QW[b])
K = (5, 15, 30)
res = {"gbdt": [0]*3, "gbdt+团代表": [0]*3}
# 统计: 团内问题排名第1的成员是金/非金的频次
vtx_gold = vtx_fake = 0
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
        rk = {c: r + 1 for r, c in enumerate(o)}
        for k_j, k in enumerate(K):
            if all(r in set(o[:k]) for r in G):
                res["gbdt"][k_j] += 1
        # 凝聚簇(F41同法)
        win = o[:30]
        clusters = [[c] for c in win]
        active = list(range(len(clusters)))
        merged = True
        while merged:
            merged = False
            bi = bj = -1
            bs = 0.60
            for ii in range(len(clusters)):
                if ii not in active:
                    continue
                for jj in range(ii + 1, len(clusters)):
                    if jj not in active:
                        continue
                    ms = max(sim(a, b) for a in clusters[ii] for b in clusters[jj])
                    if ms > bs:
                        bs, bi, bj = ms, ii, jj
            if bi >= 0:
                clusters[bi].extend(clusters[bj])
                active.remove(bj)
                merged = True
        # ①团代表: 每簇内问题cos(双空间平均)第1名成员 = "团代表", 提到簇内最前
        qv1024 = X[qi]
        qv256 = Q256[k_i]
        new_order = []
        for ii in active:
            members = clusters[ii]
            qcos = [(0.5 * float(qv1024 @ D[m]) + 0.5 * float(qv256 @ QW[m]), m) for m in members]
            qcos.sort(reverse=True)
            vtx_member = qcos[0][1]
            if qcos[0][1] in G:
                vtx_gold += 1
            elif len(members) >= 2:
                vtx_fake += 1
            # 簇内按问题cos排序展开(代表=顶点在最前)
            for _, m in qcos:
                new_order.append(m)
        for c in o:
            if c not in set(new_order):
                new_order.append(c)
        rkn = {c: r + 1 for r, c in enumerate(new_order)}
        for k_j, k in enumerate(K):
            if all(r in set(new_order[:k]) for r in G):
                res["gbdt+团代表"][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 团内顶点假设检验 =====")
P("团代表(团内问题cos第1名)是金=%d 是假金=%d (金占比=%.1f%%)" % (
    vtx_gold, vtx_fake, 100.0 * vtx_gold / max(1, vtx_gold + vtx_fake)))
P("\n===== 团代表重排计分板 (n=%d) =====" % n)
for nm in ("gbdt", "gbdt+团代表"):
    a, b, c = res[nm]
    P("%-12s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%%" % (
        nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n)))
P("FOUNDRY46_DONE %.0fs" % (time.time() - t0))
LOG.close()
