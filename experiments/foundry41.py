# -*- coding: utf-8 -*-
"""foundry41.py — 用户两阶段闭环: ①片找片聚合(窗口内) ②问题×聚合定位
阶段1: GBDT压缩器产窗口30 → 窗口内片-片相似度凝聚(相似片并入簇, 簇心更新)
阶段2: 每簇打分 = 问题相似度(簇心×问题) + 簇内GBDT峰分 → 重排簇 → 展开回记录序
靶: all-gold@15/30 vs 纯GBDT。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry41_results.txt"
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
KW = (15, 30)
res = {"gbdt": [0]*2, "gbdt+聚合": [0]*2, "final": [0]*2}
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
        o_gbdt = [pool[i] for i in np.argsort(-s)]
        rk = {c: r + 1 for r, c in enumerate(o_gbdt)}
        # 纯GBDT
        for k_j, k in enumerate(KW):
            if all(r in set(o_gbdt[:k]) for r in G):
                res["gbdt"][k_j] += 1
        of = list(np.argsort(-FINAL[qi]))
        rkf = {c: r + 1 for r, c in enumerate(of)}
        for k_j, k in enumerate(KW):
            if all(r in set(of[:k]) for r in G):
                res["final"][k_j] += 1
        # ===== 阶段1: 窗口内片找片聚合(凝聚聚类) =====
        win = o_gbdt[:30]
        win_set = set(win)
        # 相似度: 双空间cos组合
        def sim(a, b):
            return 0.5 * float(D[a] @ D[b]) + 0.5 * float(QW[a] @ QW[b])
        # 凝聚: 每条自成一簇, 相似>θ合并(贪心)
        TH = 0.60
        clusters = [[c] for c in win]
        cl_active = list(range(len(clusters)))
        merged = True
        while merged:
            merged = False
            best_ii = best_jj = -1
            best_s = TH
            for ii in range(len(clusters)):
                if ii not in cl_active:
                    continue
                for jj in range(ii + 1, len(clusters)):
                    if jj not in cl_active:
                        continue
                    # 簇间相似 = 两簇任意成员最大相似
                    ms = max(sim(a, b) for a in clusters[ii] for b in clusters[jj])
                    if ms > best_s:
                        best_s = ms; best_ii, best_jj = ii, jj
            if best_ii >= 0:
                clusters[best_ii].extend(clusters[best_jj])
                cl_active.remove(best_jj)
                merged = True
        # ===== 阶段2: 问题×聚合定位 =====
        qv1024 = X[qi]
        qv256 = Q256[k_i]
        cl_score = []
        for ii in cl_active:
            members = clusters[ii]
            cD = l2n(D[members].mean(0, keepdims=True))[0]
            cQ = l2n(QW[members].mean(0, keepdims=True))[0]
            qs = 0.5 * float(cD @ qv1024) + 0.5 * float(cQ @ qv256)
            gpeak = max(float(rk.get(m, 99)) for m in members)
            cl_score.append((-(qs) - 0.02 * gpeak, ii, qs))
        cl_score.sort()
        # 簇序展开回记录序: 簇按分排, 簇内按GBDT分排
        new_order = []
        for _, ii, _ in cl_score:
            members = sorted(clusters[ii], key=lambda m: rk.get(m, 999))
            new_order.extend(members)
        for c in o_gbdt:
            if c not in set(new_order):
                new_order.append(c)
        rkn = {c: r + 1 for r, c in enumerate(new_order)}
        for k_j, k in enumerate(KW):
            if all(r in set(new_order[:k]) for r in G):
                res["gbdt+聚合"][k_j] += 1
    P("  fold %s done" % hold)

P("\n===== 两阶段闭环计分板 (n=%d) =====" % n)
for nm in ("gbdt", "gbdt+聚合", "final"):
    a, b = res[nm]
    P("%-10s all@15=%.1f%% all@30=%.1f%%" % (nm, 100.0*a/max(1,n), 100.0*b/max(1,n)))
P("FOUNDRY41_DONE %.0fs" % (time.time() - t0))
LOG.close()
