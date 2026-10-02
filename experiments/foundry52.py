# -*- coding: utf-8 -*-
"""foundry52.py — 问题×真金/假金 全数值对终极扫描
A. 标量特征24个(F44b全套)+新增9个词法/语义标量
B. 数列对比: 256/1024逐维差的统计量(均值/方差/偏度/分位数) — "数列形态"特征
C. 全特征组合 vs GBDT头部排序: 二次仲裁AUC
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry52_results.txt"
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
from sklearn.metrics import roc_auc_score
FN52 = []
rows = []
labs = []
holds = []
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
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
        qi = IDX[qa]
        pool = POOLA[k_i]
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)]
        q = Q[qa]
        ql = q["question"].lower()
        qwords = [w for w in re.findall(r"[a-z']+", ql) if w not in QSTOP and len(w) > 2]
        qstems = set(stem(w) for w in qwords)
        qv256 = Q256[k_i]
        qv1024 = X[qi]
        for c in o[:30]:
            lab = 1 if c in G else 0
            ct = ctoks(c)
            inter = qtok & ct
            jac = len(inter) / max(1, len(qtok | ct))
            qcov = len(inter) / max(1, qlen)
            vqc = float(qv256 @ QW[c])
            dqc = float(D[c] @ qv1024)
            # 新增9词法/语义标量
            rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
            g1 = len(qstems & rstems) / max(1, len(qstems))
            rcws = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
            g2 = sum(1 for w in rcws if stem(w) not in qstems) / max(1, len(rcws)) if rcws else 0.0
            # 数列形态: 256/1024逐维差的统计量
            dif256 = np.abs(qv256 - QW[c])
            dif1024 = np.abs(qv1024 - D[c])
            m256, s256 = float(dif256.mean()), float(dif256.std())
            m1024, s1024 = float(dif1024.mean()), float(dif1024.std())
            p90_256 = float(np.quantile(dif256, 0.9))
            p90_1024 = float(np.quantile(dif1024, 0.9))
            neg1024 = float((qv1024 * D[c] < 0).mean())   # 异号维占比
            big256 = float((dif256 > 0.5).mean())          # 大差异维占比
            feat = [jac, qcov, vqc, dqc, (vqc + dqc) / 2, g1, g2,
                    m256, s256, m1024, s1024, p90_256, p90_1024, neg1024, big256]
            rows.append(feat)
            labs.append(lab)
            holds.append(hold)
    P("  fold %s done" % hold)
F = np.array(rows, dtype=np.float32)
LA = np.array(labs, dtype=np.int8)
HA = np.array(holds)
gsplit = np.array([int(hashlib.md5((str(h) + "f52").encode()).hexdigest(), 16) % 2 == 0 for h in HA])
FN52 = ["A_jac", "A_qcov", "A_vqc", "A_dqc", "A_avg", "G1_词干呼应", "G2_增量率",
        "N_256差均值", "N_256差std", "N_1024差均值", "N_1024差std",
        "N_256差p90", "N_1024差p90", "N_异号维率", "N_256大差率"]
P("\n样本=%d 金=%d" % (len(LA), int(LA.sum())))
P("\n===== 问题×真金/假金 全数值对AUC榜 =====")
results = []
for j, nm in enumerate(FN52):
    try:
        a = roc_auc_score(LA[~gsplit], F[~gsplit, j])
        a_tr = roc_auc_score(LA[gsplit], F[gsplit, j])
    except Exception:
        continue
    results.append((max(a, 1 - a), nm, a, a_tr, F[LA == 1, j].mean(), F[LA == 0, j].mean()))
results.sort(reverse=True)
for a, nm, raw, a_tr, mg, mn in results:
    flag = " ← 普适" if (a >= 0.55 and max(a_tr, 1 - a_tr) >= 0.55) else ""
    P("%-13s AUC=%.3f (tr=%.3f) 金=%.3f 假=%.3f%s" % (nm, a, a_tr, mg, mn, flag))
# 组合
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
good_idx = [j for j, (a, nm, _, _, _, _) in enumerate(results) if a >= 0.55]
if len(good_idx) >= 2:
    FG = F[:, good_idx]
    sclr = StandardScaler().fit(FG[gsplit])
    lr = LogisticRegression(max_iter=2000).fit(sclr.transform(FG[gsplit]), LA[gsplit])
    auc_c = roc_auc_score(LA[~gsplit], lr.predict_proba(sclr.transform(FG[~gsplit]))[:, 1])
    P("\n%d条普适信号组合 AUC=%.3f" % (len(good_idx), max(auc_c, 1 - auc_c)))
P("FOUNDRY52_DONE %.0fs" % (time.time() - t0))
LOG.close()
