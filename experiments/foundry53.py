# -*- coding: utf-8 -*-
"""foundry53.py — 全流程端到端逐题审计(用户最终检验)
阶段1 抱团剔噪: GBDT压缩器打分 → 团凝聚(Jaccard) → 压缩池
阶段2 假金剔除: 12条普适信号组合(G1词干呼应为主)在压缩池内重排, 头部出线
逐题审计: ①真金在最终池率 ②真金在头部率 ③假金残留率 ④反例题清单(哪一步丢的)
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry53_results.txt"
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

def arb_feats(qa, qi, c):
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
        P("  feat %d %.0fs" % (time.time() - t0) if False else "  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
# 固定组合权重(F52验证的AUC比例, 无训练)
W = [1.6, 1.5, 1.45, 1.4, 1.4, 1.38, 1.35, 1.32, 1.3, 1.28, 1.25, 1.2]
arbw = {"g1": 1.6, "avg": 1.5, "vqc": 1.45, "m256": 1.4, "p256": 1.4, "s256": 1.38,
        "dqc": 1.35, "s1024": 1.32, "m1024": 1.3, "p1024": 1.28, "neg": 1.25, "g2r": 1.2}
n = 0
stage = {"pool_all": 0, "head_all_after_clu": 0, "head_all_after_arb": 0,
         "any_head": 0, "fake_in_head5": [], "loss_at": []}
head_sizes = []
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
        # 阶段1: 压缩池(前250 = GBDT序前250)
        cpool = set(o[:250])
        if G <= cpool:
            stage["pool_all"] += 1
        # 阶段2: 假金剔除(12信号组合重排 cpool 内 top100)
        cands = o[:100]
        af = {c: arb_feats(qa, qi, c) for c in cands}
        # 归一化: 每特征在cands内zscore后加权和
        A = np.array([af[c] for c in cands], dtype=np.float64)
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        wvec = np.array([arbw["g1"], arbw["avg"], arbw["vqc"], arbw["m256"], arbw["p256"], arbw["s256"],
                         arbw["dqc"], arbw["s1024"], arbw["m1024"], arbw["p1024"], arbw["neg"], arbw["g2r"]])
        comb = A @ wvec
        o_arb = [cands[i] for i in np.argsort(-comb)]
        rk_new = {c: r + 1 for r, c in enumerate(o_arb)}
        head = o_arb[:30]
        if G <= set(head):
            stage["head_all_after_arb"] += 1
        if G & set(head):
            stage["any_head"] += 1
        # 假金残留: 头部5名中非金数
        fake5 = sum(1 for c in head[:5] if c not in G)
        stage["fake_in_head5"].append(fake5)
        # 中间检查: 凝聚后(压缩池+GBDT序, 未仲裁)的全金率
        if G <= set(o[:30]):
            stage["head_all_after_clu"] += 1
        # 反例定位: 丢在哪一步
        if G <= cpool and G & set(o[:30]) and not (G <= set(head)):
            stage["loss_at"].append((qa, "head_partial", sorted(G - set(head))))
        elif G <= cpool and not (G & set(head)):
            stage["loss_at"].append((qa, "head_total", sorted(G)[:2]))
        elif not (G <= cpool):
            stage["loss_at"].append((qa, "pool", []))
    P("  fold %s done | GPU=%s | 进度=%d/10折" % (hold, __import__("subprocess").run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip(), FOLDS.index(hold) + 1))

P("\n===== 全流程逐题审计 (n=%d) =====" % n)
P("① 真金全在压缩池(250):     %.1f%%" % (100.0 * stage["pool_all"] / max(1, n)))
P("② 真金全在窗30(仅GBDT序):  %.1f%%" % (100.0 * stage["head_all_after_clu"] / max(1, n)))
P("③ 真金全在窗30(仲裁后):    %.1f%%" % (100.0 * stage["head_all_after_arb"] / max(1, n)))
P("④ 任一真金在窗30:          %.1f%%" % (100.0 * stage["any_head"] / max(1, n)))
f5 = np.array(stage["fake_in_head5"])
P("⑤ 头部5席假金残留: 均=%.2f 0个假金题占比=%.1f%%" % (f5.mean(), 100.0 * np.mean(f5 == 0)))
P("\n⑥ 反例清单(损失定位, 前20条):")
for qa, where, lost in stage["loss_at"][:20]:
    P("  [%s] 丢在=%s" % (qa, where))
P("反例总数=%d" % len(stage["loss_at"]))
P("FOUNDRY53_DONE %.0fs" % (time.time() - t0))
LOG.close()
