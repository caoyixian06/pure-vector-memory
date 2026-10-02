# -*- coding: utf-8 -*-
"""foundry42.py — 为什么金金不能进第一: 头部竞争解剖+团属信号二次定位
①头部身份普查: top1/top3里金vs假金(同会话高GBDT分噪声)的构成
②真金第1 vs 假金第1的区别: 与2-5名团的连接度(片团信号)
③二次定位: 用"团属连接度"重排头部, 登顶率能提多少
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry42_results.txt"
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

# 收集: 头部身份普查 + 真金第1 vs 假金第1 的团连接度 + 二次定位实测
K = (1, 2, 3, 5)
head_gold = {"gbdt": [0]*len(K), "final": [0]*len(K)}
top1_conn = {"gold": [], "fake": []}
top1_before = top1_after = n = 0
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
        # ①头部身份
        for k_j, k in enumerate(K):
            if G & set(o[:k]):
                head_gold["gbdt"][k_j] += 1
        of = list(np.argsort(-FINAL[qi]))
        for k_j, k in enumerate(K):
            if G & set(of[:k]):
                head_gold["final"][k_j] += 1
        # ②top1是假金时, 真金与"2-6名团"的连接度 vs 假金连接度
        t1 = o[0]
        if t1 not in G:
            golds_in_win = [g for g in G if g in set(o[:30])]
            if golds_in_win:
                # 连接度 = 与2-6名(排除top1)的双空间均值相似
                ref = o[1:6]
                conn_t1 = np.mean([sim(t1, r) for r in ref])
                conns_gold = [np.mean([sim(g, r) for r in ref if r != g]) for g in golds_in_win]
                top1_conn["fake"].append(conn_t1)
                top1_conn["gold"].append(max(conns_gold))
                # ③二次定位: 头部2-8名中, 与2-6名团连接度最高者提为第1
                cand_head = o[1:8]
                resc = None
                bestc, bv = None, -1e18
                for c in cand_head:
                    cc = np.mean([sim(c, r) for r in ref if r != c])
                    if cc > bv:
                        bv, bestc = cc, c
                if bestc in G:
                    top1_after += 1
        if min([o.index(g) for g in G if g in o]) == 0 if any(g in o for g in G) else False:
            top1_before += 1
P("\n===== ①头部身份普查(n=%d) =====" % n)
P("头部含金(任一片进K): ")
P("K:      " + "".join("%7d" % k for k in K))
for nm in ("gbdt", "final"):
    P("%-6s " % nm + "".join("%8.1f%%" % (100.0 * head_gold[nm][i] / max(1, n)) for i in range(len(K))))
P("\n===== ②真金第1 vs 假金第1 的团连接度 =====")
if top1_conn["gold"]:
    P("假金top1与2-6团连接度: %.3f (n=%d)" % (np.mean(top1_conn["fake"]), len(top1_conn["fake"])))
    P("真金与2-6团连接度:     %.3f (同题)" % np.mean(top1_conn["gold"]))
    P("真金连接度>假金连接度的题占比: %.1f%%" % (100.0 * np.mean([g > f for g, f in zip(top1_conn["gold"], top1_conn["fake"])])))
P("\n===== ③二次定位(头部2-8名中按团连接度重选第1) =====")
P("二次定位后登顶率: %.1f%% (原登顶率=%.1f%%)" % (
    100.0 * top1_after / max(1, n), 100.0 * top1_before / max(1, n)))
P("FOUNDRY42_DONE %.0fs" % (time.time() - t0))
LOG.close()
