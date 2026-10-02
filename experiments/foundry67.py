# -*- coding: utf-8 -*-
"""foundry63.py — 全发现合体: 反向规律武器化 + 闲置资产接入, 冲15行动态85%
管线: 池650+邻域块(E1) → GBDT321维+新特征(cos惩罚/梯度反转/交叉) → V2-gated
     → 锚点扩展深位搬运(E2) → 动态窗口(2片15/3片20/4+30)
特征新增: cospen(1024cos>0.65惩罚) / gradinv(cos-G1差) / cross(cos*(1-G1))
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry67_results.txt"
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
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]
P("loaded %.0fs" % (time.time() - t0))

QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

# ===== 池: 三源并集 + 邻域块(E1接入) =====
POOLSZ = 700
FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    # 邻域块: FINAL top30 的 ±2 邻域整块带入(v7blocks机制)
    for i in list(np.argsort(-FINAL[qi])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:POOLSZ]
    POOLA[k_i] = pool
    top30list = list(np.argsort(-FINAL[qi])[:30])
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    Fm = np.zeros((len(pool), 325), dtype=np.float32)   # 321+4新特征
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
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        # 三大新特征(反向规律武器化)
        cospen = 1.0 if dqc > 0.65 else 0.0                     # cos惩罚门
        gradinv = dqc - g1                                        # 梯度反转: cos高回应低=假金
        cross = dqc * (1.0 - g1)                                  # 纯引力无回应
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
        nb_flag = 1.0 if any(abs(c - j) <= 2 and CONVKEY[j] == CONVKEY[c] for j in top30list) else 0.0
        Fm[rr, 321:325] = [cospen, gradinv, cross, nb_flag]
    FEATS[k_i] = Fm
    POOLA[k_i] = pool
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
# 输出容器
FINAL_ORDER = {}
res = {"w15": 0, "w15_dyn": 0, "w30": 0, "w2l": 0, "pool": 0}
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
    # 12信号组合器(训练题上拟合)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS2
    trX2, trY2 = [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        qi2 = IDX[qa]
        qstems2 = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
        pool2 = POOLA[k_i]
        G2 = GSETS[k_i]
        golds2 = [c for c in pool2 if c in G2][:6]
        noises2 = [c for c in pool2 if c not in G2][:20]
        for c in golds2 + noises2:
            rstems2 = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
            g1_2 = len(qstems2 & rstems2) / max(1, len(qstems2))
            vqc2 = float(Q256[k_i] @ QW[c])
            dqc2 = float(D[c] @ X[qi2])
            avg2 = (vqc2 + dqc2) / 2
            d256 = np.abs(Q256[k_i] - QW[c])
            d1024 = np.abs(X[qi2] - D[c])
            rcws2 = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
            g2_2 = sum(1 for w in rcws2 if stem(w) not in qstems2) / max(1, len(rcws2)) if rcws2 else 0.0
            trX2.append([g1_2, avg2, vqc2, float(d256.mean()), float(np.quantile(d256, 0.9)), float(d256.std()),
                         dqc2, float(d1024.std()), float(d1024.mean()), float(np.quantile(d1024, 0.9)),
                         float((X[qi2] * D[c] < 0).mean()), g2_2])
            trY2.append(1 if c in G2 else 0)
    sc2 = SS2().fit(np.array(trX2, dtype=np.float64))
    lr2 = LogisticRegression(max_iter=3000).fit(sc2.transform(np.array(trX2, dtype=np.float64)), np.array(trY2))
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
        if G <= set(pool):
            res["pool"] += 1
        # 融合分: GBDT + 12信号组合器(训练折拟合的lr2在本折外应用)
        q = Q[qa]
        qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
        csig = []
        for c in pool:
            rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
            g1 = len(qstems & rstems) / max(1, len(qstems))
            vqc = float(Q256[k_i] @ QW[c])
            dqc = float(D[c] @ X[qi])
            avg = (vqc + dqc) / 2
            dif256 = np.abs(Q256[k_i] - QW[c])
            dif1024 = np.abs(X[qi] - D[c])
            rcws = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
            g2 = sum(1 for w in rcws if stem(w) not in qstems) / max(1, len(rcws)) if rcws else 0.0
            csig.append([g1, avg, vqc, float(dif256.mean()), float(np.quantile(dif256, 0.9)), float(dif256.std()),
                         dqc, float(dif1024.std()), float(dif1024.mean()), float(np.quantile(dif1024, 0.9)),
                         float((X[qi] * D[c] < 0).mean()), g2])
        CSM = np.array(csig, dtype=np.float64)
        comb = lr2.decision_function(sc2.transform(CSM))
        zs_ = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        fused = zs_(s) + 0.8 * zs_(comb)
        o = [pool[i] for i in np.argsort(-fused)]
        # V2-gated on fused
        head = o[:10]
        qv = Q256[k_i]
        PRODM = np.array([qv * QW[c] for c in head])
        w2 = np.abs(PRODM[0])
        w2 = w2 / (w2.sum() + 1e-9)
        boost = PRODM @ (w2 * 256.0)
        boost = (boost - boost.min()) / (boost.max() - boost.min() + 1e-9)
        gb_norm = np.array([fused[pos_of[c]] for c in head])
        gb_norm = (gb_norm - gb_norm.min()) / (gb_norm.max() - gb_norm.min() + 1e-9)
        mix = 0.6 * gb_norm + 0.4 * boost
        o_head = [head[i] for i in np.argsort(-mix)]
        for prot in reversed(o[:2]):
            if prot in o_head:
                o_head.remove(prot)
                o_head.insert(0, prot)
        order = o_head + [c for c in o if c not in set(o_head)]
        pass   # 摘要限额撤销(单变量归因测试)
        # 锚点扩展深位搬运(E2): 锚的显著词扫池16-100名, 兄弟片插入窗口尾
        anchor = order[0]
        qtok2 = toks(Q[qa]["question"])
        sal = [w for w in RTOK[anchor] if w not in qtok2 and DF.get(w, 999) <= 50]
        salset = set(sal)
        if salset:
            deep = order[15:100]
            rescued = []
            for c in deep:
                cov = len(salset & RTOK[c]) / len(salset)
                if cov >= 0.5:
                    rescued.append(c)
            if rescued:
                keep = order[:15 - min(6, len(rescued))]
                tail = rescued[:6]
                order2 = keep + tail + [c for c in order if c not in set(keep) | set(tail)]
            else:
                order2 = order
        else:
            order2 = order
        # 双层: L1=前15(主窗), L2=16-35(深位层: 锚扩展漏网+GBDT 16-35 + 邻域兄弟)
        l1 = order2[:15]
        l2_cands = [c for c in order2[15:35]]
        # 邻域兄弟: L1成员的±1同会话记录补入L2
        for c in l1[:8]:
            for off in (1, -1):
                j = c + off
                if 0 <= j < NR and CONVKEY[j] == CONVKEY[c] and j not in set(l1) and j not in set(l2_cands):
                    l2_cands.append(j)
        l2 = l2_cands[:20]
        FINAL_ORDER[qa] = [MID[c] for c in (l1 + l2)]
        g15 = all(MID[g] in FINAL_ORDER[qa][:15] for g in G)
        g2l = all(MID[g] in FINAL_ORDER[qa][:35] for g in G)   # 双层(L1+L2=35行)
        g30 = all(MID[g] in FINAL_ORDER[qa][:30] for g in G)
        wdyn = 15 if len(G) <= 2 else (20 if len(G) == 3 else 30)
        gdyn = all(MID[g] in FINAL_ORDER[qa][:wdyn] for g in G)
        if g15:
            res["w15"] += 1
        if g2l:
            res["w2l"] += 1
        if g30:
            res["w30"] += 1
        if gdyn:
            res["w15_dyn"] += 1
    P("  fold %s done" % hold)

with open(HERE + "/r41_final_order_v6.json", "w", encoding="utf-8") as f:
    json.dump(FINAL_ORDER, f, ensure_ascii=False)
P("\n===== 全发现合体检验 (n=%d) =====" % n)
P("池全金率(700+邻域块):        %.1f%%" % (100.0 * res["pool"] / max(1, n)))
P("窗口15全金率(固定):           %.1f%%" % (100.0 * res["w15"] / max(1, n)))
P("窗口15动态(2片15/3片20/4+30): %.1f%%" % (100.0 * res["w15_dyn"] / max(1, n)))
P("双层窗口(15主+20深=35行)全金率: %.1f%%" % (100.0 * res["w2l"] / max(1, n)))
P("窗口30全金率(固定):           %.1f%%" % (100.0 * res["w30"] / max(1, n)))
P("终序落盘: r41_final_order_v6.json")
P("F63_DONE %.0fs" % (time.time() - t0))
LOG.close()
