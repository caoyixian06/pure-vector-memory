# -*- coding: utf-8 -*-
"""foundry70.py — 审计泄漏消融: A(FINAL特征列)/E(训练采噪会话侧漏)/D(分母口径) 定量
纯CPU零GPU零API。臂:
  ctrl        F68复刻锚(验证可复现: 动态66.3 / 双层72.3)
  nofinal     删FINAL特征列317 (A项: FINAL在同一基准调参的泄漏)
  sessfilt    训练采噪排除hold会话候选行 (E项: 测试会话记录以负例身份进训练)
  light       nofinal+sessfilt
  clean_full  池改C0+256双通道构造(零FINAL依赖)+nofinal+sessfilt — 无FINAL真实力
全部臂双分母: nG(金非空,F68口径) / 1382全量 (D项)
clean_full终序落盘 r41_final_order_v8_clean.json
资源: OMP_NUM_THREADS=4, 峰值内存~1.6GB, 两套池顺序构造逐套释放
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry70_results.txt"
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
QWQ256 = l2n(Q256 @ QW.T)
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
NALL = len(IDS)
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]
TWIN = {}
MID2I = {m: i for i, m in enumerate(MID)}
for i, m in enumerate(MID):
    tw = REC.get(m, {}).get("raw_of") or REC.get(m, {}).get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
P("loaded %.0fs NR=%d NQ=%d" % (time.time() - t0, NR, NALL))

QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]
POOLSZ = 700

def build_pool(k_i, mode):
    qi = k_i
    if mode == "fc":
        pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
        top30 = list(np.argsort(-FINAL[qi])[:30])
    else:
        pool = set(np.argsort(-C0[qi])[:350]) | set(np.argsort(-QWQ256[k_i])[:350])
        top30 = list(np.argsort(-C0[qi])[:30])
    for i in top30:
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    return sorted(pool)[:POOLSZ], top30

def build_feats(k_i, pool, top30list):
    qa = IDS[k_i]
    qi = k_i
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(q["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = (q.get("question") or "x").strip().lower().split()[0] if (q.get("question") or "").strip() else ""
    n = len(pool)
    Fm = np.zeros((n, 329), dtype=np.float32)
    CS = np.zeros((n, 12), dtype=np.float64)
    for rr in range(n):
        c = pool[rr]
        ct = ctoks(c)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lr = len(RAW[c]) / max(1, len(q.get("question") or "x"))
        vqc = float(qv256 @ QW[c])
        dqc = float(D[c] @ qv1024)
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        cospen = 1.0 if dqc > 0.65 else 0.0
        gradinv = dqc - g1
        cross = dqc * (1.0 - g1)
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
        nb_flag = 1.0 if any(abs(c - j) <= 2 and CONVKEY[j] == CONVKEY[c] for j in top30list) else 0.0
        rl_ = RAW[c].lower()
        seq = 0; mx = 0; last_p = -1
        for w in qstems:
            p = rl_.find(w)
            if p >= 0 and p > last_p:
                seq += 1; last_p = p; mx = max(mx, seq)
            elif p >= 0:
                seq = 1; last_p = p
            else:
                seq = 0
        r1 = mx / max(1, len(qstems))
        big = [(qstems_list[i], qstems_list[i + 1]) for i in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAW[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        tw = TWIN.get(c)
        twin_q = float(D[tw] @ qv1024) if tw is not None else 0.0
        Fm[rr, 321:329] = [cospen, gradinv, cross, nb_flag, r1, r2, wvotes, twin_q]
        avg = (vqc + dqc) / 2
        dif256 = np.abs(qv256 - QW[c])
        dif1024 = np.abs(qv1024 - D[c])
        rcws = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
        g2 = sum(1 for w in rcws if stem(w) not in qstems) / max(1, len(rcws)) if rcws else 0.0
        CS[rr] = [g1, avg, vqc, float(dif256.mean()), float(np.quantile(dif256, 0.9)), float(dif256.std()),
                  dqc, float(dif1024.std()), float(dif1024.mean()), float(np.quantile(dif1024, 0.9)),
                  float((qv1024 * D[c] < 0).mean()), g2]
    return Fm, CS

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as SS2

COL_KEEP = np.array([i for i in range(329) if i != 317])  # A项: 剔除FINAL列

def run_arm(name, FEATS, CSIGS, POOL, drop_final, sessfilt, save_order):
    res = {"w15": 0, "w15_dyn": 0, "w30": 0, "w2l": 0, "pool": 0}
    n = 0
    FORD = {}
    cols = COL_KEEP if drop_final else None
    for hold in FOLDS:
        trF, trY = [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool = POOL[k_i]
            G = GSETS[k_i]
            gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
            if sessfilt:
                noise_rows = [rr for rr, c in enumerate(pool) if c not in G and CONVKEY[c] != hold]
            else:
                noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
            noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
            for rr in gold_rows:
                trF.append(FEATS[k_i][rr]); trY.append(1)
            for rr in noisepick:
                trF.append(FEATS[k_i][rr]); trY.append(0)
        trF = np.array(trF, dtype=np.float32)
        if cols is not None:
            trF = trF[:, cols]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(trF, np.array(trY, dtype=np.int8))
        trX2, trY2 = [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool2 = POOL[k_i]
            G2 = GSETS[k_i]
            golds2 = [c for c in pool2 if c in G2][:6]
            noises2 = [c for c in pool2 if c not in G2][:20]
            pos2 = {c: i for i, c in enumerate(pool2)}
            for c in golds2:
                trX2.append(CSIGS[k_i][pos2[c]]); trY2.append(1)
            for c in noises2:
                trX2.append(CSIGS[k_i][pos2[c]]); trY2.append(0)
        sc2 = SS2().fit(np.array(trX2, dtype=np.float64))
        lr2 = LogisticRegression(max_iter=3000).fit(sc2.transform(np.array(trX2, dtype=np.float64)), np.array(trY2))
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] != hold:
                continue
            G = GSETS[k_i]
            if not G:
                continue
            n += 1
            pool = POOL[k_i]
            pos_of = {c: i for i, c in enumerate(pool)}
            FMq = FEATS[k_i] if cols is None else FEATS[k_i][:, cols]
            s = clf.decision_function(FMq)
            comb = lr2.decision_function(sc2.transform(CSIGS[k_i]))
            zs_ = lambda a: (a - a.mean()) / (a.std() + 1e-9)
            fused = zs_(s) + 0.8 * zs_(comb)
            o = [pool[i] for i in np.argsort(-fused)]
            if G <= set(pool):
                res["pool"] += 1
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
            l1 = order2[:15]
            l2_cands = [c for c in order2[15:35]]
            for c in l1[:8]:
                for off in (1, -1):
                    j = c + off
                    if 0 <= j < NR and CONVKEY[j] == CONVKEY[c] and j not in set(l1) and j not in set(l2_cands):
                        l2_cands.append(j)
            l2 = l2_cands[:20]
            full35 = l1 + l2
            if save_order:
                FORD[qa] = [MID[c] for c in full35]
            g15 = all(g in full35[:15] for g in G)
            g2l = all(g in full35[:35] for g in G)
            g30 = all(g in full35[:30] for g in G)
            wdyn = 15 if len(G) <= 2 else (20 if len(G) == 3 else 30)
            gdyn = all(g in full35[:wdyn] for g in G)
            if g15:
                res["w15"] += 1
            if g2l:
                res["w2l"] += 1
            if g30:
                res["w30"] += 1
            if gdyn:
                res["w15_dyn"] += 1
        P("  [%s] fold %s done %.0fs" % (name, hold[-12:], time.time() - t0))
    P("[%s] nG=%d (空金%d题双分母下计错)" % (name, n, NALL - n))
    P("  池全金率:      %5.1f%% / %5.1f%%" % (100.0 * res["pool"] / max(1, n), 100.0 * res["pool"] / NALL))
    P("  窗15固定:      %5.1f%% / %5.1f%%" % (100.0 * res["w15"] / max(1, n), 100.0 * res["w15"] / NALL))
    P("  窗15动态:      %5.1f%% / %5.1f%%" % (100.0 * res["w15_dyn"] / max(1, n), 100.0 * res["w15_dyn"] / NALL))
    P("  窗30固定:      %5.1f%% / %5.1f%%" % (100.0 * res["w30"] / max(1, n), 100.0 * res["w30"] / NALL))
    P("  双层35:        %5.1f%% / %5.1f%%" % (100.0 * res["w2l"] / max(1, n), 100.0 * res["w2l"] / NALL))
    return FORD

# ===== 套1: 池final+c0 (F68原版池) → ctrl/nofinal/sessfilt/light =====
POOL1, TOP1 = [None] * NALL, [None] * NALL
for k_i in range(NALL):
    POOL1[k_i], TOP1[k_i] = build_pool(k_i, "fc")
FEATS1, CSIG1 = [None] * NALL, [None] * NALL
for k_i in range(NALL):
    FEATS1[k_i], CSIG1[k_i] = build_feats(k_i, POOL1[k_i], TOP1[k_i])
    if k_i % 400 == 0:
        P("  P1 feat %d %.0fs" % (k_i, time.time() - t0))
P("P1 built %.0fs" % (time.time() - t0))
run_arm("ctrl", FEATS1, CSIG1, POOL1, False, False, False)
run_arm("nofinal(A项)", FEATS1, CSIG1, POOL1, True, False, False)
run_arm("sessfilt(E项)", FEATS1, CSIG1, POOL1, False, True, False)
run_arm("light(A+E)", FEATS1, CSIG1, POOL1, True, True, False)
del FEATS1
P("P1 arms done %.0fs" % (time.time() - t0))

# ===== 套2: 池C0+256双通道(零FINAL) → clean_full =====
POOL2, TOP2 = [None] * NALL, [None] * NALL
for k_i in range(NALL):
    POOL2[k_i], TOP2[k_i] = build_pool(k_i, "clean")
FEATS2, CSIG2 = [None] * NALL, [None] * NALL
for k_i in range(NALL):
    FEATS2[k_i], CSIG2[k_i] = build_feats(k_i, POOL2[k_i], TOP2[k_i])
    if k_i % 400 == 0:
        P("  P2 feat %d %.0fs" % (k_i, time.time() - t0))
P("P2 built %.0fs" % (time.time() - t0))
FORD2 = run_arm("clean_full(A+E+池无FINAL)", FEATS2, CSIG2, POOL2, True, True, True)
with open(HERE + "/r41_final_order_v8_clean.json", "w", encoding="utf-8") as f:
    json.dump(FORD2, f, ensure_ascii=False)
P("clean_full终序落盘: r41_final_order_v8_clean.json")
P("F70_DONE %.0fs" % (time.time() - t0))
LOG.close()
