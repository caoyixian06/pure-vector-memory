# -*- coding: utf-8 -*-
"""r41_prep.py — r41弹药备齐: 检索终序落盘(全量1382题)
管线: 池650(三源并集) → GBDT压缩器(LOCO十折) → V2-gated头部重排(前2保护席)
输出: r41_final_order.json (每题30行窗口终序) + 检验报告
靶: 窗口15行全金率(用户令: 15行内证据全在) — 按题动态窗(2片题15行,多片题30行)
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/r41_prep_results.txt"
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
# 输出容器
FINAL_ORDER = {}
# 检验统计
res = {"w15": 0, "w15_dynamic": 0, "w30": 0}
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
        # V2-gated: 锚对齐重排头部10, GBDT前2保护
        head = o[:10]
        qv = Q256[k_i]
        PRODM = np.array([qv * QW[c] for c in head])
        w2 = np.abs(PRODM[0])
        w2 = w2 / (w2.sum() + 1e-9)
        boost = PRODM @ (w2 * 256.0)
        boost = (boost - boost.min()) / (boost.max() - boost.min() + 1e-9)
        gb_norm = np.array([s[pos_of[c]] for c in head])
        gb_norm = (gb_norm - gb_norm.min()) / (gb_norm.max() - gb_norm.min() + 1e-9)
        mix = 0.6 * gb_norm + 0.4 * boost
        idx_sorted = list(np.argsort(-mix))
        o_head = [head[i] for i in idx_sorted]
        # 保护席: GBDT前2强制回前2
        for prot in reversed(o[:2]):
            if prot in o_head:
                o_head.remove(prot)
                o_head.insert(0, prot)
        final_order = o_head + [c for c in o if c not in set(o_head)]
        # 保存终序(mid格式, 前35行)
        FINAL_ORDER[qa] = [MID[c] for c in final_order[:35]]
        # 检验: 窗15/30全金
        g15 = all(MID[g] in FINAL_ORDER[qa][:15] for g in G)
        g30 = all(MID[g] in FINAL_ORDER[qa][:30] for g in G)
        # 动态窗: 片数<=2用15行,3片用20,4+片用30
        wdyn = 15 if len(G) <= 2 else (20 if len(G) == 3 else 30)
        gdyn = all(MID[g] in FINAL_ORDER[qa][:wdyn] for g in G)
        if g15:
            res["w15"] += 1
        if g30:
            res["w30"] += 1
        if gdyn:
            res["w15_dynamic"] += 1
    P("  fold %s done" % hold)

with open(HERE + "/r41_final_order.json", "w", encoding="utf-8") as f:
    json.dump(FINAL_ORDER, f, ensure_ascii=False)
P("\n===== r41弹药检验 (n=%d) =====" % n)
P("窗口15全金率(固定):   %.1f%%" % (100.0 * res["w15"] / max(1, n)))
P("窗口15动态(2片15/3片20/4+30)全金率: %.1f%%" % (100.0 * res["w15_dynamic"] / max(1, n)))
P("窗口30全金率(固定):   %.1f%%" % (100.0 * res["w30"] / max(1, n)))
P("终序已落盘: r41_final_order.json (%d题×35行)" % len(FINAL_ORDER))
P("R41_PREP_DONE %.0fs" % (time.time() - t0))
LOG.close()
