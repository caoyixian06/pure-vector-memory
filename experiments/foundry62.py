# -*- coding: utf-8 -*-
"""foundry62.py — 不变1322题与翻错15题的机理剖析
A. 不变题分类: (a)本来就全对(GBDT已装齐,V2无空间动) (b)本来就错太深(差太远,V2够不着)
   → 不变≠浪费, 大部分是"无事可做"
B. 翻错15题: 被挤出的真金与顶进来的候选的V2分差 — 放大算法在哪失手
   具体看: 挤入者与锚的维度对齐度 vs 被挤真金与锚的对齐度 — 为什么假的对齐更高
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry62_results.txt"
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
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

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
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
# A. 不变题细分
unchanged = {"already_ok": 0, "too_far": 0, "in_range_no_flip": 0}
# B. 翻错机理
neg_detail = []
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
        ok_before = all(r in set(o[:5]) for r in G)
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
        o_v = [head[i] for i in np.argsort(-mix)] + [c for c in o if c not in set(head)]
        ok_after = all(r in set(o_v[:5]) for r in G)
        if ok_before == ok_after:
            # 细分不变
            if ok_before:
                unchanged["already_ok"] += 1
            else:
                # 错题: 最浅缺失片的位置
                rk = {c: r + 1 for r, c in enumerate(o_v)}
                worst = max(rk.get(g, 999) for g in G)
                if worst > 15:
                    unchanged["too_far"] += 1
                else:
                    unchanged["in_range_no_flip"] += 1
        elif ok_before and not ok_after:
            # 翻错: 记录挤入者与被挤者
            lost = [g for g in G if g not in set(o_v[:5])][0]
            gainers = [c for c in o_v[:5] if c not in set(o[:5])][:1]
            if gainers:
                gnr = gainers[0]
                # 对齐度: 与锚的256维cos
                anchor = o[0]
                ali_lost = float(QW[lost] @ QW[anchor])
                ali_gnr = float(QW[gnr] @ QW[anchor])
                neg_detail.append((qa, RAW[lost][:55], RAW[gnr][:55], ali_lost, ali_gnr))
    P("  fold %s done" % hold)

P("\n===== A. 不变1322题细分 =====")
for k, v in unchanged.items():
    P("%-20s %4d" % (k, v))
P("\n===== B. 翻错15题机理(被挤真金 vs 挤入者的锚对齐度) =====")
for qa, lt, gt, al, ag in neg_detail[:15]:
    P("  [%s]" % qa)
    P("    被挤真金(锚对齐=%.3f): %s" % (al, lt))
    P("    挤入者(锚对齐=%.3f):   %s" % (ag, gt))
    P("    → 挤入者对齐%s真金" % ("更高! V2被维度相似误导" if ag > al else "更低但仍赢(GBDT分高)"))
P("F62_DONE %.0fs" % (time.time() - t0))
LOG.close()
