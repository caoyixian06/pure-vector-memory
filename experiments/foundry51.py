# -*- coding: utf-8 -*-
"""foundry51.py — 回答性指纹: 问题动词骨架+信息增量槽
对照组: 窗口头部30 真金(1) vs 假金(0)
特征:
  G1 动词骨架呼应: 问题的动词词根在记录中以任意形态出现(词干匹配)
  G2 信息增量率: 记录中"非问题词"的内容词占比(新信息载体)
  G3 骨架+增量组合: 动词呼应 × 新实体(大写专名不在问题中)
  G4 增量绝对量: 记录中未在问题出现的低频内容词数
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry51_results.txt"
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
# DF表(信息增量的低频判定)
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
P("loaded+DF %.0fs" % (time.time() - t0))

QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this these those it its i you he she they we my your his her their".split())

def ans_features(qa, qi, c):
    q = Q[qa]
    ql = q["question"].lower()
    qwords = [w for w in re.findall(r"[a-z']+", ql) if w not in QSTOP]
    qstems = set(stem(w) for w in qwords)
    # G1 动词/内容词干呼应: 问题词干在记录中以词干形态出现
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
    g1 = len(qstems & rstems) / max(1, len(qstems))
    # G2 信息增量率: 记录内容词中非问题词干占比
    rcws = [w for w in re.findall(r"[a-z']+", RAW[c].lower()) if len(w) > 3 and w not in QSTOP]
    if rcws:
        new = sum(1 for w in rcws if stem(w) not in qstems)
        g2 = new / len(rcws)
    else:
        g2 = 0.0
    # G3 新实体: 大写专名(不在问题中)出现
    qcaps = set(w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", q["question"]))
    newcaps = [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", RAW[c]) if w.lower() not in qcaps]
    g3 = min(1.0, len(newcaps) / 2.0)
    # G4 增量绝对量: 低频(DF<80)非问题内容词数
    g4 = sum(1 for w in rcws if stem(w) not in qstems and DF.get(stem(w), 999) < 80)
    return [g1, g2, g3, g4]

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
rows = []
labs = []
holds = []
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
        for c in o[:30]:
            lab = 1 if c in G else 0
            g = ans_features(qa, qi, c)
            vqc = float(Q256[k_i] @ QW[c])
            dqc = float(D[c] @ X[qi])
            rows.append(g + [vqc, dqc])
            labs.append(lab)
            holds.append(hold)
F = np.array(rows, dtype=np.float32)
LA = np.array(labs, dtype=np.int8)
HA = np.array(holds)
gsplit = np.array([int(hashlib.md5((str(h) + "f51").encode()).hexdigest(), 16) % 2 == 0 for h in HA])
P("样本=%d 金=%d" % (len(LA), int(LA.sum())))
FN = ["G1_词干呼应", "G2_增量率", "G3_新实体", "G4_低频增量", "对照_256cos", "对照_1024cos"]
P("\n===== 回答性指纹检验 (真金=1) =====")
for j, nm in enumerate(FN):
    try:
        a = roc_auc_score(LA[~gsplit], F[~gsplit, j])
    except Exception:
        a = 0.5
    a = max(a, 1 - a)
    tag = " ← 普适" if a >= 0.55 else ""
    P("%-12s AUC=%.3f 真金=%.3f 假金=%.3f%s" % (nm, a, F[LA == 1, j].mean(), F[LA == 0, j].mean(), tag))
P("FOUNDRY51_DONE %.0fs" % (time.time() - t0))
LOG.close()
