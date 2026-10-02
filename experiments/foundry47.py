# -*- coding: utf-8 -*-
"""foundry47.py — 接话性假说: 假金=高接话性(问题词序复现/句式模板), 真金=回答不复述
对照组: 窗口头部30名内 真金(1) vs 假金(0)
特征:
  R1 词序复现率: 问题词在记录中的最长连续子序列长度(归一)
  R2 框架句式相似: 问题去掉疑问词的骨架词序在记录中的保序命中率(Kendall-tau近似)
  R3 问句镜像: 记录是否以与问题相同的疑问词/框架开头
  R4 词汇复述率: 问题实词的原文连续bigram命中率
  对照: 绝对cos(已知0.72-0.77)
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry47_results.txt"
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

QW_C = {}
def ctoks(qa_i, i):
    if (qa_i, i) not in QW_C:
        QW_C[(qa_i, i)] = toks(RAW[i])
    return QW_C[(qa_i, i)]

# ===== 接话性特征 =====
QW_WORDS = re.compile(r"\b(what|when|where|who|which|how|why|did|do|does|is|are)\b", re.I)
def echo_features(qa, qi, c):
    q = Q[qa]
    ql = q["question"].lower()
    qtoks = re.findall(r"[a-z']+", ql)
    qstop = set("what when where who which how why did do does is are was the a an of in on at to for and or".split())
    qcontent = [w for w in qtoks if w not in qstop and len(w) > 2]
    rl = RAW[c].lower()
    rwords = re.findall(r"[a-z']+", rl)
    rset = set(rwords)
    # R1 词序复现: 问题内容词在记录中的最长保序子序列
    best_len = 0
    cur = 0
    pos_list = []
    for w in qcontent:
        p = rl.find(w)
        pos_list.append((w, p))
    seq = 0
    max_seq = 0
    last_p = -1
    for w, p in pos_list:
        if p >= 0 and p > last_p:
            seq += 1
            last_p = p
            max_seq = max(max_seq, seq)
        elif p >= 0:
            seq = 1
            last_p = p
        else:
            seq = 0
    r1 = max_seq / max(1, len(qcontent))
    # R2 bigram复现: 问题内容bigram在记录中出现的比例
    big = [(qcontent[i], qcontent[i+1]) for i in range(len(qcontent)-1)]
    hit = sum(1 for a, b in big if a in rset and b in rset)
    r2 = hit / max(1, len(big))
    # R3 疑问框架镜像: 记录含问题开头的疑问词或"you"指向
    qhead = qtoks[0] if qtoks else ""
    r3 = 1.0 if (qhead and qhead in rwords[:6]) else 0.0
    # R4 词汇复述率: 问题实词在记录前半段的密度
    half = " ".join(rwords[:max(1, len(rwords)//2)])
    rep = sum(1 for w in qcontent if w in half) / max(1, len(qcontent))
    r4 = rep
    return [r1, r2, r3, r4]

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
        ct = ctoks(qa_i := k_i, c) if False else toks(RAW[c])
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
# ===== 头部30名内 收集 真金/假金 + 接话性特征 =====
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
        head = o[:30]
        for c in head:
            lab = 1 if c in G else 0
            r1, r2, r3, r4 = echo_features(qa, qi, c)
            vqc = float(Q256[k_i] @ QW[c])
            dqc = float(D[c] @ X[qi])
            rows.append([r1, r2, r3, r4, vqc, dqc])
            labs.append(lab)
            holds.append(hold)
F = np.array(rows, dtype=np.float32)
LA = np.array(labs, dtype=np.int8)
HA = np.array(holds)
P("样本=%d 金=%d %.0fs" % (len(LA), int(LA.sum()), time.time() - t0))

gsplit = np.array([int(hashlib.md5((str(h) + "f47").encode()).hexdigest(), 16) % 2 == 0 for h in HA])
FN = ["R1_词序复现", "R2_bigram复现", "R3_疑问镜像", "R4_前半段复述", "对照_256cos", "对照_1024cos"]
P("\n===== 接话性假说检验 (真金=1 vs 假金=0) =====")
P("%-14s %8s %10s %10s" % ("特征", "AUC", "真金均值", "假金均值"))
for j, nm in enumerate(FN):
    try:
        a = roc_auc_score(LA[~gsplit], F[~gsplit, j])
    except Exception:
        a = 0.5
    a = max(a, 1 - a)
    tag = " ← 普适" if a >= 0.55 else ""
    P("%-14s %8.3f %10.3f %10.3f%s" % (nm, a, F[LA == 1, j].mean(), F[LA == 0, j].mean(), tag))
P("FOUNDRY47_DONE %.0fs" % (time.time() - t0))
LOG.close()
