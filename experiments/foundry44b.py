# -*- coding: utf-8 -*-
"""foundry44.py — 问题×真金/假金 全形态扫描: 找普适弱规律
对照组: 同题内窗口头部(GBDT前8)的真金片 vs 假金片(非金的高分记录)
特征群:
  A 问题-记录: 1024cos/256cos/Jaccard/qcov
  B 问题-记录 交叉: 问题-记录cos × 与团连接度(组合)
  C 位置形态: 记录在对话中的时序位置/与锚片距离
  D 词形: 疑问词残留/代词率/数字率/说话人
判据: 每特征AUC + 拆半; 找 AUC>=0.55 的普适弱信号
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry44b_results.txt"
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
ISRAW = np.array([(json.loads(l).get("kind") or "summary") == "raw"
                  for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()])
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
# 全库记录的时序位置(会话内序号)
RECIDX = {}
for i, m in enumerate(MID):
    m2 = re.match(r".*-(\d+)$", m)
    RECIDX[i] = i
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

POOLSZ = 650
FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
FN = 22
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
    Fm = np.zeros((len(pool), 329), dtype=np.float32)
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
        # D组: 词形/位置
        pron = len(re.findall(r"\b(it|this|that|they|them)\b", RAW[c].lower())) / max(1, len(RAW[c].split()))
        dig = len(re.findall(r"\b\d+\b", RAW[c])) / max(1, len(RAW[c].split()))
        pos_in_pool = rr / max(1, len(pool))
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
        firsthit = None
        for w in qtok:
            p = norm(RAW[c]).find(w)
            if p >= 0:
                firsthit = p / max(1, len(RAW[c]))
                break
        density = 100.0 * len(inter) / max(1, len(RAW[c].split()))
        Fm[rr, 321:327] = [pron, dig, pos_in_pool, float(ISRAW[c]), ccov - qcov, (vqc + dqc) / 2]
        Fm[rr, 327:329] = [firsthit if firsthit is not None else 1.0, density]
    FEATS[k_i] = Fm
    POOLA[k_i] = pool
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
FN44 = ["A_jac", "A_qcov", "A_ccov", "A_qmark", "A_iecho", "A_lr", "A_vqc", "A_vqc-d", "A_dqc", "A_d-v", "A_d-q", "A_q-d",
        "B_d616", "B_d538", "B_d218", "B_d408",
        "D_pron", "D_dig", "D_pos", "D_israw", "D_ccov-qcov", "D_avgcos", "E_firsthit", "E_density"]
auc_rows = []
top1_rescued = 0
n_top1_fake = 0
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
        o = [pool[i] for i in np.argsort(-s)]
        # 收集头部8名: 真金(1) vs 假金(0)
        head = o[:30]
        golds_head = [c for c in head if c in G]
        fake_head = [c for c in head if c not in G]
        if not golds_head or not fake_head:
            continue
        for c in head:
            lab = 1 if c in G else 0
            rr = pool.index(c)
            auc_rows.append((list(FEATS[k_i][rr, 0:12]) + list(FEATS[k_i][rr, 12:16]) +
                             list(FEATS[k_i][rr, 321:329]), lab, hold))
        if head[0] not in G and golds_head:
            n_top1_fake += 1
P("头部样本收集完成 %.0fs" % (time.time() - t0))

FA = np.array([r[0] for r in auc_rows], dtype=np.float32)
LA = np.array([r[1] for r in auc_rows], dtype=np.int8)
HA = np.array([r[2] for r in auc_rows])
gsplit = np.array([int(hashlib.md5((str(h) + "f44").encode()).hexdigest(), 16) % 2 == 0 for h in HA])
P("头部样本=%d 金=%d" % (len(LA), int(LA.sum())))
P("\n===== 真金vs假金 全形态AUC榜 =====")
results = []
for j, nm in enumerate(FN44):
    try:
        a = roc_auc_score(LA[~gsplit], FA[~gsplit, j])
        a_tr = roc_auc_score(LA[gsplit], FA[gsplit, j])
    except Exception:
        continue
    results.append((max(a, 1 - a), nm, a, a_tr))
results.sort(reverse=True)
for a, nm, raw, a_tr in results:
    flag = " ← 普适弱信号" if (max(a, 1 - a) >= 0.55 and max(a_tr, 1 - a_tr) >= 0.55) else ""
    P("  %-12s AUC=%.3f (train侧=%.3f)%s" % (nm, a, a_tr, flag))
P("FOUNDRY44_DONE %.0fs" % (time.time() - t0))
LOG.close()
