# -*- coding: utf-8 -*-
"""foundry32.py — 多片金证据的独特相似性: 金-金 vs 金-噪声 vs 噪-噪 三方对比
形态: 文本(Jaccard/共享独有词) / 256(qwen cos) / 1024(bge cos)
判据: 兄弟对 vs 干扰对 的三形态AUC + 拆半
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry32_results.txt"
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
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

# ===== 收集三种配对(同一题的池内) =====
import random
rng = np.random.RandomState(5)
FEATN = 5
rows = []   # (jac, v256, d1024, same_conv, label)  label: 1=金金 0=金噪
qa_of = []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if len(G) < 2:
        continue
    pool = [i for i in np.argsort(-C0[qi])[:150]]
    golds = sorted(G)
    noises = [i for i in pool if i not in G]
    if not noises:
        continue
    # 金金对(全部两两, 上限8对)
    gg_pairs = [(a, b) for ai, a in enumerate(golds) for b in golds[ai+1:]][:8]
    # 金噪对(随机配, 数量=金金对×2)
    nneed = min(2 * len(gg_pairs), len(golds) * len(noises))
    gn_pairs = []
    for _ in range(nneed * 2):
        if len(gn_pairs) >= nneed:
            break
        a = golds[rng.randint(len(golds))]
        b = noises[rng.randint(len(noises))]
        if a != b:
            gn_pairs.append((a, b))
    for a, b in gg_pairs:
        ta, tb = ctoks(a), ctoks(b)
        jac = len(ta & tb) / max(1, len(ta | tb))
        v = float(QW[a] @ QW[b])
        d = float(D[a] @ D[b])
        sc = 1 if CONVKEY[a] == CONVKEY[b] else 0
        rows.append((jac, v, d, sc, 1))
        qa_of.append(k_i)
    for a, b in gn_pairs:
        ta, tb = ctoks(a), ctoks(b)
        jac = len(ta & tb) / max(1, len(ta | tb))
        v = float(QW[a] @ QW[b])
        d = float(D[a] @ D[b])
        sc = 1 if CONVKEY[a] == CONVKEY[b] else 0
        rows.append((jac, v, d, sc, 0))
        qa_of.append(k_i)
    if k_i % 200 == 0:
        P("  pairs %d %.0fs" % (k_i, time.time() - t0))
F = np.array(rows, dtype=np.float64)
Y = F[:, 4].astype(int)
P("金金对=%d 金噪对=%d" % (int(Y.sum()), int(len(Y) - Y.sum())))

from sklearn.metrics import roc_auc_score
gsplit = np.array([int(hashlib.md5((str(q) + "f32").encode()).hexdigest(), 16) % 2 == 0 for q in qa_of])
P("\n===== 三形态对比 (金金=1 vs 金噪=0) =====")
for j, nm in enumerate(("文本Jaccard", "256cos", "1024cos")):
    a = roc_auc_score(Y[~gsplit], F[~gsplit, j])
    P("%-10s AUC=%.3f | 金金均值=%.3f 金噪均值=%.3f" % (
        nm, max(a, 1-a), F[gsplit & (Y==1), j].mean(), F[gsplit & (Y==0), j].mean()))
# 同会话率
a = roc_auc_score(Y[~gsplit], F[~gsplit, 3])
P("%-10s AUC=%.3f | 金金同会话率=%.1f%% 金噪同会话率=%.1f%%" % (
    "同会话", max(a, 1-a), 100*F[gsplit & (Y==1), 3].mean(), 100*F[gsplit & (Y==0), 3].mean()))
# 组合
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(F[gsplit, :4])
clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[gsplit, :4]), Y[gsplit])
a = roc_auc_score(Y[~gsplit], clf.predict_proba(sc.transform(F[~gsplit, :4]))[:, 1])
P("四特征组合 AUC=%.3f" % max(a, 1-a))
P("FOUNDRY32_DONE %.0fs" % (time.time() - t0))
LOG.close()
