# -*- coding: utf-8 -*-
"""foundry35.py — 三角关系第二步: 问题-金金 / 问题-金噪 / 问题-噪噪
对每片记录算与问题的相似度(1024cos/256cos/Jaccard), 按片团归属分三组比较:
①金片与问题 ②噪声与问题 ③以及(跨题)金片对问题 vs 噪声对问题
判据: 问题视角下三团是否可分(AUC)。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry35_results.txt"
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
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
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

import random
rng = random.Random(5)
rows = []
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
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv1024 = X[qi]
    qv256 = Q256[k_i]
    def add(i, lab):
        ct = ctoks(i)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        v = float(qv256 @ QW[i])
        d = float(D[i] @ qv1024)
        rows.append((jac, v, d, lab))
        qa_of.append(k_i)
    # ①问题-金片(全部, 上限6)
    for i in golds[:6]:
        add(i, 0)
    # ②问题-噪声(上限6)
    for i in noises[:6]:
        add(i, 1)
    if k_i % 200 == 0:
        P("  scan %d %.0fs" % (k_i, time.time() - t0))
F = np.array(rows, dtype=np.float64)
LAB = F[:, 3].astype(int)
P("问题-金片=%d 问题-噪声=%d" % (int((LAB == 0).sum()), int((LAB == 1).sum())))

P("\n===== 三角关系: 问题视角下的两团 =====")
names = ("问题-金片", "问题-噪声")
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    P("%-8s: %s=%.3f  %s=%.3f" % (nm, names[0], F[LAB == 0, j].mean(), names[1], F[LAB == 1, j].mean()))
from sklearn.metrics import roc_auc_score
P("\n判别AUC:")
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    a = roc_auc_score(LAB, F[:, j])
    P("  %-8s AUC=%.3f" % (nm, max(a, 1 - a)))

# 混入"团内相似度"视角: 把F34的三团数据与本题的问题相似度做交叉表
P("\n交叉解读(问题相似度 x 团内相似度):")
P("  金片: 团内Jaccard=0.563(高) + 问题Jac=?", )
P("FOUNDRY35_DONE %.0fs" % (time.time() - t0))
LOG.close()
