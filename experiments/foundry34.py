# -*- coding: utf-8 -*-
"""foundry34.py — 三团对比: 金金抱团 vs 噪噪抱团 vs 金噪抱团
如果三组可分 → 抱团结构本身携带金噪信息, 以片找片引擎有理论根基;
如果噪噪≈金金 → 噪声也抱团且无法区分, 以片找片必须靠问题门救。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry34_results.txt"
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
    if len(noises) < 2:
        continue
    def add(a, b, lab):
        ta, tb = ctoks(a), ctoks(b)
        jac = len(ta & tb) / max(1, len(ta | tb))
        v = float(QW[a] @ QW[b])
        d = float(D[a] @ D[b])
        rows.append((jac, v, d, lab))
        qa_of.append(k_i)
    gg = [(a, b) for ai, a in enumerate(golds) for b in golds[ai+1:]][:6]
    for a, b in gg:
        add(a, b, 0)
    nn = [(noises[i], noises[i+1]) for i in range(0, min(len(noises)-1, 12), 2)][:6]
    for a, b in nn:
        add(a, b, 1)
    gn = []
    for _ in range(12):
        a = golds[rng.randrange(len(golds))]
        b = noises[rng.randrange(len(noises))]
        if a != b:
            gn.append((a, b))
    for a, b in gn[:6]:
        add(a, b, 2)
    if k_i % 200 == 0:
        P("  pairs %d %.0fs" % (k_i, time.time() - t0))
F = np.array(rows, dtype=np.float64)
LAB = F[:, 3].astype(int)
P("金金对=%d 噪噪对=%d 金噪对=%d" % (
    int((LAB == 0).sum()), int((LAB == 1).sum()), int((LAB == 2).sum())))

P("\n===== 三团相似度对比 =====")
names = ("金金", "噪噪", "金噪")
for j, nm in enumerate(("文本Jaccard", "256cos", "1024cos")):
    P("%-10s: " % nm + "  ".join("%s=%.3f" % (names[l], F[LAB == l, j].mean()) for l in range(3)))
from sklearn.metrics import roc_auc_score
P("\n判别AUC(去同会话因素前):")
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    m01 = np.isin(LAB, [0, 1])
    m02 = np.isin(LAB, [0, 2])
    a01 = roc_auc_score(LAB[m01] == 0, F[m01, j])
    a02 = roc_auc_score(LAB[m02] == 0, F[m02, j])
    P("  %-9s 金金vs噪噪=%.3f  金金vs金噪=%.3f" % (nm, max(a01, 1-a01), max(a02, 1-a02)))
P("FOUNDRY34_DONE %.0fs" % (time.time() - t0))
LOG.close()
