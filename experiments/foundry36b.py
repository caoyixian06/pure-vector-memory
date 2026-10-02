# -*- coding: utf-8 -*-
"""foundry36.py — 全库验证: 问题-金/问题-噪 弱区分是否普遍 + 弱信号叠加价值
全1374题, 全部金片vs池外噪声对照; 1024反向规律复核; 问题门控×片团信号叠加实测。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry36_results.txt"
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
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

# ===== 全库收集: 问题-金 / 问题-噪(池150内噪声) =====
import random
rng = random.Random(5)
rows = []
qa_of = []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
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
    for i in golds:
        add(i, 0)
    for i in noises[:4]:
        add(i, 1)
    if k_i % 200 == 0:
        P("  scan %d %.0fs" % (k_i, time.time() - t0))
F = np.array(rows, dtype=np.float64)
LAB = F[:, 3].astype(int)
P("问题-金=%d 问题-噪=%d" % (int((LAB == 0).sum()), int((LAB == 1).sum())))

from sklearn.metrics import roc_auc_score
P("\n===== 全库三角验证 =====")
names = ("问题-金", "问题-噪")
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    P("%-8s: %s=%.3f  %s=%.3f  AUC=%.3f" % (
        nm, names[0], F[LAB == 0, j].mean(), names[1], F[LAB == 1, j].mean(),
        max(roc_auc_score(LAB, F[:, j]), 1 - roc_auc_score(LAB, F[:, j]))))

# ===== 分类别复核(1024反向是否普遍) =====
P("\n分题型1024反向复核:")
CATS = {qa: Q[qa].get("category") for qa in IDS}
for cat in sorted(set(CATS.values())):
    mask = np.array([CATS[IDS[qa_of[i]]] == cat for i in range(len(LAB))])
    if mask.sum() < 100:
        continue
    a = roc_auc_score(LAB[mask], F[mask, 2])
    P("  cat%s: 问题-金=%.3f 问题-噪=%.3f AUC(金=1)=%.3f" % (
        cat, F[mask & (LAB == 0), 2].mean(), F[mask & (LAB == 1), 2].mean(), a))

# ===== 弱信号叠加: 片团信号 + 问题信号 =====
P("\n===== 弱信号叠加: 片团Jaccard × 问题门 =====")
# 同题内: 已知一片金s, 候选c的入座分 = 片团Jaccard(s,c) + w·qcos(c)
import collections
def sim_jac(a, b):
    ta, tb = ctoks(a), ctoks(b)
    return len(ta & tb) / max(1, len(ta | tb))
res = {"片团单飞": [0, 0], "叠加w=0.5": [0, 0], "叠加w=1.0": [0, 0]}
nmulti = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if len(G) < 2:
        continue
    nmulti += 1
    qi = IDX[qa]
    pool = [i for i in np.argsort(-C0[qi])[:150]]
    golds = sorted(G)
    anchor = golds[0]
    qtok = toks(Q[qa]["question"])
    qv1024 = X[qi]
    others = [g for g in golds[1:]]
    noise = [i for i in pool if i not in G][:60]
    cands = [c for c in pool if c != anchor]
    # 两个信号
    jac_sig = {c: sim_jac(anchor, c) for c in cands}
    q_sig = {c: float(qv1024 @ D[c]) for c in cands}
    for w, nm in ((0.0, "片团单飞"), (0.5, "叠加w=0.5"), (1.0, "叠加w=1.0")):
        sc = {c: jac_sig[c] + w * q_sig[c] for c in cands}
        o = sorted(cands, key=sc.get, reverse=True)[:15]
        got_all = all(g in o for g in others) and all(g in o[:15] for g in others)
        ok = all(g in o for g in others)
        if ok:
            res[nm][0] += 1
        # any-brother in top15
        if any(g in o for g in others):
            res[nm][1] += 1
P("多片题n=%d (锚外兄弟片全部进top15):" % nmulti)
for nm in res:
    P("  %-10s %.1f%%" % (nm, 100.0 * res[nm][0] / max(1, nmulti)))
# ===== 三团全量复核(F34推广): 全题+分题型 =====
P("== 三团全量复核 ==")
import collections
tri = collections.defaultdict(list)
cat_tri = collections.defaultdict(lambda: collections.defaultdict(list))
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if len(G) < 2:
        continue
    qi = IDX[qa]
    pool = [i for i in np.argsort(-C0[qi])[:300]]
    golds = sorted(G)
    noises = [i for i in pool if i not in G]
    if len(noises) < 2:
        continue
    cat = Q[qa].get("category")
    gg = [(a, b) for ai, a in enumerate(golds) for b in golds[ai+1:]][:6]
    nn = [(noises[i], noises[i+2]) for i in range(0, min(len(noises)-2, 18), 3)][:6]
    gn = [(golds[i % len(golds)], noises[(i*3) % len(noises)]) for i in range(6)]
    for a, b, lab in ([(a, b, 0) for a, b in gg] + [(a, b, 1) for a, b in nn] + [(a, b, 2) for a, b in gn]):
        ta, tb = ctoks(a), ctoks(b)
        jac = len(ta & tb) / max(1, len(ta | tb))
        v = float(QW[a] @ QW[b])
        d = float(D[a] @ D[b])
        tri[lab].append((jac, v, d))
        cat_tri[cat][lab].append((jac, v, d))
def stat(pairs, j):
    return np.mean([p[j] for p in pairs]) if pairs else float("nan")
P("%-8s %18s %18s %18s" % ("形态", "金金", "噪噪", "金噪"))
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    P("%-8s %14.3f %16.3f %16.3f" % (nm, stat(tri[0], j), stat(tri[1], j), stat(tri[2], j)))
from sklearn.metrics import roc_auc_score
allp = tri[0] + tri[1]
y01 = [0]*len(tri[0]) + [1]*len(tri[1])
for j, nm in enumerate(("Jaccard", "256cos", "1024cos")):
    vals = [p[j] for p in allp]
    a = roc_auc_score(y01, vals)
    P("  %s 金金vs噪噪 AUC=%.3f" % (nm, max(a, 1-a)))
P("== 分题型 金金vs噪噪 Jaccard ==")
for cat in sorted(cat_tri):
    gg_j = stat(cat_tri[cat][0], 0)
    nn_j = stat(cat_tri[cat][1], 0)
    if gg_j == gg_j and nn_j == nn_j:
        P("  cat%s: 金金=%.3f 噪噪=%.3f 比=%.1f倍" % (cat, gg_j, nn_j, gg_j / max(1e-9, nn_j)))

P("FOUNDRY36_DONE %.0fs" % (time.time() - t0))
LOG.close()
