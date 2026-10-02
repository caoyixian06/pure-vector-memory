# -*- coding: utf-8 -*-
"""foundry58.py — 256维深挖: 用户判断"256里有解"
对4对案例+全量统计:
  A. 逐维对比: 真金与问题的256维乘积(x*r) vs 假金的 — 找出方向性区分维
  B. 分维度AUC: 全部256维单独测"真金vs假金"的判别AUC(头部场景) — 256维排行榜
  C. 时间街区(d245-255)与全维的贡献分解
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry58_results.txt"
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

# ===== A. 4对案例的逐维对比 =====
def find_rec(substr):
    ns = norm(substr)
    for i, rn in enumerate(RAWN):
        if ns[:50] in rn:
            return i
    return None

CASES = [
    ("conv-26#q0074", "5 years already", "Hey Mel! Good to see you! How have you been"),
    ("conv-26#q0119", "Yeah, I drew it. It stands for", "Drawing flowers is one of my fav"),
    ("conv-26#q0103", "It was Matt Patterson", "That's really sweet. Is this your"),
    ("conv-26#q0104", "I'm obsessed with those, so I made", "Good to see you! I'm swamped with the"),
]
P("===== A. 案例逐维分析 =====")
for qa, gtxt, ftxt in CASES:
    qi = IDX[qa]
    g = find_rec(gtxt)
    f = find_rec(ftxt)
    if g is None or f is None:
        P("[%s] 定位失败" % qa)
        continue
    qv = Q256[qi]
    prodg = qv * QW[g]
    prodf = qv * QW[f]
    # 同号且大的维度 = 共振维; 真金共振>假金共振的维度数
    res_g = prodg - prodf
    n_pos = int((res_g > 0.005).sum())
    n_neg = int((res_g < -0.005).sum())
    top_dims = np.argsort(-res_g)[:5]
    bot_dims = np.argsort(res_g)[:5]
    P("[%s] %s" % (qa, Q[qa]["question"][:60]))
    P("  真金占优维: %d | 假金占优维: %d | 平: %d" % (n_pos, n_neg, 256 - n_pos - n_neg))
    P("  真金最强5维: %s" % ", ".join("d%d(+%.3f)" % (d, res_g[d]) for d in top_dims))
    P("  假金最强5维: %s" % ", ".join("d%d(%.3f)" % (d, res_g[d]) for d in bot_dims))

# ===== B. 全量: 每个维度的"交互积x*r"对真金vs假金的AUC =====
from sklearn.metrics import roc_auc_score
import hashlib
P("\n===== B. 256维交互积(x*r)AUC排行 =====")
rows_g = []   # (qa, rec) gold in head30
rows_f = []
labels = []
qids_col = []
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
        continue
    order = np.argsort(-FINAL[qi])
    head = list(order[:30])
    for c in head:
        if c in G:
            rows_g.append((k_i, c))
            labels.append(1)
            qids_col.append(k_i)
        else:
            rows_f.append((k_i, c))
            labels.append(0)
            qids_col.append(k_i)
    if len(rows_g) + len(rows_f) > 60000:
        break
PROD = np.zeros((len(labels), 256), dtype=np.float32)
for r, (k_i, c) in enumerate(list(rows_g) + list(rows_f)):
    PROD[r] = Q256[k_i] * QW[c]
LA = np.array(labels, dtype=np.int8)
QA_i = np.array(qids_col)
gsplit = np.array([int(hashlib.md5(str(q).encode()).hexdigest(), 16) % 2 == 0 for q in QA_i])
aucs = []
for d in range(256):
    try:
        a = roc_auc_score(LA[~gsplit], PROD[~gsplit, d])
    except Exception:
        a = 0.5
    aucs.append((max(a, 1 - a), d, a))
aucs.sort(reverse=True)
P("样本=%d 金=%d" % (len(LA), int(LA.sum())))
P("Top15维:")
for a, d, raw_a in aucs[:15]:
    P("  d%-4d AUC=%.3f (方向:%s)" % (d, a, "+" if raw_a >= 0.5 else "-"))
P("AUC>0.55的维数: %d/256" % sum(1 for a, _, _ in aucs if a > 0.55))
P("AUC>0.60的维数: %d/256" % sum(1 for a, _, _ in aucs if a > 0.60))
# 时间街区贡献
tb = [a for a, d, _ in aucs if 245 <= d <= 255]
P("时间街区(d245-255)平均AUC: %.3f" % np.mean([a for a, _, _ in aucs if any(dd in (d,) for dd in range(245, 256))][:11]))
# C. 组合: topK维交互积求和
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
for K in (10, 30, 60, 120):
    dims = [d for _, d, _ in aucs[:K]]
    F = PROD[:, dims]
    sc = StandardScaler().fit(F[gsplit])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(F[gsplit]), LA[gsplit])
    a = roc_auc_score(LA[~gsplit], lr.predict_proba(sc.transform(F[~gsplit]))[:, 1])
    P("Top%d维组合 AUC=%.3f" % (K, max(a, 1 - a)))
P("F58_DONE")
LOG.close()
