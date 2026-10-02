# -*- coding: utf-8 -*-
"""foundry13.py — 九宫格规律搜索: 题/金/噪 × 文本/256/1024
每题: 金证据记录 + 真实竞争噪声(top50非金采样) → 逐对特征
A. 单特征判别榜(哪个形态哪一格带信息)
B. 跨形态对比特征(换说法指纹/回声指纹)
C. held-out组合AUC(逻辑回归, 题级拆分)
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry13_results.txt"
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
Q256 = np.zeros((len(IDS), 256), dtype=np.float32)
# 题256: 用mean of token vecs? 题向量256需嵌入 -> 用QW空间的替代: 无题256缓存, 改用词表均值? 不可靠。
# 方案: 题256用 ollama? 零API约束内ollama本地可用 -> 嵌入缓存
CACHE = HERE + "/q256_cache.npz"
if os.path.exists(CACHE):
    Q256 = np.load(CACHE)["Q"]
else:
    import urllib.request
    def oemb(texts):
        out = []
        for s in range(0, len(texts), 64):
            for att in range(4):
                try:
                    body = json.dumps({"model": "qwen3-embedding:latest", "input": texts[s:s + 64],
                                       "dimensions": 256}).encode()
                    req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                    break
                except Exception as e:
                    print("retry", repr(e)[:50], flush=True)
                    time.sleep(3 * (att + 1))
            else:
                raise RuntimeError("oemb fail")
        return np.concatenate(out)
    Q256 = l2n(oemb([Q[i]["question"] for i in IDS]))
    np.savez(CACHE, Q=Q256)
P("embedded+loaded %.0fs" % (time.time() - t0))

def gold_idx_list(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = []
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.append(i)
    return g

import random
rng = random.Random(11)
FEATN = 12
rows, labels, qids_row = [], [], []
FN = ["t_jacc", "t_qcov", "t_ccov", "t_qmark", "t_interr", "t_lenratio",
      "v256_qc", "v256_gap", "d1024_qc", "d1024_gap",
      "x_reword", "x_echo"]
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    glist = gold_idx_list(qa)
    if not glist:
        continue
    gset = set(glist)
    top50 = [i for i in np.argsort(-C0[qi])[:50]]
    noises = [i for i in top50 if i not in gset][:20]
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    first_w = (Q[qa].get("question") or "").strip().lower()
    interr = first_w.split()[0] if first_w else ""
    for lab, cands in ((1, glist), (0, noises)):
        for c in cands:
            ct = toks(RAW[c])
            inter = qtok & ct
            jac = len(inter) / max(1, len(qtok | ct))
            qcov = len(inter) / qlen
            ccov = len(inter) / max(1, len(ct))
            qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
            iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
            lenratio = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
            v_qc = float(Q256[k_i] @ QW[c])
            d_qc = float(D[c] @ qv1024)
            f = [jac, qcov, ccov, qmark, iecho, lenratio,
                 v_qc, v_qc - d_qc, d_qc, d_qc - v_qc,
                 d_qc - qcov, qcov - d_qc]
            rows.append(f)
            labels.append(lab)
            qids_row.append(qa)
    if k_i % 300 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
F = np.array(rows, dtype=np.float32)
Y = np.array(labels, dtype=np.int8)
QA = np.array(qids_row)
P("rows=%d gold=%d noise=%d" % (len(Y), int(Y.sum()), int(len(Y) - Y.sum())))

# ===== A. 单特征判别榜(题级拆分held-out) =====
from sklearn.metrics import roc_auc_score
gsplit = np.array([int(hashlib.md5((qa + "f13").encode()).hexdigest(), 16) % 2 == 0 for qa in QA])
P("\n===== A. 单特征判别榜(held-out AUC, 金=1 噪声=0) =====")
aucs = []
for j, nm in enumerate(FN):
    try:
        a = roc_auc_score(Y[~gsplit], F[~gsplit, j])
    except Exception:
        a = 0.5
    aucs.append((max(a, 1 - a), nm, a))
aucs.sort(reverse=True)
for a, nm, raw in aucs:
    P("  %-10s AUC=%.3f" % (nm, a))

# ===== B+C. 组合(逻辑回归 held-out) =====
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Ftr, Ytr = F[gsplit], Y[gsplit]
Fte, Yte = F[~gsplit], Y[~gsplit]
sc = StandardScaler().fit(Ftr)
clf = LogisticRegression(max_iter=2000).fit(sc.transform(Ftr), Ytr)
pte = clf.predict_proba(sc.transform(Fte))[:, 1]
auc_all = roc_auc_score(Yte, pte)
P("\n===== C. 12特征组合 held-out AUC=%.3f =====" % auc_all)
coefs = sorted(zip(FN, clf.coef_[0]), key=lambda x: -abs(x[1]))
P("系数榜: " + ", ".join("%s=%+.2f" % (n, c) for n, c in coefs[:8]))

# 均值画像
P("\n===== 画像(均值) =====")
for j, nm in enumerate(FN):
    P("  %-10s 金=%.3f 噪声=%.3f" % (nm, F[gsplit & (Y == 1), j].mean(), F[gsplit & (Y == 0), j].mean()))
P("FOUNDRY13_DONE %.0fs" % (time.time() - t0))
LOG.close()
