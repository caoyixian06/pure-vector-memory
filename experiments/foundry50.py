# -*- coding: utf-8 -*-
"""foundry50.py — 金金独特性压缩池: 池595 → 团凝聚压缩 → 全金保持率
方法: 池内GBDT高分锚片团凝聚(θ=0.60双空间), 簇代表(簇内GBDT最高分)出线,
      压缩池 = 各簇代表 + 独立簇(未合并者) + FINAL/C0池头50保底
靶: 全金在压缩池率 vs 原池89.7%, 压缩比。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry50_results.txt"
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

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

POOLSZ = 595
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
import torch
Dt = torch.from_numpy(D).to("cuda")
Qt = torch.from_numpy(QW).to("cuda")
THS = (0.60, 0.68, 0.75)
res = {}
for th in THS:
    res[("θ=%.2f" % th)] = {"pool": 0, "sizes": []}
res["uncompressed"] = {"pool": 0, "sizes": []}
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
        pool = POOLA[k_i]
        pos_of = {c: i for i, c in enumerate(pool)}
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)]
        if G <= set(pool):
            res["uncompressed"]["pool"] += 1
        res["uncompressed"]["sizes"].append(len(pool))
        pool_t = torch.tensor(pool, dtype=torch.long, device="cuda")
        poolD = Dt[pool_t]
        poolQ = Qt[pool_t]
        for th in THS:
            # 团凝聚: 从GBDT第1名开始, 相似>th并入簇, 簇代表=簇内GBDT最高分; 出线集合
            assigned = set()
            reps = []
            for c in o:
                if c in assigned:
                    continue
                reps.append(c)
                # 吸收相似成员
                ci = pos_of[c]
                simsD = poolD @ poolD[ci]
                simsQ = poolQ @ poolQ[ci]
                sims = 0.5 * simsD + 0.5 * simsQ
                close = (sims > th).cpu().numpy()
                for i2, c2 in enumerate(pool):
                    if close[i2] and c2 not in assigned:
                        assigned.add(c2)
                assigned.add(c)
                if len(reps) > 120:
                    break
            # 压缩池 = 簇代表 + 每簇第2名(保底兄弟) + FINAL/C0头50
            comp = set(reps)
            for c in reps:
                ci = pos_of[c]
                simsD = poolD @ poolD[ci]
                simsQ = poolQ @ poolQ[ci]
                sims = (0.5 * simsD + 0.5 * simsQ).cpu().numpy()
                close = [pool[i2] for i2 in np.argsort(-sims)[:3] if pool[i2] != c]
                comp.update(close[:2])
            comp |= set(o[:50])
            res["θ=%.2f" % th]["sizes"].append(len(comp))
            if G <= comp:
                res["θ=%.2f" % th]["pool"] += 1
    P("  fold %s done" % hold)

P("\n===== 金金独特性压缩池 (n=%d) =====" % n)
P("未压缩: 全金=%.1f%% 池均=%.0f" % (
    100.0 * res["uncompressed"]["pool"] / max(1, n),
    np.mean(res["uncompressed"]["sizes"]) if res["uncompressed"]["sizes"] else 0))
for th in THS:
    r = res["θ=%.2f" % th]
    avg = np.mean(r["sizes"]) if r["sizes"] else 0
    P("θ=%.2f: 全金=%.1f%% 池均=%.0f 压缩比=%.1f%%" % (
        th, 100.0 * r["pool"] / max(1, n), avg,
        100.0 * avg / max(1, np.mean(res["uncompressed"]["sizes"]) if res["uncompressed"]["sizes"] else 1)))
P("FOUNDRY50_DONE %.0fs" % (time.time() - t0))
LOG.close()
