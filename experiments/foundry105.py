# -*- coding: utf-8 -*-
"""foundry105.py — LoCoMo 324维全序数LOCO(单模式假说另一半, 干净独立版)
对照: 12维序数=3.1全金@5 | 328维绝对值(v11)~69全金@5/80.5全金@15"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s, flush=True)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
QW = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
Q256 = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]
CONV_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(CONV_of.values()))
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
P("loaded %.0fs" % (time.time() - t0))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

def pool_rank(Fm):
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    for j in range(Fm.shape[1]):
        out[np.argsort(Fm[:, j], kind="stable"), j] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return out

POOL, FEAT, GS = {}, {}, {}
for k_i, qa in enumerate(IDS):
    c0 = D @ X[k_i]
    pool = set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-c0)[:250])
    for i in list(np.argsort(-FINAL[k_i])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(MID) and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:700]
    qtext = Q[qa]["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    qlen = max(1, len(qtok))
    Fm = np.zeros((len(pool), 324), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAWS[c].rstrip().endswith("?") else 0.0
        lr = len(RAWS[c].split()) / max(1, len(qtext.split()))
        vqc = float(QW[c] @ Q256[k_i])
        dqc = float(D[c] @ X[k_i])
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAWS[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, lr, vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]
        Fm[rr, 12:112] = D[c][:100]
        Fm[rr, 112:212] = np.abs(X[k_i][:100] - D[c][:100])
        Fm[rr, 212:312] = QW[c][:100]
        Fm[rr, 312:324] = [qmark, 0.0, min(lr, 4) / 4, 1.0 if dqc > 0.65 else 0.0,
                           dqc - g1, dqc * (1.0 - g1), vqc - dqc, qcov - dqc,
                           dqc - qcov, ccov, len(RAWS[c]) / 2000.0, wvotes / 10.0]
    POOL[k_i] = pool
    FEAT[k_i] = pool_rank(Fm)
    GS[k_i] = gold_set(qa)
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("feats done %.0fs" % (time.time() - t0))

import lightgbm as lgb
res = {"a5": 0, "a15": 0, "any5": 0}
nG = 0
for hold in FOLDS:
    trF, trY, trG = [], [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOL[k_i]
        G = GS[k_i] & set(pool)
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        gset = set(gold_rows)
        noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
        for rr in gold_rows + noise:
            trF.append(FEAT[k_i][rr])
            trY.append(1 if rr in gset else 0)
        trG.append(len(gold_rows) + len(noise))
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                        num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                        random_state=0, verbosity=-1, n_jobs=4)
    rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GS[k_i]
        if not G:
            continue
        nG += 1
        pool = POOL[k_i]
        s = rk.predict(FEAT[k_i])
        seq = [pool[i] for i in np.argsort(-s)]
        if all(g in set(seq[:15]) for g in G):
            res["a15"] += 1
        if all(g in set(seq[:5]) for g in G):
            res["a5"] += 1
        if G & set(seq[:5]):
            res["any5"] += 1
    P("  fold %s %.0fs" % (hold[-12:], time.time() - t0))

P("===== LoCoMo 324-dim full-ordinal LOCO =====")
P("allgold@5=%.1f%%  allgold@15=%.1f%%  any@5=%.1f%%" % (
    100.0 * res["a5"] / nG, 100.0 * res["a15"] / nG, 100.0 * res["any5"] / nG))
P("ref: 12dim-ordinal allgold@5=3.1 | 328dim-absolute(v11) allgold@5~69 allgold@15=80.5")
P("F105_DONE %.0fs" % (time.time() - t0))
