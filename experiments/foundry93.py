# -*- coding: utf-8 -*-
"""foundry93.py — 打开ranker黑盒: 隐式规律的显式化
对象=直迁弹药(clean池+328维无FINAL+全量训练)
输出: gain榜/SHAP榜/金噪SHAP方向/top交互解读"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry93_results.txt"
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
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

NAMES = ["jac", "qcov", "ccov", "qmark", "dead4", "lr", "vqc", "vqc-dqc", "dqc", "dqc-vqc", "dqc-qcov", "qcov-dqc"]
NAMES += ["recBGE_d%d" % i for i in range(100)]
NAMES += ["absdiff_d%d" % i for i in range(100)]
NAMES += ["rec256_w%d" % i for i in range(100)]
NAMES += ["qmark_dup", "dead13", "lr4", "cospen", "gradinv", "cross", "nb_flag", "r1", "r2", "wvotes", "dead22", "dead23"]
NAMES += ["pad%d" % i for i in range(328 - len(NAMES))]
assert len(NAMES) >= 328

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
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
X = np.load(HERE + "/xz_cache.npz")["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QWQ256 = l2n(Q256 @ QW.T)
NALL = len(IDS)
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]

def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR = {}
seen = {}
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR[i] = seen[k]; PAIR[seen[k]] = i
    else:
        seen[k] = i

def gold_groups(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(NR) if k in RAWN[i])
        if not hits:
            continue
        for i in list(hits):
            tt = PAIR.get(i)
            if tt is not None:
                hits.add(tt)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups
GSETS = [gold_groups(qa) for qa in IDS]
P("loaded %.0fs" % (time.time() - t0))

POOLSZ = 700
POOLA = [None] * NALL
FEATS = [None] * NALL
for k_i in range(NALL):
    qi = k_i
    q = Q[IDS[k_i]]
    pool = set(np.argsort(-C0[qi])[:350]) | set(np.argsort(-QWQ256[k_i])[:350])
    top30 = list(np.argsort(-C0[qi])[:30])
    for i in top30:
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:POOLSZ]
    POOLA[k_i] = pool
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(q["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    Fm = np.zeros((len(pool), 328), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAW[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        lr = len(RAW[c]) / max(1, len(q.get("question") or "x"))
        vqc = float(qv256 @ QW[c])
        dqc = float(D[c] @ qv1024)
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, 0.0, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:112] = D[c][:100]
        Fm[rr, 112:212] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 212:312] = QW[c][:100]
        Fm[rr, 312:315] = [qmark, 0.0, min(lr, 4) / 4]
        nb_flag = 1.0 if any(abs(c - j) <= 2 and CONVKEY[j] == CONVKEY[c] for j in top30) else 0.0
        rl_ = RAW[c].lower()
        seqn = 0; mx = 0; last_p = -1
        for w in qstems:
            p = rl_.find(w)
            if p >= 0 and p > last_p:
                seqn += 1; last_p = p; mx = max(mx, seqn)
            elif p >= 0:
                seqn = 1; last_p = p
            else:
                seqn = 0
        r1 = mx / max(1, len(qstems))
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAW[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        Fm[rr, 315:323] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1, r2, wvotes, 0.0]
    FEATS[k_i] = Fm
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

import lightgbm as lgb
trF, trY, trG = [], [], []
for k_i, qa in enumerate(IDS):
    pool = POOLA[k_i]
    G = set()
    for grp in GSETS[k_i]:
        G |= grp
    gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
    gset = set(gold_rows)
    noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
    noisepick = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:40]) if noise_rows else []
    rows = gold_rows + noisepick
    for rr in rows:
        trF.append(FEATS[k_i][rr])
        trY.append(1 if rr in gset else 0)
    trG.append(len(rows))
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                        num_leaves=63, min_child_samples=30, lambdarank_truncation_level=5,
                        random_state=0, verbosity=-1, n_jobs=4)
ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
P("ranker trained %.0fs" % (time.time() - t0))

# ===== 1) gain importance =====
gain = ranker.booster_.feature_importance("gain")
order = np.argsort(-gain)
P("\n===== gain top40 =====")
for r in order[:40]:
    P("%3d %-14s gain=%.1f" % (r, NAMES[r] if r < len(NAMES) else str(r), gain[r]))

# ===== 2) SHAP (头部场景: 每题C0序前15) =====
try:
    import shap
    expl = shap.TreeExplainer(ranker.booster_)
    rng = np.random.RandomState(0)
    sample_q = rng.choice(NALL, 150, replace=False)
    Xs, ys = [], []
    for k_i in sample_q:
        pool = POOLA[k_i]
        G = set()
        for grp in GSETS[k_i]:
            G |= grp
        head_order = np.argsort(-C0[k_i][pool])[:15]
        for rr in head_order:
            Xs.append(FEATS[k_i][rr])
            ys.append(1 if pool[rr] in G else 0)
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys)
    sv = expl.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[0]
    P("\n===== SHAP top30 (n=%d, 金=%d) =====" % (len(ys), ys.sum()))
    ms = np.abs(sv).mean(0)
    o2 = np.argsort(-ms)
    for r in o2[:30]:
        g_m = sv[ys == 1, r].mean()
        n_m = sv[ys == 0, r].mean()
        P("%3d %-14s |SHAP|=%.4f  金向=%+.4f 噪向=%+.4f  特征值(金/噪)=%.3f/%.3f" % (
            r, NAMES[r] if r < len(NAMES) else str(r), ms[r], g_m, n_m,
            Xs[ys == 1, r].mean(), Xs[ys == 0, r].mean()))
except Exception as e:
    P("SHAP failed: %s" % str(e)[:200])

# ===== 3) 树路径共现 (粗交互) =====
try:
    dump = ranker.booster_.dump_model()
    co = {}
    for tree in dump["tree_info"]:
        feats = set()
        stack = [tree["tree_structure"]]
        while stack:
            nd = stack.pop()
            if "split_index" in nd:
                feats.add(nd["split_feature"])
                stack.append(nd["left_child"])
                stack.append(nd["right_child"])
        fl = sorted(feats)
        for a2 in range(len(fl)):
            for b2 in range(a2 + 1, len(fl)):
                k2 = (fl[a2], fl[b2])
                co[k2] = co.get(k2, 0) + 1
    top_co = sorted(co.items(), key=lambda x: -x[1])[:15]
    P("\n===== 同树共现top15 (交互代理) =====")
    for (a2, b2), c2 in top_co:
        P("%-14s × %-14s  %d树" % (NAMES[a2], NAMES[b2], c2))
except Exception as e:
    P("co-occurrence failed: %s" % str(e)[:150])
P("F93_DONE %.0fs" % (time.time() - t0))
LOG.close()
