# -*- coding: utf-8 -*-
"""foundry84.py — LGBMRanker(listwise/lambdarank): 训练目标直接对齐排序, 头部专攻
特征=F74同款329维, LOCO按会话十折, 采样=金8+噪40/题(group=题)
对照: pointwise v10(核心@5=76.7/全部@15=78.6)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry84_results.txt"
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
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
CONV_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(CONV_of.values()))
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

def core_group(qa, grps):
    ans = " ".join(str(x) for x in (Q[qa].get("answer") or []))
    aw = set(stem(w) for w in re.findall(r"[a-z]+", ans.lower()) if len(w) > 2)
    if not aw:
        return grps[0]
    best, bv = grps[0], -1.0
    for grp in grps:
        tw = set()
        for g in grp:
            tw |= set(stem(w) for w in re.findall(r"[a-z']+", RAW[g].lower()))
        cov = len(aw & tw) / len(aw)
        if cov > bv:
            bv, best = cov, grp
    return best

P("loaded %.0fs" % (time.time() - t0))

POOLSZ = 700
POOLA = [None] * NALL
FEATS = [None] * NALL
for k_i, qa in enumerate(IDS):
    qi = k_i
    q = Q[qa]
    pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    top30list = list(np.argsort(-FINAL[qi])[:30])
    for i in top30list:
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
    Fm = np.zeros((len(pool), 329), dtype=np.float32)
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
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, 0.0, min(lr, 4) / 4]
        nb_flag = 1.0 if any(abs(c - j) <= 2 and CONVKEY[j] == CONVKEY[c] for j in top30list) else 0.0
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
        Fm[rr, 321:329] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1, r2, wvotes, 0.0]
    FEATS[k_i] = Fm
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

import lightgbm as lgb

def v9_post(seq_idx, qa):
    myconv = "loco-" + qa.split("#")[0]
    out, used = [], set()
    for i in seq_idx:
        if CONVKEY[i] != myconv:
            continue
        p = PAIR.get(i)
        if p is not None and p in used:
            continue
        out.append(i)
        used.add(i)
    return out

NS_NOISE = 40
all15 = core5 = 0
n = 0
for hold in FOLDS:
    trF, trY, trG = [], [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOLA[k_i]
        G = set()
        for grp in GSETS[k_i]:
            G |= grp
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        noisepick = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:NS_NOISE]) if noise_rows else []
        rows = gold_rows + noisepick
        for rr in rows:
            trF.append(FEATS[k_i][rr])
            trY.append(1 if rr in set(gold_rows) else 0)
        trG.append(len(rows))
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.08,
        num_leaves=63, min_child_samples=30, lambdarank_truncation_level=10,
        random_state=0, verbosity=-1, n_jobs=4)
    ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        pool = POOLA[k_i]
        s = ranker.predict(FEATS[k_i], num_iteration=ranker.best_iteration_ or 300)
        seq = [pool[i] for i in np.argsort(-s)]
        post = v9_post(seq, qa)
        if all(any(g in post[:15] for g in grp) for grp in G):
            all15 += 1
        cg = core_group(qa, G)
        if any(g in post[:5] for g in cg):
            core5 += 1
    P("  fold %s done %.0fs" % (hold[-12:], time.time() - t0))

P("\n===== LGBMRanker(lambdarank) n=%d =====" % n)
P("全部@15=%5.1f%%  核心@5=%5.1f%%" % (100.0 * all15 / n, 100.0 * core5 / n))
P("对照 pointwise v10: 全部@15=78.6 核心@5=76.7")
P("F84_DONE %.0fs" % (time.time() - t0))
LOG.close()
