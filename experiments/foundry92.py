# -*- coding: utf-8 -*-
"""foundry92.py — R2修复: ranker版FINAL三臂消融 (直迁前必做)
臂1 v11现状(FINAL列+FINAL池,329维, 已知78.2)
臂2 nofinal_ranker(删FINAL列328维, 池仍fc)
臂3 clean_ranker(C0+256双通道池+无FINAL列, 328维) = 真正的直迁弹药"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry92_results.txt"
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
QWQ256 = l2n(Q256 @ QW.T)
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
POOL_FC = [None] * NALL
POOL_CL = [None] * NALL
for k_i in range(NALL):
    qi = k_i
    pool_fc = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    top30 = list(np.argsort(-FINAL[qi])[:30])
    pool_cl = set(np.argsort(-C0[qi])[:350]) | set(np.argsort(-QWQ256[k_i])[:350])
    top30c = list(np.argsort(-C0[qi])[:30])
    for src, pool, top in ((0, pool_fc, top30), (1, pool_cl, top30c)):
        for i in top:
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                    pool.add(j)
    POOL_FC[k_i] = sorted(pool_fc)[:POOLSZ]
    POOL_CL[k_i] = sorted(pool_cl)[:POOLSZ]
P("pools %.0fs" % (time.time() - t0))

def build_feats(k_i, pool, top30list, use_final):
    qa = IDS[k_i]
    qi = k_i
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(q["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    nd = 329 if use_final else 328
    Fm = np.zeros((len(pool), nd), dtype=np.float32)
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
        Fm[rr, 312] = float(C0[qi][c])
        if use_final:
            Fm[rr, 313] = float(FINAL[qi][c])
        base = 313 if use_final else 312
        Fm[rr, base:base + 3] = [qmark, 0.0, min(lr, 4) / 4]
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
        Fm[rr, base + 3:base + 11] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1, r2, wvotes, 0.0]
    return Fm

FEATS = {}
for k_i in range(NALL):
    FEATS[("fc", True)] = None
for mode, usef, POOL in (("fc", True, POOL_FC), ("nofc", False, POOL_FC), ("clean", False, POOL_CL)):
    arr = [None] * NALL
    tops = {k_i: (list(np.argsort(-FINAL[k_i])[:30]) if mode == "fc" else list(np.argsort(-C0[k_i])[:30])) for k_i in range(NALL)}
    for k_i in range(NALL):
        arr[k_i] = build_feats(k_i, POOL[k_i], tops[k_i], usef)
    FEATS[(mode, usef)] = arr
    P("feats %s %.0fs" % (mode, time.time() - t0))

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

ARMS = [("v11现状(fc池+FINAL列)", "fc", True), ("nofinal_ranker(删FINAL列)", "nofc", False), ("clean_ranker(直迁弹药)", "clean", False)]
RES = {a[0]: [0, 0] for a in ARMS}
for hold in FOLDS:
    for name, mode, usef in ARMS:
        POOL = POOL_FC if mode != "clean" else POOL_CL
        FARR = FEATS[(mode, usef)]
        trF, trY, trG = [], [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool = POOL[k_i]
            G = set()
            for grp in GSETS[k_i]:
                G |= grp
            gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
            gset = set(gold_rows)
            noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
            noisepick = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:40]) if noise_rows else []
            rows = gold_rows + noisepick
            for rr in rows:
                trF.append(FARR[k_i][rr])
                trY.append(1 if rr in gset else 0)
            trG.append(len(rows))
        ranker = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=500, learning_rate=0.08,
            num_leaves=63, min_child_samples=30, lambdarank_truncation_level=5,
            random_state=0, verbosity=-1, n_jobs=4)
        ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] != hold:
                continue
            G = GSETS[k_i]
            if not G:
                continue
            pool = POOL[k_i]
            s = ranker.predict(FARR[k_i])
            seq = [pool[i] for i in np.argsort(-s)]
            post = v9_post(seq, qa)
            if all(any(g in post[:15] for g in grp) for grp in G):
                RES[name][0] += 1
            cg = core_group(qa, G)
            if any(g in post[:5] for g in cg):
                RES[name][1] += 1
    P("  fold %s done %.0fs" % (hold[-12:], time.time() - t0))

nG = sum(1 for g in GSETS if g)
P("===== ranker三臂 n=%d =====" % nG)
for name, _, _ in ARMS:
    P("%-28s all15=%5.1f%%  core5=%5.1f%%" % (name, 100.0 * RES[name][0] / nG, 100.0 * RES[name][1] / nG))
P("F92_DONE %.0fs" % (time.time() - t0))
LOG.close()
