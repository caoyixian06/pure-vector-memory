# -*- coding: utf-8 -*-
"""foundry95.py — 闲置规律的岗位补齐: 时间街区Δ投影+词向量轴投影+凝聚度+E指纹交互 4新特征
基座=v11(329维+fc池+ranker trunc5_est500), 对照v11=80.5-80.8/78.2-78.7"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry95_results.txt"
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

# ===== 岗位1: 时间街区Δ投影(零标签derive: 含时间词vs不含, 与LME同口径) =====
DATE_WORDS = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
is_dated = np.array([bool(DATE_WORDS.search(r)) for r in RAW])
DELTA_T = QW[is_dated].mean(0) - QW[~is_dated].mean(0)
DELTA_T = DELTA_T / max(np.linalg.norm(DELTA_T), 1e-9)
REC_TPROJ = QW @ DELTA_T          # 每记录的时间敏感度分
P("时间街区: dated=%d  Δ投影就绪 %.0fs" % (is_dated.sum(), time.time() - t0))

# ===== 岗位2: 词向量轴投影(月份轴/星期轴, word_vecs查表) =====
WHd = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WHd.keys())
WV = l2n(np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
def axis(words):
    idxs = [W2I[w] for w in words if w in W2I]
    if len(idxs) < 2:
        return None
    v = WV[idxs].mean(0)
    return v / max(np.linalg.norm(v), 1e-9)
MONTH_AX = axis(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])
WEEK_AX = axis(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
def text_axis_proj(text, ax):
    if ax is None:
        return 0.0
    ws = set(stem(w) for w in re.findall(r"[a-z']+", str(text).lower()) if len(w) > 2)
    idxs = [W2I[w] for w in ws if w in W2I]
    if not idxs:
        return 0.0
    return float(WV[idxs] @ ax)
P("词向量轴: month=%s week=%s %.0fs" % (MONTH_AX is not None, WEEK_AX is not None, time.time() - t0))

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
    first = (q.get("question") or "x").strip().lower().split()[0] if (q.get("question") or "").strip() else ""
    e_when = 1.0 if first == "when" else 0.0
    q_tproj = float(qv256 @ DELTA_T)
    q_mproj = text_axis_proj(q.get("question") or "", MONTH_AX)
    pool_tproj = REC_TPROJ[pool]
    pool_toks = [toks(RAW[c]) for c in pool]
    Fm = np.zeros((len(pool), 334), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = pool_toks[rr]
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
        # ===== 岗位1: 时间投影 3列 =====
        Fm[rr, 329] = float(pool_tproj[rr])
        Fm[rr, 330] = float(pool_tproj[rr] * q_tproj)
        Fm[rr, 331] = float(pool_tproj[rr] * e_when)
        # ===== 岗位2: 词向量轴投影 =====
        c_mproj = text_axis_proj(RAW[c], MONTH_AX)
        Fm[rr, 332] = float(c_mproj * q_mproj)
        # ===== 岗位3: 凝聚度(与池内其他候选的平均Jaccard, 头部近似=与top10的平均) =====
        Fm[rr, 333] = float(np.mean([len(ct & pool_toks[j]) / max(1, len(ct | pool_toks[j])) for j in range(min(10, len(pool))) if j != rr]))
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

CONFIGS = [("v15_334dim", 5, 500, 40)]
RES = {c[0]: [0, 0] for c in CONFIGS}
for hold in FOLDS:
    for name, trunc, nest, nnoise in CONFIGS:
        trF, trY, trG = [], [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool = POOLA[k_i]
            G = set()
            for grp in GSETS[k_i]:
                G |= grp
            gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
            gset = set(gold_rows)
            noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
            noisepick = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:nnoise]) if noise_rows else []
            rows = gold_rows + noisepick
            for rr in rows:
                trF.append(FEATS[k_i][rr])
                trY.append(1 if rr in gset else 0)
            trG.append(len(rows))
        ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=nest, learning_rate=0.08,
                                num_leaves=63, min_child_samples=30, lambdarank_truncation_level=trunc,
                                random_state=0, verbosity=-1, n_jobs=4)
        ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] != hold:
                continue
            G = GSETS[k_i]
            if not G:
                continue
            pool = POOLA[k_i]
            s = ranker.predict(FEATS[k_i])
            seq = [pool[i] for i in np.argsort(-s)]
            post = v9_post(seq, qa)
            if all(any(g in post[:15] for g in grp) for grp in G):
                RES[name][0] += 1
            cg = core_group(qa, G)
            if any(g in post[:5] for g in cg):
                RES[name][1] += 1
    P("  fold %s done %.0fs" % (hold[-12:], time.time() - t0))

nG = sum(1 for g in GSETS if g)
P("===== v15岗位补齐 n=%d =====" % nG)
for name, _, _, _ in CONFIGS:
    P("%-14s all15=%5.1f%%  core5=%5.1f%%" % (name, 100.0 * RES[name][0] / nG, 100.0 * RES[name][1] / nG))
P("v11对照: all15=80.5-80.8 core5=78.2-78.7")
P("F95_DONE %.0fs" % (time.time() - t0))
LOG.close()
