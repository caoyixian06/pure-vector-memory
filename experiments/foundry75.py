# -*- coding: utf-8 -*-
"""foundry75.py — 闲置资产全接入: 256维后156维(时间街区在d151-256)+E指纹+反回声+双通道rank
臂: base(329, 应复现69.4) / wide(651维: 329+D[100:256]+QW[100:256]+E3+反回声5+rank2)
训练: GBDT随机负例(F74 base配方) / 计分: v9后处理, 折叠金口径@3/@5
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry75_results.txt"
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
XSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your really cool wow thanks hey yeah okay ok just been being get got getting lot bit kind sort pretty much many some all also too very there their here have has had having not no yes but so as by be am will would can could should may might dont didnt".split())

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
P("loaded %.0fs" % (time.time() - t0))

SPK = {}
for i in range(NR):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    if mm:
        SPK.setdefault(CONVKEY[i], set()).add(mm.group(1))
def speaker_of(i):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    return mm.group(1) if mm else None

POOLSZ = 700
NWIDE = 329 + 156 + 156 + 3 + 5 + 2   # 651
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
    qw_text = q.get("question") or "x"
    first = qw_text.strip().lower().split()[0] if qw_text.strip() else ""
    e_which = 1.0 if first == "which" else 0.0
    e_when = 1.0 if first == "when" else 0.0
    e_what = 1.0 if first == "what" else 0.0
    qtext = qw_text.lower()
    conv_speakers = SPK.get(CONVKEY[pool[0]] if pool else "x", set())
    q_spk = set(s for s in conv_speakers if s and s.lower() in qtext)
    uniq_spk = (list(q_spk)[0] if len(q_spk) == 1 else None)
    # 双通道rank
    rank1024 = np.empty(len(pool), dtype=np.float32)
    rank256 = np.empty(len(pool), dtype=np.float32)
    o1 = np.argsort(-C0[qi][pool])
    o2 = np.argsort(-QWQ256[k_i][pool])
    r1_ = np.empty(len(pool)); r2_ = np.empty(len(pool))
    for pos, rr in enumerate(o1):
        r1_[rr] = pos / len(pool)
    for pos, rr in enumerate(o2):
        r2_[rr] = pos / len(pool)
    Fm = np.zeros((len(pool), NWIDE), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAW[c])
        cts = set(stem(w) for w in ct)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        lr = len(RAW[c]) / max(1, len(qw_text))
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
        r1f = mx / max(1, len(qstems))
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAW[c].lower()))
        r2f = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        Fm[rr, 321:329] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1f, r2f, wvotes, 0.0]
        # ===== 新接入: 256后半段(时间街区所在) =====
        Fm[rr, 329:485] = D[c][100:256]
        Fm[rr, 485:641] = QW[c][100:256]
        # E指纹
        Fm[rr, 641:644] = [e_which, e_when, e_what]
        # 反回声
        new_ratio = len((cts - set(stem(w) for w in qtok)) - XSTOP) / max(1, len(cts))
        echo = len(ct & qtok) / max(1, len(ct))
        s = speaker_of(c)
        spk_mis = 1.0 if (uniq_spk is not None and s is not None and s != uniq_spk) else 0.0
        Fm[rr, 644:649] = [new_ratio, echo, new_ratio - echo, spk_mis, 0.0]
        # 双通道rank
        Fm[rr, 649:651] = [r1_[rr], r2_[rr]]
    FEATS[k_i] = Fm
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier

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

ARMS = [("base329", 329), ("wide651", NWIDE)]
RES = {a[0]: {"a5": 0, "a3": 0} for a in ARMS}
n = 0
for hold in FOLDS:
    for name, ndim in ARMS:
        trF, trY = [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool = POOLA[k_i]
            G = set()
            for grp in GSETS[k_i]:
                G |= grp
            gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
            noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
            noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
            for rr in gold_rows:
                trF.append(FEATS[k_i][rr][:ndim]); trY.append(1)
            for rr in noisepick:
                trF.append(FEATS[k_i][rr][:ndim]); trY.append(0)
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
            s = clf.decision_function(FEATS[k_i][:, :ndim])
            seq = [pool[i] for i in np.argsort(-s)]
            post = v9_post(seq, qa)
            if all(any(g in post[:5] for g in grp) for grp in G):
                RES[name]["a5"] += 1
            if all(any(g in post[:3] for g in grp) for grp in G):
                RES[name]["a3"] += 1
        P("  [%s] fold %s done %.0fs" % (name, hold[-12:], time.time() - t0))

P("\n===== 结果 (n=%d) =====" % n)
for name, _ in ARMS:
    P("%-10s @3=%5.1f%%  @5=%5.1f%%" % (name, 100.0 * RES[name]["a3"] / max(1, n), 100.0 * RES[name]["a5"] / max(1, n)))
P("F75_DONE %.0fs" % (time.time() - t0))
LOG.close()
