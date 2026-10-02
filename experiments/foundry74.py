# -*- coding: utf-8 -*-
"""foundry74.py — 难负例自举GBDT: 头部判别专训 + 反回声/说话人新特征, 目标@5
管线: F70特征334维(+5新) → fold内两阶段(GBDT-1随机负例→初排→头部难负例→GBDT-2)
     → fused → v9后处理(分域+rbak去重) → all-gold@5
臂: base(329旧特征+GBDT-1, 应≈68.3) / hard(329+GBDT-2) / hardfeat(334+GBDT-2)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry74_results.txt"
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
MID2I = {m: i for i, m in enumerate(MID)}
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
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1

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
NWithG = sum(1 for g in GSETS if g)
P("loaded %.0fs NR=%d nG=%d" % (time.time() - t0, NR, NWithG))

SPK = {}
for i in range(NR):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    if mm:
        SPK.setdefault(CONVKEY[i], set()).add(mm.group(1))
def speaker_of(i):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    return mm.group(1) if mm else None

POOLSZ = 700
POOLA = [None] * NALL
FEATS = [None] * NALL   # 334维
QOBJ = [None] * NALL
for k_i, qa in enumerate(IDS):
    qi = k_i
    q = Q[qa]
    QOBJ[k_i] = q
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
    qtext = q["question"].lower()
    conv_speakers = SPK.get(CONVKEY[pool[0]] if pool else "x", set())
    q_spk = set(s for s in conv_speakers if s and s.lower() in qtext)
    uniq_spk = (list(q_spk)[0] if len(q_spk) == 1 else None)
    Fm = np.zeros((len(pool), 334), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAW[c])
        cts = set(stem(w) for w in ct)
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
        seq2 = 0; mx = 0; last_p = -1
        for w in qstems:
            p = rl_.find(w)
            if p >= 0 and p > last_p:
                seq2 += 1; last_p = p; mx = max(mx, seq2)
            elif p >= 0:
                seq2 = 1; last_p = p
            else:
                seq2 = 0
        r1 = mx / max(1, len(qstems))
        big = [(qstems_list[i], qstems_list[i + 1]) for i in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAW[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        Fm[rr, 321:329] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1, r2, wvotes, 0.0]
        # ===== 新5维: 反回声+说话人 =====
        new_ratio = len((cts - set(stem(w) for w in qtok)) - XSTOP) / max(1, len(cts))
        echo = len(ct & qtok) / max(1, len(ct))
        long_ratio = sum(1 for w in (cts - set(stem(w) for w in qtok)) - XSTOP if len(w) >= 6) / max(1, len(cts))
        anti = new_ratio - echo
        s = speaker_of(c)
        spk_mis = 1.0 if (uniq_spk is not None and s is not None and s != uniq_spk) else 0.0
        Fm[rr, 329:334] = [new_ratio, echo, long_ratio, anti, spk_mis]
    FEATS[k_i] = Fm
    if k_i % 400 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier

COLS_OLD = np.arange(329)
COLS_NEW = np.arange(334)

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

ARMS = [("base(旧329+GBDT1)", COLS_OLD, False), ("hardfeat(334+GBDT2难负例)", COLS_NEW, True)]
RES = {a[0]: {"a5": 0, "a3": 0} for a in ARMS}
FORD10 = {}
n = 0
for hold in FOLDS:
    tr_idx = [k for k, qa in enumerate(IDS) if CONV_of[qa] != hold]
    # GBDT-1: 随机负例(同F70)
    trF1, trY1 = [], []
    for k_i in tr_idx:
        pool = POOLA[k_i]
        G = set()
        for grp in GSETS[k_i]:
            G |= grp
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
        for rr in gold_rows:
            trF1.append(FEATS[k_i][rr][:329]); trY1.append(1)
        for rr in noisepick:
            trF1.append(FEATS[k_i][rr][:329]); trY1.append(0)
    clf1 = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf1.fit(np.array(trF1, dtype=np.float32), np.array(trY1, dtype=np.int8))
    # GBDT-2: 难负例 = GBDT-1 初排头部20噪声 + 5随机深噪 (334维)
    trF2, trY2 = [], []
    for k_i in tr_idx:
        pool = POOLA[k_i]
        Fq = FEATS[k_i]
        s1 = clf1.decision_function(Fq[:, :329])
        order = np.argsort(-s1)
        G = set()
        for grp in GSETS[k_i]:
            G |= grp
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        hard = [rr for rr in order[:20] if pool[rr] not in G][:12]
        deep = list(np.random.RandomState(k_i + 777).permutation(
            [rr for rr in range(len(pool)) if pool[rr] not in G])[:5])
        for rr in gold_rows:
            trF2.append(Fq[rr]); trY2.append(1)
        for rr in hard + deep:
            trF2.append(Fq[rr]); trY2.append(0)
    clf2 = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf2.fit(np.array(trF2, dtype=np.float32), np.array(trY2, dtype=np.int8))
    # 推理
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        pool = POOLA[k_i]
        for name, cols, use2 in ARMS:
            clf = clf2 if use2 else clf1
            s = clf.decision_function(FEATS[k_i][:, cols])
            seq = [pool[i] for i in np.argsort(-s)]
            post = v9_post(seq, qa)
            if not use2:
                FORD10[qa] = [MID[i] for i in post[:35]]
            Gflat = set()
            for grp in G:
                Gflat |= grp
            if all(any(g in post[:5] for g in grp) for grp in G):
                RES[name]["a5"] += 1
            if all(any(g in post[:3] for g in grp) for grp in G):
                RES[name]["a3"] += 1
    P("  fold %s done %.0fs" % (hold[-12:], time.time() - t0))

P("\n===== 结果 (n=%d, v9后处理, 折叠金口径) =====" % n)
for name, _, _ in ARMS:
    P("%-22s @3=%5.1f%%  @5=%5.1f%%" % (name, 100.0 * RES[name]["a3"] / max(1, n), 100.0 * RES[name]["a5"] / max(1, n)))
with open(HERE + "/r41_final_order_v10.json", "w", encoding="utf-8") as f:
    json.dump(FORD10, f, ensure_ascii=False)
P("v10(GBDT裸排+分域+rbak去重,前35)落盘: r41_final_order_v10.json")
P("F74_DONE %.0fs" % (time.time() - t0))
LOG.close()
