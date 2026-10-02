# -*- coding: utf-8 -*-
"""lme_local.py — LME本地LOCO重训参考臂: 头部陪衬负例(probe3配方) + 328维同特征
5折按题, 测试session@5/@15, 对照: 直迁0增益/裸cos 71.0/78.8"""
import io, json, os, re, sys, time, hashlib
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

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

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sess, sid2h = {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    sid2h[sid] = h
    if h not in sess:
        sess[h] = turns
hashes = list(sess.keys())
RAWS, SIDS = [], []
for si, h in enumerate(hashes):
    RAWS.append("[hdr]")
    SIDS.append("lme-s" + h[:12])
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
        SIDS.append("lme-s" + h[:12])
NR = len(RAWS)
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS]
P("loaded NR=%d %.0fs" % (NR, time.time() - t0))

NQ = len(d)
# 预计算: 每题的域内行/金行/池(域内cos排序前700)
META = []
for qi, q in enumerate(d):
    gold_rows = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gold_rows |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows = np.array(sorted(set(r for k2 in hay_keys for r in SID2ROWS.get(k2, []))))
    if not gold_rows or len(hay_rows) < 100:
        META.append(None)
        continue
    cq = D[hay_rows] @ X[qi] + V256[hay_rows] @ Q256[qi]
    pool = hay_rows[np.argsort(-cq)[:700]]
    gold_h = set("lme-s" + sid2h[s][:12] for s in q["answer_session_ids"] if s in sid2h)
    META.append((pool, gold_rows, gold_h))
    if qi % 100 == 0:
        P("  meta %d %.0fs" % (qi, time.time() - t0))
P("meta done %.0fs" % (time.time() - t0))

def build_feats(qi, pool, top30):
    q = d[qi]
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    qlen = max(1, len(qtok))
    qv256 = Q256[qi]
    qv1024 = X[qi]
    Fm = np.zeros((len(pool), 328), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAWS[c].rstrip().endswith("?") else 0.0
        lr = len(RAWS[c]) / max(1, len(qtext))
        vqc = float(qv256 @ V256[c])
        dqc = float(D[c] @ qv1024)
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, 0.0, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:112] = D[c][:100]
        Fm[rr, 112:212] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 212:312] = V256[c][:100]
        Fm[rr, 312:315] = [qmark, 0.0, min(lr, 4) / 4]
        nb_flag = 1.0 if any(abs(c - j) <= 2 and SIDS[j] == SIDS[c] for j in top30) else 0.0
        rl_ = RAWS[c].lower()
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
        rset_ = set(re.findall(r"[a-z']+", RAWS[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        Fm[rr, 315:323] = [1.0 if dqc > 0.65 else 0.0, dqc - g1, dqc * (1.0 - g1), nb_flag, r1, r2, wvotes, 0.0]
    return Fm

FEATS = {}
for qi in range(NQ):
    m = META[qi]
    if m is None:
        continue
    pool, _, _ = m
    top30 = list(pool[:30])
    FEATS[qi] = build_feats(qi, pool, top30)
P("feats %d题 %.0fs" % (len(FEATS), time.time() - t0))

import lightgbm as lgb
# 按haystack分折(修分布泄漏: 同haystack的题必须同折)
def hay_key(q):
    return hashlib.md5(json.dumps(sorted(q["haystack_session_ids"]), ensure_ascii=False).encode()).hexdigest()
hk = {}
for qi, q in enumerate(d):
    hk.setdefault(hay_key(q), []).append(qi)
keys = sorted(hk)
kfold = {}
for i, k in enumerate(keys):
    kfold[k] = i % 5
folds = np.array([kfold[hay_key(q)] for q in d])
P("haystack数=%d, 题均=%.2f/haystack" % (len(keys), NQ / len(keys)))
valid = [qi for qi in FEATS]
res_l = {"a5": 0, "a15": 0, "c5": 0, "c15": 0}
n = 0
for fd in range(5):
    tr = [qi for qi in valid if folds[qi] != fd]
    te = [qi for qi in valid if folds[qi] == fd]
    trF, trY, trG = [], [], []
    for qi in tr:
        pool, gold_rows, _ = META[qi]
        gold_idx = [rr for rr, c in enumerate(pool) if c in gold_rows][:8]
        gset = set(gold_idx)
        # 头部陪衬负例: 池序(域内cos序)前40非金
        noise_idx = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
        rows = gold_idx + noise_idx
        for rr in rows:
            trF.append(FEATS[qi][rr])
            trY.append(1 if rr in gset else 0)
        trG.append(len(rows))
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                            num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                            random_state=0, verbosity=-1, n_jobs=4)
    ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    for qi in te:
        pool, gold_rows, gold_h = META[qi]
        n += 1
        s = ranker.predict(FEATS[qi])
        order = [pool[i] for i in np.argsort(-s)]
        seen5 = set(SIDS[r] for r in order[:5])
        seen15 = set(SIDS[r] for r in order[:15])
        if gold_h <= seen5:
            res_l["a5"] += 1
        if gold_h <= seen15:
            res_l["a15"] += 1
    P("  fold %d done %.0fs" % (fd, time.time() - t0))

P("\n===== LME本地LOCO(头部负例) n=%d =====" % n)
P("session@5=%.1f%%  @15=%.1f%%" % (100.0 * res_l["a5"] / n, 100.0 * res_l["a15"] / n))
P("对照: 裸cos域内71.0/78.8, ranker直迁70.8/78.4")
P("done %.0fs" % (time.time() - t0))
