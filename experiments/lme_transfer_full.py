# -*- coding: utf-8 -*-
"""lme_transfer.py — GBDT/ranker直迁LME冒烟: LoCoMo模型零改动+冻结配置
检查: ①池全金 ②裸cos基线 ③ranker直迁增益 ④haystack域对照 ⑤非空率"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["OMP_NUM_THREADS"] = "4"
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
# ===== Stage A 确定性重建(与建库同逻辑) =====
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sess, sdate, sid2h = {}, {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    sid2h[sid] = h
    if h not in sess:
        sess[h] = turns
        sdate[h] = dt
hashes = list(sess.keys())
RAWS, SIDS = [], []
for si, h in enumerate(hashes):
    dt = (sdate[h] or "")[:10]
    RAWS.append("[Session %d — %s]" % (si + 1, dt))
    SIDS.append("lme-s" + h[:12])
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
        SIDS.append("lme-s" + h[:12])
NR = len(RAWS)
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
P("records=%d sessions=%d %.0fs" % (NR, len(sess), time.time() - t0))

D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
assert D.shape[0] == NR and V256.shape[0] == NR, "row mismatch %s %s %d" % (D.shape, V256.shape, NR)
P("vectors loaded D=%s V=%s %.0fs" % (D.shape, V256.shape, time.time() - t0))

import lightgbm as lgb
booster = lgb.Booster(model_file="C:/locomo_refined/memsys/ranker_v11clean_328.txt")
P("ranker loaded %.0fs" % (time.time() - t0))

RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS]
# 邻域: 同SIDS的相邻记录(记录序即会话内序)
def neighbors(i, top_list):
    out = []
    for j in top_list:
        for off in (-2, -1, 1, 2):
            k2 = j + off
            if 0 <= k2 < NR and SIDS[k2] == SIDS[j]:
                out.append(k2)
    return out

NQ = len(d)
NSEL = 500
rng = np.random.RandomState(0)
sel = sorted(list(range(NQ)))

def build_feats(qi, pool, top30list):
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
        nb_flag = 1.0 if any(abs(c - j) <= 2 and SIDS[j] == SIDS[c] for j in top30list) else 0.0
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

stats = {"pool_all": 0, "pool_any": 0, "cos_a5": 0, "cos_a15": 0, "rank_a5": 0, "rank_a15": 0,
         "rankd_a5": 0, "rankd_a15": 0, "n": 0, "empty": 0}
qtype_stat = {}
for qi in sel:
    q = d[qi]
    gold_sids = set(q.get("answer_session_ids") or [])
    gold_h = set(sid2h[s] for s in gold_sids if s in sid2h)
    gold_rows = set()
    for s in gold_sids:
        gold_rows |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    if not gold_rows:
        stats["empty"] += 1
        continue
    stats["n"] += 1
    C0q = D @ X[qi]
    C256q = V256 @ Q256[qi]
    pool = set(np.argsort(-C0q)[:350]) | set(np.argsort(-C256q)[:350])
    top30 = list(np.argsort(-C0q)[:30])
    for j in neighbors(0, top30):
        pool.add(j)
    pool = sorted(pool)[:700]
    poolset = set(pool)
    if gold_rows <= poolset:
        stats["pool_all"] += 1
    if gold_rows & poolset:
        stats["pool_any"] += 1
    # 裸cos基线
    cos_sc = C0q[pool] + C256q[pool]
    cos_order = [pool[i] for i in np.argsort(-cos_sc)]
    # ranker直迁
    Fm = build_feats(qi, pool, top30)
    rs = booster.predict(Fm)
    rank_order = [pool[i] for i in np.argsort(-rs)]
    # haystack域过滤版(ranker)
    hay_h = set(sid2h[s] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows_ok = lambda r: SIDS[r] in set("lme-s" + h[:12] for h in hay_h)
    rank_order_d = [r for r in rank_order if hay_rows_ok(r)] if len(hay_h) < len(sess) else rank_order

    def allk(order, k):
        seen = set()
        for r in order[:k]:
            seen.add(SIDS[r])
        return all(("lme-s" + h[:12]) in seen for h in gold_h)
    if allk(cos_order, 5):
        stats["cos_a5"] += 1
    if allk(cos_order, 15):
        stats["cos_a15"] += 1
    if allk(rank_order, 5):
        stats["rank_a5"] += 1
    if allk(rank_order, 15):
        stats["rank_a15"] += 1
    if allk(rank_order_d, 5):
        stats["rankd_a5"] += 1
    if allk(rank_order_d, 15):
        stats["rankd_a15"] += 1
    qt = str(q.get("question_type"))[:14]
    qtype_stat.setdefault(qt, [0, 0])
    qtype_stat[qt][0] += 1
    if allk(rank_order, 5):
        qtype_stat[qt][1] += 1

n = stats["n"]
P("\n===== LME直迁冒烟 n=%d (空金跳过%d) =====" % (n, stats["empty"]))
P("①池全金率(all/any): %.1f%% / %.1f%%" % (100.0 * stats["pool_all"] / n, 100.0 * stats["pool_any"] / n))
P("②裸cos基线  session@5=%.1f%%  @15=%.1f%%" % (100.0 * stats["cos_a5"] / n, 100.0 * stats["cos_a15"] / n))
P("③ranker直迁 session@5=%.1f%%  @15=%.1f%%" % (100.0 * stats["rank_a5"] / n, 100.0 * stats["rank_a15"] / n))
P("④+haystack域 session@5=%.1f%%  @15=%.1f%%" % (100.0 * stats["rankd_a5"] / n, 100.0 * stats["rankd_a15"] / n))
P("分题型@5:")
for qt, (a, b) in sorted(qtype_stat.items(), key=lambda x: -x[1][0]):
    P("  %-14s %5.1f%% (%d/%d)" % (qt, 100.0 * b / a, b, a))
P("done %.0fs" % (time.time() - t0))
