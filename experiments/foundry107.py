# -*- coding: utf-8 -*-
"""foundry107.py — 公式实战验收(用户: 窗口含金量应=记忆库含金量):
LoCoMo+LME双基准, LOCO折外测试题:
①测试版C(k)校准(修F106训练集偏差) ②窗口含金量/库含金量 ③全金覆盖率"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry107_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

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
def pool_rank(Fm):
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    for j in range(Fm.shape[1]):
        out[np.argsort(Fm[:, j], kind="stable"), j] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return out

import lightgbm as lgb
FROZEN = dict(objective="lambdarank", n_estimators=500, learning_rate=0.08,
              num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
              random_state=0, verbosity=-1, n_jobs=4)

# ================= LoCoMo =================
P("===== LoCoMo(turn级金) =====")
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

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

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
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS[c])
        inter = qtok & ct
        vqc = float(QW[c] @ Q256[k_i])
        dqc = float(D[c] @ X[k_i])
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAWS[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        Fm[rr] = [len(inter) / max(1, len(qtok | ct)), len(inter) / qlen, len(inter) / max(1, len(ct)),
                  1.0 if RAWS[c].rstrip().endswith("?") else 0.0,
                  len(RAWS[c].split()) / max(1, len(qtext.split())), vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]
    POOL[k_i] = pool
    FEAT[k_i] = pool_rank(Fm)
    GS[k_i] = gold_set(qa)
P("feats %.0fs" % (time.time() - t0))

rank_gold = {}
win_gold_cnt = {5: 0, 15: 0}
lib_gold_cnt = 0
allgold = {5: 0, 15: 0}
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
    rk = lgb.LGBMRanker(**FROZEN)
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
        order = np.argsort(-s)
        seq = [pool[i] for i in order]
        for pos, c in enumerate(seq[:20]):
            rank_gold.setdefault(pos + 1, [0, 0])
            rank_gold[pos + 1][1] += 1
            if c in G:
                rank_gold[pos + 1][0] += 1
        lib_gold_cnt += len(G)
        for k2 in (5, 15):
            wgold = len(set(seq[:k2]) & G)
            win_gold_cnt[k2] += wgold
            if wgold == len(G):
                allgold[k2] += 1
    P("  fold %s %.0fs" % (hold[-12:], time.time() - t0))

P("\nLoCoMo测试版C(k)(折外): " + " ".join(
    "k%d=%.1f%%" % (k, 100.0 * rank_gold[k][0] / rank_gold[k][1]) for k in (1, 2, 3, 5, 10, 15) if k in rank_gold))
P("窗口含金量/库含金量: @5=%.1f%%  @15=%.1f%%  (库内金均%.1f条/题)" % (
    100.0 * win_gold_cnt[5] / lib_gold_cnt, 100.0 * win_gold_cnt[15] / lib_gold_cnt, lib_gold_cnt / nG))
P("全金覆盖: @5=%.1f%%  @15=%.1f%%" % (100.0 * allgold[5] / nG, 100.0 * allgold[15] / nG))
P("")

# ================= LME(会话级金, 用LoCoMo直迁模型) =================
P("===== LME(会话级金, 公式直迁) =====")
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS_L, SIDS_L = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS_L.append(r.get("raw") or "")
    SIDS_L.append(r.get("session_id") or "")
D_L = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256_L = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X_L = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256_L = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
RTOK_L = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS_L]
booster = lgb.Booster(model_file="C:/locomo_refined/memsys/ranker_ordinal_luco.txt")
rank_gold_s = {}
win_g5 = win_g15 = 0
lib_g5 = lib_g15 = 0
n = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = sorted(set(i for i, s2 in enumerate(SIDS_L) if s2 in hay_keys))
    cq = D_L[domain] @ X_L[qi] + V256_L[domain] @ Q256_L[qi]
    pool = [domain[i] for i in np.argsort(-cq)[:700]]
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    qlen = max(1, len(qtok))
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS_L[c])
        inter = qtok & ct
        vqc = float(V256_L[c] @ Q256_L[qi])
        dqc = float(D_L[c] @ X_L[qi])
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS_L[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        wvotes = sum(1 for w in qstems_list if w in RTOK_L[c])
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAWS_L[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        Fm[rr] = [len(inter) / max(1, len(qtok | ct)), len(inter) / qlen, len(inter) / max(1, len(ct)),
                  1.0 if RAWS_L[c].rstrip().endswith("?") else 0.0,
                  len(RAWS_L[c].split()) / max(1, len(qtext.split())), vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]
    s = booster.predict(pool_rank(Fm))
    seq = [pool[i] for i in np.argsort(-s)]
    gold_rows = set(i for i, s2 in enumerate(SIDS_L) if s2 in gold_h) & set(pool)
    for pos, c in enumerate(seq[:20]):
        rank_gold_s.setdefault(pos + 1, [0, 0])
        rank_gold_s[pos + 1][1] += 1
        if c in gold_rows:
            rank_gold_s[pos + 1][0] += 1
    n += 1
    for k2, acc in ((5, "a"), (15, "b")):
        seen = set(SIDS_L[r] for r in seq[:k2])
        hit = len(gold_h & seen)
        tot = len(gold_h)
        if k2 == 5:
            win_g5 += hit
            lib_g5 += tot
        else:
            win_g15 += hit
            lib_g15 += tot

P("LME测试版C_session(k): " + " ".join(
    "k%d=%.1f%%" % (k, 100.0 * rank_gold_s[k][0] / rank_gold_s[k][1]) for k in (1, 2, 3, 5, 10, 15) if k in rank_gold_s))
P("金会话命中量/库内金会话量: @5=%.1f%%  @15=%.1f%%" % (
    100.0 * win_g5 / lib_g5, 100.0 * win_g15 / lib_g15))
P("F107_DONE %.0fs" % (time.time() - t0))
