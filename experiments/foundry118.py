# -*- coding: utf-8 -*-
"""foundry118.py — 两库混训通用ranker(用户: 不是拒收ranker,是要通用的):
训练=LoCoMo全量(真turn标签)+LME A半(会话弱标签) -> 测试=LME B半(haystack零重叠)
对照=纯LoCoMo直迁(60.7基线) | 验证=answer词干对错"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry118_results.txt", "w", encoding="utf-8")
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
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)
def pool_rank(Fm):
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    for j in range(Fm.shape[1]):
        out[np.argsort(Fm[:, j], kind="stable"), j] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return out

def feats12(qtext, rec_text, qe, qw, re1024, re256, qstems, qstems_list, qtok, RTOK_r):
    ct = toks(rec_text)
    inter = qtok & ct
    vqc = float(re256 @ qw)
    dqc = float(re1024 @ qe)
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", rec_text.lower()))
    g1 = len(qstems & rstems) / max(1, len(qstems))
    wvotes = sum(1 for w in qstems_list if w in RTOK_r)
    big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
    rset_ = set(re.findall(r"[a-z']+", rec_text.lower()))
    r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
    return [len(inter) / max(1, len(qtok | ct)), len(inter) / max(1, len(qtok)), len(inter) / max(1, len(ct)),
            1.0 if rec_text.rstrip().endswith("?") else 0.0,
            len(rec_text.split()) / max(1, len(qtext.split())), vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]

# ===== LoCoMo =====
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
RAWS_C = [rec_raw(m) for m in MID]
RAWN_C = [norm(x) for x in RAWS_C]
D_C = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
QW_C = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW_C[i] = v
QW_C = l2n(QW_C)
Q_C = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q_C[q["qa_id"]] = q
X_C = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
Q256_C = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS_C = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]
RTOK_C = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS_C]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN_C[i] for k in keys))
P("LoCoMo loaded %.0fs" % (time.time() - t0))

# ===== LME =====
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
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS_L])
def hay_key(q):
    return hashlib.md5(json.dumps(sorted(q["haystack_session_ids"]), ensure_ascii=False).encode()).hexdigest()
hk = {}
for qi, q in enumerate(d):
    hk.setdefault(hay_key(q), []).append(qi)
keys = sorted(hk)
fd = {k: i % 2 for i, k in enumerate(keys)}
qfold = np.array([fd[hay_key(q)] for q in d])  # 0=A半(训练), 1=B半(测试)
P("LME loaded %.0fs" % (time.time() - t0))

# ===== 构造训练池特征 =====
# LoCoMo侧
LC_POOL, LC_FEAT, LC_GOLD = {}, {}, {}
for k_i, qa in enumerate(IDS_C):
    c0 = D_C @ X_C[k_i]
    pool = set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-c0)[:250])
    for i in list(np.argsort(-FINAL[k_i])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(MID) and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:700]
    qtext = Q_C[qa]["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS_C[c], X_C[k_i], Q256_C[k_i], D_C[c], QW_C[c], qstems, qstems_list, qtok, RTOK_C[c])
    LC_POOL[k_i] = pool
    LC_FEAT[k_i] = pool_rank(Fm)
    LC_GOLD[k_i] = gold_set(qa) & set(pool)
P("LoCoMo feats %.0fs" % (time.time() - t0))

# LME侧(A半训练+B半测试都构造)
LME_POOL, LME_FEAT, LME_GOLDH = {}, {}, {}
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS_L) if s2 in hay_keys and NOHDR[i]])
    cq = D_L[domain] @ X_L[qi] + V256_L[domain] @ Q256_L[qi]
    pool = domain[np.argsort(-cq)[:700]].tolist()
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS_L[c], X_L[qi], Q256_L[qi], D_L[c], V256_L[c], qstems, qstems_list, qtok, RTOK_L[c])
    LME_POOL[qi] = pool
    LME_FEAT[qi] = pool_rank(Fm)
    LME_GOLDH[qi] = gold_h
P("LME feats %.0fs" % (time.time() - t0))

import lightgbm as lgb
def mk_ranker():
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                          num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                          random_state=0, verbosity=-1, n_jobs=4)

# ===== 混训ranker: LoCoMo真标签 + LME A半弱标签 =====
trF, trY, trG = [], [], []
for k_i, qa in enumerate(IDS_C):
    pool = LC_POOL[k_i]
    G = LC_GOLD[k_i]
    gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
    gset = set(gold_rows)
    noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
    for rr in gold_rows + noise:
        trF.append(LC_FEAT[k_i][rr]); trY.append(1 if rr in gset else 0)
    trG.append(len(gold_rows) + len(noise))
nA = 0
for qi in sorted(LME_POOL):
    if qfold[qi] != 0:
        continue
    pool = LME_POOL[qi]
    gold_rows = [rr for rr, c in enumerate(pool) if SIDS_L[c] in LME_GOLDH[qi]][:8]
    gset = set(gold_rows)
    noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
    for rr in gold_rows + noise:
        trF.append(LME_FEAT[qi][rr]); trY.append(1 if rr in gset else 0)
    trG.append(len(gold_rows) + len(noise))
    nA += 1
P("混训: LoCoMo=%d题 + LME_A=%d题, 总行=%d" % (len(IDS_C), nA, len(trF)))
rk_mix = mk_ranker()
rk_mix.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
P("混训ranker完成 %.0fs" % (time.time() - t0))

# ===== 对照: 纯LoCoMo ranker =====
trF2, trY2, trG2 = [], [], []
for k_i, qa in enumerate(IDS_C):
    pool = LC_POOL[k_i]
    G = LC_GOLD[k_i]
    gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
    gset = set(gold_rows)
    noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
    for rr in gold_rows + noise:
        trF2.append(LC_FEAT[k_i][rr]); trY2.append(1 if rr in gset else 0)
    trG2.append(len(gold_rows) + len(noise))
rk_pure = mk_ranker()
rk_pure.fit(np.array(trF2, dtype=np.float32), np.array(trY2, dtype=np.int8), group=trG2)
P("纯LoCoMo ranker完成 %.0fs" % (time.time() - t0))

# ===== 测试: LME B半(turn级answer验证) =====
res_mix = {1: 0, 3: 0, 5: 0}
res_pure = {1: 0, 3: 0, 5: 0}
resc = {1: 0, 3: 0, 5: 0}
nt = 0
for qi in sorted(LME_POOL):
    if qfold[qi] != 1:
        continue
    ans_raw = d[qi].get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    pool = LME_POOL[qi]
    nt += 1
    om = [pool[i] for i in np.argsort(-rk_mix.predict(LME_FEAT[qi]))]
    op = [pool[i] for i in np.argsort(-rk_pure.predict(LME_FEAT[qi]))]
    oc = [pool[i] for i in np.argsort(-(D_L[pool] @ X_L[qi]))]
    for k2 in (1, 3, 5):
        for od, acc in ((om, res_mix), (op, res_pure), (oc, resc)):
            if any(len(ans_stems & stems_of(RAWS_L[i])) / len(ans_stems) >= 0.5 for i in od[:k2]):
                acc[k2] += 1

P("\n===== LME B半 turn级(n=%d, A半训练零重叠) =====" % nt)
P("裸cos:        top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * resc[1] / nt, 100.0 * resc[3] / nt, 100.0 * resc[5] / nt))
P("纯LoCoMo直迁: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res_pure[1] / nt, 100.0 * res_pure[3] / nt, 100.0 * res_pure[5] / nt))
P("混训(LoCoMo+A半): top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res_mix[1] / nt, 100.0 * res_mix[3] / nt, 100.0 * res_mix[5] / nt))
P("F118_DONE %.0fs" % (time.time() - t0))
