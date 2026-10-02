# -*- coding: utf-8 -*-
"""foundry101.py — 序数化ranker(用户洞察: 边界硬编码→池内分位数化):
LoCoMo训练(全列分位变换)→LME零标注直迁→直迁增益应从0恢复
臂: A裸cos基线 B序数ranker直迁(LoCoMo训) C序数ranker本地(上界参考)"""
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
# ===== LoCoMo =====
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
RAWN_C = [re.sub(r"[^a-z0-9]+", "", x.lower()) for x in RAWS_C]
D_C = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
QW_C = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW_C[i] = v
QW_C = l2n(QW_C)
Q_C = {}
for l in io.open(HERE + "/locomo_qs.jsonl", encoding="utf-8") if os.path.exists(HERE + "/locomo_qs.jsonl") else io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q_C[q["qa_id"]] = q
X_C = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS_C = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]
CONV_of = {qa: qa.split("#")[0] for qa in IDS_C}
FOLDS_C = sorted(set(CONV_of.values()))
RTOK_C = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS_C]
def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR_C = {}
seen = set()
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR_C[i] = 1
    seen.add(k)
def gold_groups(qa):
    keys = [re.sub(r"[^a-z0-9]+", "", (em.get("text") or "").lower())[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(re.sub(r"[^a-z0-9]+", "", (em.get("text") or "").lower())) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN_C[i])
        if not hits:
            continue
        for i in list(hits):
            if i in PAIR_C:
                hits.add(i - 1) if False else None
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups
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
P("LME loaded %.0fs" % (time.time() - t0))

Q256_C = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))
NFEAT = 12

def build_pool_feats_LoCoMo(k_i):
    qa = IDS_C[k_i]
    q = Q_C[qa]
    qtext = q["question"]
    pool = set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-(D_C @ X_C[k_i]))[:250])
    for i in list(np.argsort(-FINAL[k_i])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(MID) and re.sub(r"_(rbak\d+k|m\d+)$", "", MID[j]) == re.sub(r"_(rbak\d+k|m\d+)$", "", MID[i]):
                pool.add(j)
    pool = sorted(pool)[:700]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    qlen = max(1, len(qtok))
    Fm = np.zeros((len(pool), NFEAT), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS_C[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAWS_C[c].rstrip().endswith("?") else 0.0
        lr = len(RAWS_C[c].split()) / max(1, len(qtext.split()))
        vqc = float(QW_C[c] @ Q256_C[k_i])
        dqc = float(D_C[c] @ X_C[k_i])
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS_C[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        wvotes = sum(1 for w in qstems_list if w in RTOK_C[c])
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAWS_C[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        Fm[rr] = [jac, qcov, ccov, qmark, lr, vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]
    return pool, Fm

def pool_rank(Fm):
    """每列→池内分位rank/(n-1)  (n, NF) -> (n, NF)"""
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    for j in range(Fm.shape[1]):
        out[np.argsort(Fm[:, j], kind="stable"), j] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return out

import types
KC = types.SimpleNamespace(RAWS=RAWS_C)
KL = types.SimpleNamespace(RAWS=RAWS_L)

def build_pool_feats_LME(qi):
    q = d[qi]
    qtext = q["question"]
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows = np.array(sorted(set(r for k2 in hay_keys for r in [i for i, s2 in enumerate(SIDS_L) if s2 == k2])))
    if len(hay_rows) == 0:
        return None, None
    cq = D_L[hay_rows] @ X_L[qi] + V256_L[hay_rows] @ Q256_L[qi]
    pool = hay_rows[np.argsort(-cq)[:700]].tolist()
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), NFEAT), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = toks(RAWS_L[c])
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / max(1, len(qtok))
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAWS_L[c].rstrip().endswith("?") else 0.0
        lr = len(RAWS_L[c].split()) / max(1, len(qtext.split()))
        vqc = float(V256_L[c] @ Q256_L[qi])
        dqc = float(D_L[c] @ X_L[qi])
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS_L[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        wvotes = sum(1 for w in qstems_list if w in RTOK_L[c])
        big = [(qstems_list[i2], qstems_list[i2 + 1]) for i2 in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAWS_L[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        Fm[rr] = [jac, qcov, ccov, qmark, lr, vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]
    return pool, Fm

P("building LoCoMo ordinal feats...")
POOLC, FEATC, FEATC_RAW = {}, {}, {}
for k_i in range(len(IDS_C)):
    pool, Fm = build_pool_feats_LoCoMo(k_i)
    POOLC[k_i], FEATC[k_i], FEATC_RAW[k_i] = pool, pool_rank(Fm), Fm.copy()
    if k_i % 400 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))
P("LoCoMo feats %.0fs" % (time.time() - t0))

import lightgbm as lgb
# ===== LoCoMo训练序数ranker(LOCO) 并保存直迁用 =====
ord_ranker = None
GSETS_C = [gold_groups(qa) for qa in IDS_C]
def train_lc_ranker(exclude_fold=None):
    trF, trY, trG = [], [], []
    for k_i, qa in enumerate(IDS_C):
        if exclude_fold is not None and CONV_of[qa] == exclude_fold:
            continue
        pool = POOLC[k_i]
        G = set()
        for g in GSETS_C[k_i]:
            G |= set(pool) & g
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        gset = set(gold_rows)
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        np_ = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:40]) if noise_rows else []
        for rr in gold_rows + np_:
            trF.append(FEATC[k_i][rr])
            trY.append(1 if rr in gset else 0)
        trG.append(len(gold_rows) + len(np_))
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                        num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                        random_state=0, verbosity=-1, n_jobs=4)
    rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    return rk

# ===== LME 建序数特征+评测 =====
P("building LME ordinal feats...")
POOLL, FEATL, GOLDH, FEATL_RAW = {}, {}, {}, {}
for qi in range(len(d)):
    pool, Fm = build_pool_feats_LME(qi)
    if pool is None:
        continue
    gold_h = set("lme-s" + sid2h[s][:12] for s in (d[qi].get("answer_session_ids") or []) if s in sid2h)
    POOLL[qi], FEATL[qi], GOLDH[qi] = pool, pool_rank(Fm), gold_h
    FEATL_RAW[qi] = Fm.copy()
P("LME feats %.0fs" % (time.time() - t0))

def eval_lme(model, feats=None):
    a5 = a15 = 0
    n = 0
    for qi in POOLL:
        n += 1
        s = model.predict((feats or FEATL)[qi])
        pool = POOLL[qi]
        order = [pool[i] for i in np.argsort(-s)]
        s5 = set(SIDS_L[r] for r in order[:5])
        s15 = set(SIDS_L[r] for r in order[:15])
        if GOLDH[qi] <= s5:
            a5 += 1
        if GOLDH[qi] <= s15:
            a15 += 1
    return 100.0 * a5 / n, 100.0 * a15 / n

# 直迁: LoCoMo全量训练(不排除任何fold) → LME推理
P("training LoCoMo full ordinal ranker...")
rk_full = train_lc_ranker(None)
rk_full.booster_.save_model("C:/locomo_refined/memsys/ranker_ordinal_luco.txt")
b5, b15 = eval_lme(rk_full)
P("\n===== 序数ranker直迁(LoCoMo训→LME零标注) =====")
P("B序数直迁: session@5=%.1f%%  @15=%.1f%%" % (b5, b15))
P("对照: A裸cos=71.0/78.8 | 旧直迁(绝对值特征)=70.8/78.4 | 本地绝对值=74.0/85.6")

# 绝对值12维对照(归因: 砍列 vs 序数化)
def train_lc_raw():
    trF, trY, trG = [], [], []
    for k_i, qa in enumerate(IDS_C):
        pool = POOLC[k_i]
        G = set()
        for g in GSETS_C[k_i]:
            G |= set(pool) & g
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        gset = set(gold_rows)
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        np_ = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:40]) if noise_rows else []
        for rr in gold_rows + np_:
            trF.append(FEATC_RAW[k_i][rr])
            trY.append(1 if rr in gset else 0)
        trG.append(len(gold_rows) + len(np_))
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                        num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                        random_state=0, verbosity=-1, n_jobs=4)
    rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    return rk

P("training raw-value control ranker...")
rk_raw = train_lc_raw()
r5, r15 = eval_lme(rk_raw, FEATL_RAW)
P("B0绝对值12维直迁: session@5=%.1f%%  @15=%.1f%%" % (r5, r15))
b5, b15 = eval_lme(rk_full)
P("B1序数12维直迁(复跑): session@5=%.1f%%  @15=%.1f%%" % (b5, b15))
P("归因: B0-旧直迁70.8=砍列贡献 | B1-B0=序数化净贡献")
P("F102_DONE %.0fs" % (time.time() - t0))
