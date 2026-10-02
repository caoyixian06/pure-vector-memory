# -*- coding: utf-8 -*-
"""unirank.bench.run_transfer_check — 库验收: 正式管线双基准跑分
LME: 序数直迁(对标79.2/92.8, 必须复现)
LoCoMo: 序数版LOCO回测(v11绝对值版78.2核心@5为对照, 掉了要归因)"""
import io, json, os, re, sys, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "C:/locomo_refined")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

from unirank.core.ordinal import pool_rank
from unirank.index.dual_channel import build_pool
from unirank.rank.lambdarank import head_negatives, FROZEN
import lightgbm as lgb

t0 = time.time()

def feats12(qtext, rec_text, qvec1024, qvec256, rec1024, rec256, qstems, qstems_list, qtok, rtok_all, RTOK):
    """foundry101逐列对齐: [jac,qcov,ccov,qmark,lr,vqc,dqc,vqc-dqc,g1,wvotes,r2,dqc-g1]"""
    ct = rtok_all
    inter = qtok & ct
    jac = len(inter) / max(1, len(qtok | ct))
    qcov = len(inter) / max(1, len(qtok))
    ccov = len(inter) / max(1, len(ct))
    qmark = 1.0 if rec_text.rstrip().endswith("?") else 0.0
    lr = len(rec_text.split()) / max(1, len(qtext.split()))
    vqc = float(rec256 @ qvec256)
    dqc = float(rec1024 @ qvec1024)
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", rec_text.lower()))
    g1 = len(qstems & rstems) / max(1, len(qstems))
    wvotes = sum(1 for w in qstems_list if w in RTOK)
    big = [(qstems_list[i], qstems_list[i + 1]) for i in range(len(qstems_list) - 1)]
    rset_ = set(re.findall(r"[a-z']+", rec_text.lower()))
    r2 = sum(1 for a, b in big if a in rset_ and b in rset_) / max(1, len(big))
    return [jac, qcov, ccov, qmark, lr, vqc, dqc, vqc - dqc, g1, wvotes, r2, dqc - g1]

def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# ================= LME =================
P("===== LME(库管线) =====")
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest() if False else None
import hashlib
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS]
booster = lgb.Booster(model_file="C:/locomo_refined/memsys/ranker_ordinal_luco.txt")
a5 = a15 = 0
n = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain_rows = sorted(set(r for k2 in hay_keys for r in [i for i, s2 in enumerate(SIDS) if s2 == k2]))
    pool, anchor = build_pool(D, V256, X[qi], Q256[qi], SIDS, topk=350, nb_anchor=30, domain=domain_rows)
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS[c], X[qi], Q256[qi], D[c], V256[c], qstems, qstems_list, qtok, toks(RAWS[c]), RTOK[c])
    order = list(np.argsort(-booster.predict(pool_rank(Fm))))
    seq = [pool[i] for i in order]
    s5 = set(SIDS[r] for r in seq[:5])
    s15 = set(SIDS[r] for r in seq[:15])
    n += 1
    if gold_h <= s5:
        a5 += 1
    if gold_h <= s15:
        a15 += 1
P("unirank-LME直迁: session@5=%.1f%%  @15=%.1f%%  (对标79.2/92.8)" % (100.0 * a5 / n, 100.0 * a15 / n))

# ================= LoCoMo序数回测 =================
P("\n===== LoCoMo(序数版LOCO, 对照v11绝对值78.2核心@5) =====")
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
CONV_of = {qa: qa.split("#")[0] for qa in IDS_C}
FOLDS = sorted(set(CONV_of.values()))
RTOK_C = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS_C]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN_C[i] for k in keys))

# 预构LoCoMo池+特征(全题)
POOLC, FEATC, GSETS = {}, {}, {}
for k_i, qa in enumerate(IDS_C):
    pool, anchor = build_pool(D_C, QW_C, X_C[k_i], Q256_C[k_i], CONVKEY, topk=350, nb_anchor=30)
    qtext = Q_C[qa]["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS_C[c], X_C[k_i], Q256_C[k_i], D_C[c], QW_C[c], qstems, qstems_list, qtok, toks(RAWS_C[c]), RTOK_C[c])
    POOLC[k_i] = pool
    FEATC[k_i] = pool_rank(Fm)
    GSETS[k_i] = gold_set(qa)
    if k_i % 400 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))
P("LoCoMo特征完成 %.0fs" % (time.time() - t0))

def core_group(qa, grps_raw):
    ans = " ".join(str(x) for x in (Q_C[qa].get("answer") or []))
    aw = set(stem(w) for w in re.findall(r"[a-z]+", ans.lower()) if len(w) > 2)
    if not aw:
        return grps_raw[0] if grps_raw else set()
    best, bv = None, -1.0
    for grp in grps_raw:
        tw = set()
        for g in grp:
            tw |= set(stem(w) for w in re.findall(r"[a-z']+", RAWS_C[g].lower()))
        cov = len(aw & tw) / len(aw)
        if cov > bv:
            bv, best = cov, grp
    return best

# LOCO按会话
res = {"c5": 0, "a15": 0, "a5all": 0}
nG = 0
for hold in FOLDS:
    trF, trY, trG = [], [], []
    for k_i, qa in enumerate(IDS_C):
        if CONV_of[qa] == hold:
            continue
        pool = POOLC[k_i]
        G = GSETS[k_i] & set(pool)
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        gset = set(gold_rows)
        noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
        for rr in gold_rows + noise:
            trF.append(FEATC[k_i][rr])
            trY.append(1 if rr in gset else 0)
        trG.append(len(gold_rows) + len(noise))
    rk = lgb.LGBMRanker(**FROZEN)
    rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    for k_i, qa in enumerate(IDS_C):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        nG += 1
        pool = POOLC[k_i]
        s = rk.predict(FEATC[k_i])
        seq = [pool[i] for i in np.argsort(-s)]
        if all(g in set(seq[:15]) for g in G):
            res["a15"] += 1
        if all(g in set(seq[:5]) for g in G):
            res["a5all"] += 1
        cg = core_group(qa, [G]) if isinstance(G, set) else None
        if cg and (cg & set(seq[:5])):
            res["c5"] += 1
    P("  fold %s %.0fs" % (hold[-12:], time.time() - t0))

P("\nunirank-LoCoMo序数版: 核心近似@5=%.1f%%  全金@5=%.1f%%  全金@15=%.1f%%" % (
    100.0 * res["c5"] / nG, 100.0 * res["a5all"] / nG, 100.0 * res["a15"] / nG))
P("对照: v11绝对值版 核心@5=78.2 全金@15=80.5")
P("DONE %.0fs" % (time.time() - t0))
