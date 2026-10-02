# -*- coding: utf-8 -*-
"""foundry114.py — 序数对齐的turn级直迁验证(用户洞见: 让库对齐LME就跑高分):
LoCoMo答案训12维序数ranker(旧库一次性) -> LME turn级零答案推理 -> answer词干对错"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry114_results.txt", "w", encoding="utf-8")
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

t0 = time.time()
# ===== LoCoMo: 训练(答案只在旧库) =====
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
IDS = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]
RTOK_C = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAWS_C]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN_C[i] for k in keys))

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

import lightgbm as lgb
trF, trY, trG = [], [], []
for k_i, qa in enumerate(IDS):
    c0 = D_C @ X_C[k_i]
    pool = set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-c0)[:250])
    for i in list(np.argsort(-FINAL[k_i])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(MID) and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:700]
    qtext = Q_C[qa]["question"]
    qstems = stems_of(qtext) - set()
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    G = gold_set(qa) & set(pool)
    gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
    gset = set(gold_rows)
    noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS_C[c], X_C[k_i], Q256_C[k_i], D_C[c], QW_C[c], qstems, qstems_list, qtok, RTOK_C[c])
    FR = pool_rank(Fm)
    for rr in gold_rows + noise:
        trF.append(FR[rr])
        trY.append(1 if rr in gset else 0)
    trG.append(len(gold_rows) + len(noise))
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                    num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                    random_state=0, verbosity=-1, n_jobs=4)
rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
P("LoCoMo训练完成(答案仅旧库) %.0fs" % (time.time() - t0))

# ===== LME: 零答案推理 + answer词干对错 =====
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
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
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
P("LME loaded %.0fs" % (time.time() - t0))

res = {1: 0, 3: 0, 5: 0}
resc = {1: 0, 3: 0, 5: 0}
nl = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    cq = D[domain] @ X[qi] + V256[domain] @ Q256[qi]
    pool = domain[np.argsort(-cq)[:700]].tolist()
    ans_raw = q.get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS[c], X[qi], Q256[qi], D[c], V256[c], qstems, qstems_list, qtok, RTOK[c])
    FR = pool_rank(Fm)
    s = rk.predict(FR)
    order = [pool[i] for i in np.argsort(-s)]
    cos_order = [pool[i] for i in np.argsort(-(D[pool] @ X[qi]))]
    nl += 1
    for k2 in (1, 3, 5):
        for order, acc in ((order, res), (cos_order, resc)):
            hit = False
            for i in order[:k2]:
                rs = stems_of(RAWS[i])
                if len(ans_stems & rs) / len(ans_stems) >= 0.5:
                    hit = True
                    break
            if hit:
                acc[k2] += 1
    if qi % 100 == 0:
        P("  %d %.0fs" % (qi, time.time() - t0))

P("\n===== LME turn级: 序数对齐直迁 vs 裸cos (验证=answer词干, n=%d) =====" % nl)
P("裸cos:      top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * resc[1] / nl, 100.0 * resc[3] / nl, 100.0 * resc[5] / nl))
P("序数ranker: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res[1] / nl, 100.0 * res[3] / nl, 100.0 * res[5] / nl))
P("F114_DONE %.0fs" % (time.time() - t0))
