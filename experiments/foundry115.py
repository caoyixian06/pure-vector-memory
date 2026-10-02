# -*- coding: utf-8 -*-
"""foundry115.py — 200题标定→300题泛化(用户: 标定200能不能提高整体500):
LME按haystack分折, 前2折~200题训练序数ranker(会话级弱标签), 其余~300题测试
对照: 零标定裸cos=60.7@5 | 验证=answer词干对错"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry115_results.txt", "w", encoding="utf-8")
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
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)
def pool_rank(Fm):
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    for j in range(Fm.shape[1]):
        out[np.argsort(Fm[:, j], kind="stable"), j] = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return out

t0 = time.time()
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

# haystack分折
def hay_key(q):
    return hashlib.md5(json.dumps(sorted(q["haystack_session_ids"]), ensure_ascii=False).encode()).hexdigest()
hk = {}
for qi, q in enumerate(d):
    hk.setdefault(hay_key(q), []).append(qi)
keys = sorted(hk)
fd = {k: i % 5 for i, k in enumerate(keys)}
qfold = np.array([fd[hay_key(q)] for q in d])
P("loaded %.0fs" % (time.time() - t0))

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

def build(qi):
    q = d[qi]
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    cq = D[domain] @ X[qi] + V256[domain] @ Q256[qi]
    pool = domain[np.argsort(-cq)[:700]].tolist()
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    for rr, c in enumerate(pool):
        Fm[rr] = feats12(qtext, RAWS[c], X[qi], Q256[qi], D[c], V256[c], qstems, qstems_list, qtok, RTOK[c])
    return pool, pool_rank(Fm), gold_h

# 标定集: 折0+1 (~200题)
trF, trY, trG = [], [], []
n_train = 0
for qi, q in enumerate(d):
    if qfold[qi] >= 2:
        continue
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    pool, FR, gold_h = build(qi)
    gold_rows = [rr for rr, c in enumerate(pool) if SIDS[c] in gold_h][:8]
    gset = set(gold_rows)
    noise = [rr for rr in range(min(48, len(pool))) if rr not in gset][:40]
    for rr in gold_rows + noise:
        trF.append(FR[rr])
        trY.append(1 if rr in gset else 0)
    trG.append(len(gold_rows) + len(noise))
    n_train += 1
P("标定集=%d题 训练行=%d %.0fs" % (n_train, len(trF), time.time() - t0))

import lightgbm as lgb
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                    num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                    random_state=0, verbosity=-1, n_jobs=4)
rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
P("标定ranker训练完成 %.0fs" % (time.time() - t0))

# 测试: 折2-4 (~300题, haystack零重叠)
res = {1: 0, 3: 0, 5: 0}
resc = {1: 0, 3: 0, 5: 0}
nt = 0
for qi, q in enumerate(d):
    if qfold[qi] < 2:
        continue
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    ans_raw = q.get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    pool, FR, _ = build(qi)
    s = rk.predict(FR)
    order = [pool[i] for i in np.argsort(-s)]
    cos_order = [pool[i] for i in np.argsort(-(D[pool] @ X[qi]))]
    nt += 1
    for k2 in (1, 3, 5):
        for od, acc in ((order, res), (cos_order, resc)):
            hit = any(len(ans_stems & stems_of(RAWS[i])) / len(ans_stems) >= 0.5 for i in od[:k2])
            if hit:
                acc[k2] += 1
    if qi % 100 == 0:
        P("  %d %.0fs" % (qi, time.time() - t0))

P("\n===== 200题标定 → 300题泛化 (n=%d, haystack零重叠) =====" % nt)
P("零标定裸cos: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * resc[1] / nt, 100.0 * resc[3] / nt, 100.0 * resc[5] / nt))
P("200题标定:   top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res[1] / nt, 100.0 * res[3] / nt, 100.0 * res[5] / nt))
P("F115_DONE %.0fs" % (time.time() - t0))
