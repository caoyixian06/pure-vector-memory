# -*- coding: utf-8 -*-
"""foundry119b.py — 清洗版伪标注ranker(用户: 标注贴近发现+LoCoMo自带标注的启示):
伪标签 = answer词干覆盖≥0.5(正例) + 寒暄排除 + 长度分位
A半训练 -> B半测试(零重叠), 三臂: 裸cos / 纯LoCoMo直迁 / 伪标注ranker / 混训"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry119b_results.txt", "w", encoding="utf-8")
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
COURT = re.compile(r"\b(hi|hey|hello|thanks|thank you|great|awesome|cool|nice|sure|okay|ok|wow|sounds good|good to know)\b", re.I)

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
def hay_key(q):
    return hashlib.md5(json.dumps(sorted(q["haystack_session_ids"]), ensure_ascii=False).encode()).hexdigest()
hk = {}
for qi, q in enumerate(d):
    hk.setdefault(hay_key(q), []).append(qi)
keys = sorted(hk)
fd = {k: i % 2 for i, k in enumerate(keys)}
qfold = np.array([fd[hay_key(q)] for q in d])
P("LME loaded %.0fs" % (time.time() - t0))

# 全题特征+伪标签构造
POOL, FEAT, PSTAR = {}, {}, {}
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
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    qtok = toks(qtext)
    Fm = np.zeros((len(pool), 12), dtype=np.float32)
    star = np.zeros(len(pool), dtype=np.int8)
    for rr, c in enumerate(pool):
        rt = RAWS[c]
        Fm[rr] = feats12(qtext, rt, X[qi], Q256[qi], D[c], V256[c], qstems, qstems_list, qtok, RTOK[c])
        # 清洗版伪标签(用户: 发现驱动清洗): 金会话内 非寒暄+非超短 = 正例
        # (不按词干筛——改写句不会被误标负例, 修F119b病根)
        is_chatty = (len(rt.split()) <= 6 and COURT.search(rt)) or len(rt.split()) <= 4
        if not is_chatty:
            star[rr] = 1
    POOL[qi] = pool
    FEAT[qi] = pool_rank(Fm)
    PSTAR[qi] = star
P("特征+伪标签 %.0fs" % (time.time() - t0))

import lightgbm as lgb
def mk():
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.08,
                          num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
                          random_state=0, verbosity=-1, n_jobs=4)

# A半训练
trF, trY, trG = [], [], []
nA = npos = 0
for qi in sorted(POOL):
    if qfold[qi] != 0:
        continue
    star = PSTAR[qi]
    pos = [rr for rr in range(len(star)) if star[rr]][:8]
    pset = set(pos)
    noise = [rr for rr in range(min(48, len(star))) if rr not in pset][:40]
    for rr in pos + noise:
        trF.append(FEAT[qi][rr])
        trY.append(1 if rr in pset else 0)
    trG.append(len(pos) + len(noise))
    nA += 1
    npos += len(pos)
rk = mk()
rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
P("伪标注ranker: A半%d题, 正例%d条(均%.1f/题) %.0fs" % (nA, npos, npos / max(1, nA), time.time() - t0))

# B半测试
res = {1: 0, 3: 0, 5: 0}
resc = {1: 0, 3: 0, 5: 0}
nt = 0
for qi in sorted(POOL):
    if qfold[qi] != 1:
        continue
    ans_raw = d[qi].get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    pool = POOL[qi]
    nt += 1
    or_ = [pool[i] for i in np.argsort(-rk.predict(FEAT[qi]))]
    oc = [pool[i] for i in np.argsort(-(D[pool] @ X[qi]))]
    for k2 in (1, 3, 5):
        for od, acc in ((or_, res), (oc, resc)):
            if any(len(ans_stems & stems_of(RAWS[i])) / len(ans_stems) >= 0.5 for i in od[:k2]):
                acc[k2] += 1

P("\n===== LME B半 turn级(n=%d) =====" % nt)
P("裸cos:          top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * resc[1] / nt, 100.0 * resc[3] / nt, 100.0 * resc[5] / nt))
P("清洗版伪标注ranker: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
    100.0 * res[1] / nt, 100.0 * res[3] / nt, 100.0 * res[5] / nt))
P("F119B_DONE %.0fs" % (time.time() - t0))
