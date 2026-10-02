# -*- coding: utf-8 -*-
"""lme_probe4.py — 新规律候选LME头部验证: 专名回声/会话位置/引号/时态回声"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

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
SESS_FIRST = {}
for si, h in enumerate(hashes):
    sid = "lme-s" + h[:12]
    SESS_FIRST[sid] = len(RAWS)
    RAWS.append("[hdr]")
    SIDS.append(sid)
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
        SIDS.append(sid)
NR = len(RAWS)
SID2ROWS = {}
SESS_LEN = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
    SESS_LEN[s] = SESS_LEN.get(s, 0) + 1
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
P("loaded %.0fs" % (time.time() - t0))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

FN = ["cap_match", "relpos", "quote", "past_match", "cap_dens"]
POS = {k: [] for k in FN}
NEG = {k: [] for k in FN}
rng = np.random.RandomState(0)
for qi, q in enumerate(d):
    gold_rows = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gold_rows |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    if not gold_rows:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows = np.array(sorted(set(r for k2 in hay_keys for r in SID2ROWS.get(k2, []))))
    if len(hay_rows) < 100:
        continue
    cq = D[hay_rows] @ X[qi] + V256[hay_rows] @ Q256[qi]
    order = hay_rows[np.argsort(-cq)[:48]]
    qtext = q["question"]
    qcaps = set(re.findall(r"[A-Z][a-z]+", qtext))
    q_past = bool(re.search(r"\bdid\b|\bwas\b|\bwere\b|\bhave\b", qtext, re.I))
    gsel = rng.choice(sorted(gold_rows), size=min(5, len(gold_rows)), replace=False).tolist()
    head_noise = [i for i in order if i not in gold_rows][:40]
    def fv(i):
        r = RAWS[i]
        words = r.split()
        nw = max(1, len(words))
        rcaps = set(re.findall(r"[A-Z][a-z]+", r))
        toks_ = re.findall(r"[a-z']+", r.lower())
        return {
            "cap_match": len(qcaps & rcaps) / max(1, len(qcaps)) if qcaps else 0.0,
            "relpos": (i - SESS_FIRST[SIDS[i]]) / max(1, SESS_LEN[SIDS[i]] - 1),
            "quote": 1.0 if '"' in r else 0.0,
            "past_match": 1.0 if (q_past and sum(1 for w in toks_ if w.endswith("ed")) > 0) else 0.0,
            "cap_dens": len(rcaps) / nw,
        }
    for i in gsel:
        f = fv(i)
        for k in FN:
            POS[k].append(f[k])
    for i in head_noise:
        f = fv(i)
        for k in FN:
            NEG[k].append(f[k])

P("金n=%d 头部噪n=%d" % (len(POS[FN[0]]), len(NEG[FN[0]])))
P("\n===== 新候选LME头部验证 =====")
for k in FN:
    a = auc(POS[k], NEG[k])
    P("%-12s AUC=%.3f  金=%.3f 噪=%.3f  (LoCoMo: cap_match反向0.218/relpos反向0.370/quote 0.045)" % (
        k, a, np.mean(POS[k]), np.mean(NEG[k])))
P("done %.0fs" % (time.time() - t0))
