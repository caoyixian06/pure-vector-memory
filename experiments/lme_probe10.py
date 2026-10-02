# -*- coding: utf-8 -*-
"""lme_probe10.py — G1的会话级max聚合(稀疏复活假说): 金会话max-G1 vs 陪衬会话max-G1"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

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
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)

rng = np.random.RandomState(0)
def sess_max_g1(qstems, rows):
    best = 0.0
    for i in rows:
        rsts = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()))
        g = len(qstems & rsts) / max(1, len(qstems))
        if g > best:
            best = g
    return best

gp, np_ = [], []
for qi, q in enumerate(d):
    gold_keys = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    noise_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h) - gold_keys
    if not gold_keys or not noise_keys:
        continue
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    for k in gold_keys:
        gp.append(sess_max_g1(qstems, SID2ROWS[k]))
    for k in rng.choice(sorted(noise_keys), size=min(3, len(noise_keys)), replace=False):
        np_.append(sess_max_g1(qstems, SID2ROWS[k]))
    if qi >= 400:
        break

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

P("金会话max-G1=%.3f (n=%d)  陪衬会话max-G1=%.3f (n=%d)  AUC=%.3f" % (
    np.mean(gp), len(gp), np.mean(np_), len(np_), auc(gp, np_)))
P("句级版对照: AUC=-0.021 | 判定: >0.7=稀疏复活成立")
P("done %.0fs" % (time.time() - t0))
