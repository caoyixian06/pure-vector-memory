# -*- coding: utf-8 -*-
"""lme_probe9.py — 四死信号的原因验尸+补救:
①三团改cos配对(句长压缩假说: Jaccard死cos活=方法错配) ②问号保留标点重切(切分虫假说)
③G1: 均值分布+IDF加权版(分辨率假说) ④时态密度化"""
import io, json, os, re, sys, time, hashlib
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

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

# IDF表(全库词干)
DF = {}
for r in RAWS:
    for w in set(stem(x) for x in re.findall(r"[a-z']+", r.lower()) if len(x) > 2):
        DF[w] = DF.get(w, 0) + 1
NLIB = len(RAWS)
P("loaded %.0fs" % (time.time() - t0))

def atoms_keep_punct(rec):
    """保留句尾标点的切分(修probe8的问号虫)"""
    out = []
    for m in re.finditer(r"[^.!?]*[.!?]?", RAWS[rec]):
        p = m.group(0).strip()
        nw = len(p.split())
        if 4 <= nw <= 45:
            out.append(p)
    return out

rng = np.random.RandomState(0)
gold_s, noise_s = [], []
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    gold_keys = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    nr = []
    for k2 in set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h):
        if k2 not in gold_keys:
            nr += SID2ROWS.get(k2, [])
    if not gs or len(nr) < 20:
        continue
    grec = rng.choice(sorted(gs), size=min(12, len(gs)), replace=False)
    nrec = rng.choice(sorted(set(nr)), size=min(12, len(nr)), replace=False)
    ga = atoms_keep_punct(int(grec[0])) if len(grec) else []
    for i in grec:
        ga += atoms_keep_punct(i)
    na = []
    for i in nrec:
        na += atoms_keep_punct(i)
    if len(ga) > 25:
        ga = [ga[j] for j in rng.choice(len(ga), size=25, replace=False)]
    if len(na) > 25:
        na = [na[j] for j in rng.choice(len(na), size=25, replace=False)]
    for t_ in ga:
        gold_s.append((qi, t_))
    for t_ in na:
        noise_s.append((qi, t_))
rng.shuffle(noise_s)
noise_s = noise_s[:len(gold_s)]
P("金句=%d 陪衬句=%d (保留标点版) %.0fs" % (len(gold_s), len(noise_s), time.time() - t0))

allt = [x[1] for x in gold_s] + [x[1] for x in noise_s]
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
V = []
for st in range(0, len(allt), 64):
    V.append(np.asarray(bge.encode(allt[st:st + 64])["dense_vecs"], dtype=np.float32))
V = l2n(np.concatenate(V))
P("embedded %.0fs" % (time.time() - t0))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

NG = len(gold_s)
# ② 问号(保留标点): 金vs噪的问句率
qm_p = [1.0 if t.rstrip().endswith("?") else 0.0 for _, t in gold_s]
qm_n = [1.0 if t.rstrip().endswith("?") else 0.0 for _, t in noise_s]
P("\n②问号(标点保留): 金=%.3f 噪=%.3f  AUC=%.3f" % (
    np.mean(qm_p), np.mean(qm_n), auc(qm_p, qm_n)))

# ③ G1: 原版 + IDF加权 + 均值分布
g1p, g1n, g1w_p, g1w_n = [], [], [], []
for idx, (qi, txt) in enumerate(gold_s):
    qst = set(stem(w) for w in re.findall(r"[a-z']+", d[qi]["question"].lower()) if w not in QSTOP and len(w) > 2)
    rsts = set(stem(w) for w in re.findall(r"[a-z']+", txt.lower()))
    inter = qst & rsts
    g1p.append(len(inter) / max(1, len(qst)))
    g1w_p.append(sum(np.log(NLIB / DF[w]) for w in inter if DF.get(w)) / max(1e-9, sum(np.log(NLIB / DF[w]) for w in qst if DF.get(w))))
for idx, (qi, txt) in enumerate(noise_s):
    qst = set(stem(w) for w in re.findall(r"[a-z']+", d[qi]["question"].lower()) if w not in QSTOP and len(w) > 2)
    rsts = set(stem(w) for w in re.findall(r"[a-z']+", txt.lower()))
    inter = qst & rsts
    g1n.append(len(inter) / max(1, len(qst)))
    g1w_n.append(sum(np.log(NLIB / DF[w]) for w in inter if DF.get(w)) / max(1e-9, sum(np.log(NLIB / DF[w]) for w in qst if DF.get(w))))
P("③G1原版: 金=%.3f 噪=%.3f AUC=%.3f | IDF加权: AUC=%.3f" % (
    np.mean(g1p), np.mean(g1n), auc(g1p, g1n), auc(g1w_p, g1w_n)))

# ④ 时态密度(ed词数/词数)
td_p = [sum(1 for w in re.findall(r"[a-z']+", t.lower()) if w.endswith("ed")) / max(1, len(t.split())) for _, t in gold_s]
td_n = [sum(1 for w in re.findall(r"[a-z']+", t.lower()) if w.endswith("ed")) / max(1, len(t.split())) for _, t in noise_s]
P("④时态密度: 金=%.4f 噪=%.4f AUC=%.3f" % (np.mean(td_p), np.mean(td_n), auc(td_p, td_n)))

# ① 三团: BGE cos配对 (金金=同题金句对/金噪/噪噪)
rng2 = np.random.RandomState(1)
gg, gn_, nn = [], [], []
NN = len(noise_s)
for _ in range(4000):
    a, b = rng2.randint(0, NG, 2)
    gg.append(float(V[a] @ V[b]))
for _ in range(4000):
    a, b = rng2.randint(0, NG, 2)
    gn_.append(float(V[a] @ V[NG + b]))
for _ in range(4000):
    a, b = rng2.randint(0, NN, 2)
    nn.append(float(V[NG + a] @ V[NG + b]))
P("①三团cos: 金金=%.3f 金噪=%.3f 噪噪=%.3f (记录粒度256版: 0.502/0.239/0.241)" % (
    np.mean(gg), np.mean(gn_), np.mean(nn)))
P("done %.0fs" % (time.time() - t0))
