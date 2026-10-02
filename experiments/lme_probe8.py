# -*- coding: utf-8 -*-
"""lme_probe8.py — 原子句粒度五信号补测: G1/问号/时态/256优势/三团
金会话内容句 vs 陪衬会话原子句(LoCoMo金vs噪的原子句版)"""
import io, json, os, re, sys, time, hashlib
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

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
COURT = re.compile(r"\b(hi|hey|hello|thanks|thank you|great|awesome|cool|nice|sure|okay|ok|wow|sounds good|good to know)\b", re.I)

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
P("loaded %.0fs" % (time.time() - t0))

def atoms_of(recs, rng, per_q=25):
    out = []
    for i in recs:
        for piece in re.split(r"[.!?]+", RAWS[i]):
            p = piece.strip()
            nw = len(p.split())
            if 4 <= nw <= 45:
                out.append(p)
    if len(out) > per_q:
        out = [out[j] for j in rng.choice(len(out), size=per_q, replace=False)]
    return out

rng = np.random.RandomState(0)
gold_s, noise_s = [], []   # (qi, text)
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    nr = []
    for k2 in hay_keys:
        if k2 not in set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or [])):
            nr += SID2ROWS.get(k2, [])
    if not gs or len(nr) < 20:
        continue
    grec = rng.choice(sorted(gs), size=min(12, len(gs)), replace=False)
    nrec = rng.choice(sorted(set(nr)), size=min(12, len(nr)), replace=False)
    for t_ in atoms_of(grec, rng):
        gold_s.append((qi, t_))
    for t_ in atoms_of(nrec, rng):
        noise_s.append((qi, t_))
rng.shuffle(noise_s)
noise_s = noise_s[:len(gold_s)]
P("金句=%d 陪衬句=%d %.0fs" % (len(gold_s), len(noise_s), time.time() - t0))

allt = [x[1] for x in gold_s] + [x[1] for x in noise_s]
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
V1024 = []
for st in range(0, len(allt), 64):
    V1024.append(np.asarray(bge.encode(allt[st:st + 64])["dense_vecs"], dtype=np.float32))
V1024 = l2n(np.concatenate(V1024))
P("bge embedded %.0fs" % (time.time() - t0))

import urllib.request
def oemb_big(texts, dims=256, bs=200):
    out = []
    for st in range(0, len(texts), bs):
        body = json.dumps({"model": "qwen3-embedding:latest", "input": texts[st:st + bs],
                           "dimensions": dims}).encode()
        for attempt in range(4):
            req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                break
            except Exception as e:
                P("retry %d %s" % (attempt + 1, str(e)[:60]))
                if attempt == 3:
                    raise
                time.sleep(8 * (attempt + 1))
        if st % 2000 == 0:
            P("  256 %d/%d %.0fs" % (st, len(texts), time.time() - t0))
    return l2n(np.concatenate(out))

V256 = oemb_big(allt)
P("256 embedded %.0fs" % (time.time() - t0))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

NG = len(gold_s)
P("\n===== 原子句粒度五信号 (金句n=%d vs 陪衬句n=%d) =====" % (NG, len(noise_s)))
# ① G1
g1p, g1n = [], []
qmark_p, qmark_n, past_p, past_n = [], [], [], []
dqc_p, dqc_n, vqc_p, vqc_n = [], [], [], []
for idx, (qi, txt) in enumerate(gold_s):
    qst = set(stem(w) for w in re.findall(r"[a-z']+", d[qi]["question"].lower()) if w not in QSTOP and len(w) > 2)
    rsts = set(stem(w) for w in re.findall(r"[a-z']+", txt.lower()))
    g1p.append(len(qst & rsts) / max(1, len(qst)))
    qmark_p.append(1.0 if txt.rstrip().endswith("?") else 0.0)
    past_p.append(1.0 if sum(1 for w in rsts if w.endswith("ed")) > 0 else 0.0)
    dqc_p.append(float(V1024[idx] @ X[qi]))
    vqc_p.append(float(V256[idx] @ Q256[qi]))
for idx, (qi, txt) in enumerate(noise_s):
    qst = set(stem(w) for w in re.findall(r"[a-z']+", d[qi]["question"].lower()) if w not in QSTOP and len(w) > 2)
    rsts = set(stem(w) for w in re.findall(r"[a-z']+", txt.lower()))
    g1n.append(len(qst & rsts) / max(1, len(qst)))
    qmark_n.append(1.0 if txt.rstrip().endswith("?") else 0.0)
    past_n.append(1.0 if sum(1 for w in rsts if w.endswith("ed")) > 0 else 0.0)
    dqc_n.append(float(V1024[NG + idx] @ X[qi]))
    vqc_n.append(float(V256[NG + idx] @ Q256[qi]))

P("①G1: AUC=%.3f (LoCoMo 0.781/LME记录粒度0.584)" % auc(g1p, g1n))
P("②问号: 金=%.3f 噪=%.3f (LoCoMo金9.6%%<噪34.2%%)" % (np.mean(qmark_p), np.mean(qmark_n)))
P("③时态ed: 金=%.3f 噪=%.3f" % (np.mean(past_p), np.mean(past_n)))
P("④256vs1024: AUC(256)=%.3f  AUC(1024)=%.3f (LoCoMo 256强9倍/LME记录粒度持平)" % (
    auc(vqc_p, vqc_n), auc(dqc_p, dqc_n)))
# ⑤ 三团: Jaccard配对
TK = [toks(t) for t in allt]
def pair_jac(a, b):
    return len(TK[a] & TK[b]) / max(1, len(TK[a] | TK[b]))
gg, gn_, nn = [], [], []
rng2 = np.random.RandomState(1)
for _ in range(3000):
    a, b = rng2.randint(0, NG, 2)
    gg.append(pair_jac(a, b))
    a, b = rng2.randint(0, NG, 2)
    gn_.append(pair_jac(a, NG + b))
    a, b = rng2.randint(0, len(noise_s), 2)
    nn.append(pair_jac(NG + a, NG + b))
P("⑤三团: 金金=%.3f 金噪=%.3f 噪噪=%.3f (LoCoMo: 0.563/0.077/0.128)" % (
    np.mean(gg), np.mean(gn_), np.mean(nn)))
P("done %.0fs" % (time.time() - t0))
