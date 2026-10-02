# -*- coding: utf-8 -*-
"""foundry57.py — 具体案例剖析: 用户指认的"应该可分"的4对真金假金
对每对, 全量打印双方的全部信号值(12普适+词法+三形态), 找出到底哪些维度分得开
"""
import io, json, os, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry57_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w

MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

def find_rec(substr):
    ns = norm(substr)
    for i, rn in enumerate(RAWN):
        if ns[:50] in rn:
            return i
    return None

def full_sig(qa, c):
    qi = IDX[qa]
    q = Q[qa]
    ql = q["question"].lower()
    qwords = [w for w in re.findall(r"[a-z']+", ql) if w not in QSTOP and len(w) > 2]
    qstems = set(stem(w) for w in qwords)
    ct = toks(RAW[c])
    qtok = toks(q["question"])
    inter = qtok & ct
    jac = len(inter) / max(1, len(qtok | ct))
    qcov = len(inter) / max(1, len(qtok))
    vqc = float(Q256[qi] @ QW[c])
    dqc = float(D[c] @ X[qi])
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
    g1 = len(qstems & rstems) / max(1, len(qstems))
    inter_stems = sorted(qstems & rstems)
    dif256 = np.abs(Q256[qi] - QW[c])
    dif1024 = np.abs(X[qi] - D[c])
    return {
        "问题": q["question"][:70],
        "文本": RAW[c][:90],
        "Jaccard": round(jac, 3),
        "qcov": round(qcov, 3),
        "256cos": round(vqc, 3),
        "1024cos": round(dqc, 3),
        "G1词干呼应": round(g1, 3),
        "呼应的词": inter_stems[:6],
        "256差均值": round(float(dif256.mean()), 4),
        "1024差均值": round(float(dif1024.mean()), 4),
        "长度": len(RAW[c]),
    }

CASES = [
    ("conv-26#q0074", "5 years already", "Hey Mel! Good to see you! How have you been"),
    ("conv-26#q0119", "Yeah, I drew it. It stands for", "Drawing flowers is one of my fav"),
    ("conv-26#q0103", "It was Matt Patterson", "That's really sweet. Is this your"),
    ("conv-26#q0104", "I'm obsessed with those, so I made", "Good to see you! I'm swamped with the"),
]

for qa, gtxt, ftxt in CASES:
    g = find_rec(gtxt)
    f = find_rec(ftxt)
    P("=" * 60)
    P("题目: %s" % Q[qa]["question"][:80])
    if g is None or f is None:
        P("定位失败 gold=%s fake=%s" % (g, f))
        continue
    sg = full_sig(qa, g)
    sf = full_sig(qa, f)
    P("%-14s %-42s 真金 → 假金" % ("信号", "数值对比"))
    for k in ("Jaccard", "qcov", "256cos", "1024cos", "G1词干呼应", "256差均值", "1024差均值", "长度"):
        mark = " ←真金高" if sg[k] > sf[k] else " ←假金高"
        P("%-14s %-42s %s" % (k, "%s → %s" % (sg[k], sf[k]), mark))
    P("真金呼应词: %s" % sg["呼应的词"])
    P("假金呼应词: %s" % sf["呼应的词"])
P("F57_DONE")
LOG.close()
