# -*- coding: utf-8 -*-
"""foundry37.py — 逐题对比: 每个多片题的金金vs噪噪分离比, 找出全部反例
每题: 金金对均值Jaccard / 噪噪对均值Jaccard / 分离比; 分离比<1 = 违反规律(反例)
逐题输出全部, 落盘 + 统计反例数量和分布, 分题型看反例聚集在哪
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry37_results.txt"
OUTD = HERE + "/foundry37_detail.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
LOGD = io.open(OUTD, "w", encoding="utf-8")
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

t0 = time.time()
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
C0 = l2n(X @ D.T)
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
P("loaded %.0fs" % (time.time() - t0))

TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

def pair_stats(pairs):
    """pairs: [(a,b)] -> mean jaccard/256/1024"""
    js, vs, ds = [], [], []
    for a, b in pairs:
        ta, tb = ctoks(a), ctoks(b)
        js.append(len(ta & tb) / max(1, len(ta | tb)))
        vs.append(float(QW[a] @ QW[b]))
        ds.append(float(D[a] @ D[b]))
    if not js:
        return None
    return (np.mean(js), np.mean(vs), np.mean(ds), len(js))

import itertools
per_q = []
viol = []
viol_by_cat = {}
n = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if len(G) < 2:
        continue
    qi = IDX[qa]
    cat = Q[qa].get("category")
    pool = [i for i in np.argsort(-C0[qi])[:300]]
    golds = sorted(G)
    noises = [i for i in pool if i not in G]
    if len(noises) < 2:
        continue
    n += 1
    gg = list(itertools.combinations(golds, 2))[:10]
    nn = [(noises[i], noises[i + 1]) for i in range(0, min(len(noises) - 1, 20), 2)][:10]
    sgg = pair_stats(gg)
    snn = pair_stats(nn)
    if sgg is None or snn is None:
        continue
    ratio = sgg[0] / max(1e-9, snn[0])
    ok = sgg[0] > snn[0]
    per_q.append((qa, cat, sgg, snn, ratio, ok))
    if not ok:
        viol.append((qa, cat, sgg, snn, ratio))
        viol_by_cat[cat] = viol_by_cat.get(cat, 0) + 1
    LOGD.write("%s cat=%s 金金Jac=%.3f 噪噪Jac=%.3f 比=%.2f %s | 金金256=%.3f 噪噪256=%.3f | 金金1024=%.3f 噪噪1024=%.3f\n" % (
        qa, cat, sgg[0], snn[0], ratio, "OK" if ok else "VIOLATION",
        sgg[1], snn[1], sgg[2], snn[2]))
LOGD.close()

P("多片题=%d" % n)
P("\n===== 分题型逐题统计 =====")
cats = sorted(set(p[1] for p in per_q))
for cat in cats:
    sub = [p for p in per_q if p[1] == cat]
    okc = sum(1 for p in sub if p[5])
    ratios = [p[4] for p in sub]
    P("cat%s: n=%d 符合=%d(%.0f%%) 违反=%d | 分离比中位=%.2f 最小=%.2f" % (
        cat, len(sub), okc, 100.0 * okc / len(sub), len(sub) - okc,
        np.median(ratios), min(ratios)))
P("\n===== 全部反例清单(金金<=噪噪) =====")
for qa, cat, sgg, snn, ratio in viol:
    P("[%s] cat=%s 比=%.2f | 金金Jac=%.3f 噪噪Jac=%.3f | 256: %.3f vs %.3f | 1024: %.3f vs %.3f" % (
        qa, cat, ratio, sgg[0], snn[0], sgg[1], snn[1], sgg[2], snn[2]))
P("\n反例分布: " + str(viol_by_cat))
P("FOUNDRY37_DONE %.0fs" % (time.time() - t0))
LOG.close()
