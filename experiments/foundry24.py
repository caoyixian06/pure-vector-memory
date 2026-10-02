# -*- coding: utf-8 -*-
"""foundry24.py — A漏网题桥宽分布 B锚点-token扩展抵消连乘损耗
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry24_results.txt"
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
    return set(w for w in re.findall(r"[a-z]{4,}", str(s).lower()))

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
ISRAW = np.array([(json.loads(l).get("kind") or "summary") == "raw"
                  for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()])

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
def gold_set_raw(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if ISRAW[i] and any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set_raw(qa) for qa in IDS]
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
P("DF表=%d词 %.0fs" % (len(DF), time.time() - t0))
P("loaded %.0fs" % (time.time() - t0))

# ===== A. 漏网题的桥宽分布 =====
P("\n===== A. 桥宽分布 =====")
import collections
ow = collections.Counter()
miss_examples = []
nq = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if not G:
        continue
    nq += 1
    qi = IDX[qa]
    pool = set(np.argsort(-C0[qi])[:828])
    if G <= pool:
        continue
    qtok = toks(Q[qa]["question"])
    worst = 99
    worst_rec = ""
    for i in G:
        ov = len(qtok & toks(RAW[i]))
        if ov < worst:
            worst = ov
            worst_rec = RAW[i]
    ow[worst] += 1
    if len(miss_examples) < 5:
        miss_examples.append((qa, worst, worst_rec[:90], Q[qa].get("question", "")[:70]))
tot = sum(ow.values())
P("漏网题=%d/%d (%.1f%%)" % (tot, nq, 100.0 * tot / max(1, nq)))
P("桥宽分布(最差片的题目词重叠): ")
for w in sorted(ow):
    P("  overlap=%d: %d题 (%.1f%%)" % (w, ow[w], 100.0 * ow[w] / tot))
P("样例:")
for qa, w, rec, q in miss_examples:
    P("  [%s] 桥宽=%d\n    Q: %s\n    E: %s" % (qa, w, q, rec))

# ===== B. 锚点-token扩展(多片题) =====
P("\n===== B. 锚点扩展 =====")
def idf_tokens(text, qtok):
    ws = [w for w in re.findall(r"[a-z]{4,}", text.lower())]
    ws = [w for w in ws if w not in qtok]
    cnt = {}
    for w in ws:
        k = norm(w)
        hosts = DF.get(k, 0)
        if hosts <= 50:
            cnt[w] = 60 - min(59, hosts // 2)
    return cnt

res = {"base30": 0, "exp30": 0, "base_any": 0, "exp_any": 0}
n_multi = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if len(G) < 2:
        continue
    n_multi += 1
    qi = IDX[qa]
    order = list(np.argsort(-C0[qi]))
    pool = order[:200]
    qtok = toks(Q[qa]["question"])
    # 锚: 池内FINAL第一名raw
    anchor = None
    for i in order:
        if ISRAW[i]:
            anchor = i
            break
    salient = {}
    if anchor is not None:
        salient = idf_tokens(RAW[anchor], qtok)
    salset = set(salient)
    # 扩展分: 池外高锚词覆盖记录
    exp_pool = {}
    for i in order[:200]:
        exp_pool[i] = float(FINAL[qi][i])
    if salset:
        for i in range(NR):
            if i in exp_pool:
                continue
            rn = set(w for w in re.findall(r"[a-z]{4,}", RAW[i].lower()))
            cov = len(salset & rn) / max(1, len(salset))
            if cov >= 0.5:
                exp_pool[i] = 5.0 + 10.0 * cov
    order_exp = sorted(exp_pool, key=exp_pool.get, reverse=True)
    for tag, od, kk in (("base", order[:30], 30), ("exp", order_exp[:30], 30)):
        ok = all(r in set(od[:kk]) for r in G)
        if tag == "base":
            res["base30"] += ok
            res["base_any"] += 1
        else:
            res["exp30"] += ok
            res["exp_any"] += 1
    if len(G) & 1:
        pass
P("多片题(n=%d): " % n_multi)
P("  FINAL基线  all-gold@30 = %d" % res["base30"])
P("  锚点扩展   all-gold@30 = %d" % res["exp30"])
P("  扩展净增 = %+d题" % (res["exp30"] - res["base30"]))
P("FOUNDRY24_DONE %.0fs" % (time.time() - t0))
LOG.close()
