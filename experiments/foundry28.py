# -*- coding: utf-8 -*-
"""foundry28.py — 通用解验证: 无监督重定价规则 vs GBDT压缩器 vs FINAL
R = (0.4·qcov + 0.3·lenratio_norm + 0.2·d1024 + 0.1·非问句) − 2.0·回声
LOCO协议, all-gold@5/15/30 + any@30。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry28_results.txt"
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
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
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

K = (5, 15, 30)
RULES = {
    "R1_对称覆盖": lambda jac, qcov, ccov, qmark, iecho, lr, vqc, dqc: 0.5 * qcov + 0.5 * ccov,
    "R2_非对称":   lambda jac, qcov, ccov, qmark, iecho, lr, vqc, dqc: qcov - 0.5 * ccov + 0.3 * min(lr, 4) / 4,
    "R3_内容承载": lambda jac, qcov, ccov, qmark, iecho, lr, vqc, dqc: qcov + 0.3 * min(lr, 4) / 4 - 1.5 * qmark,
    "R4_全项":     lambda jac, qcov, ccov, qmark, iecho, lr, vqc, dqc: 0.5 * qcov + 0.2 * min(lr, 4) / 4 - 1.5 * qmark - iecho + 0.3 * vqc,
    "R5_全项+v1024": lambda jac, qcov, ccov, qmark, iecho, lr, vqc, dqc: 0.5 * qcov + 0.2 * min(lr, 4) / 4 - 1.5 * qmark - iecho + 0.3 * vqc + 0.3 * dqc,
}
res = {r: [0] * 3 for r in RULES}
anyr = {r: 0 for r in RULES}
res["FINAL"] = [0] * 3
anyr["FINAL"] = 0
n = 0
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
        continue
    n += 1
    qtok = toks(Q[qa]["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    pool = set(np.argsort(-C0[qi])[:300]) | set(np.argsort(-FINAL[qi])[:250])
    for i in list(np.argsort(-FINAL[qi]))[:50]:
        pool.add(max(0, i - 1))
        pool.add(min(NR - 1, i + 1))
    pool = sorted(pool)
    for rname, fn in RULES.items():
        sc = {}
        for c in pool:
            ct = ctoks(c)
            inter = qtok & ct
            jac = len(inter) / max(1, len(qtok | ct))
            qcov = len(inter) / qlen
            ccov = len(inter) / max(1, len(ct))
            qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
            iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
            lr = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
            vqc = float(qv256 @ QW[c])
            dqc = float(D[c] @ qv1024)
            sc[c] = fn(jac, qcov, ccov, qmark, iecho, lr, vqc, dqc)
        o = sorted(sc, key=sc.get, reverse=True)
        rks = [o.index(i) + 1 if i in o else 10**9 for i in G]
        for k_j, k in enumerate(K):
            if max(rks) <= k:
                res[rname][k_j] += 1
        if min(rks) <= 30:
            anyr[rname] += 1
    forder = [i for i in np.argsort(-FINAL[qi])[:30]]
    for k_j, k in enumerate(K):
        if all(r in forder[:k] for r in G):
            res["FINAL"][k_j] += 1
    if all(r in set(forder) for r in G):
        anyr["FINAL"] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== 通用解验证 (n=%d, 池~550, 全LOCO免训=规则无学习) =====" % n)
for rname in list(RULES) + ["FINAL"]:
    a, b, c = res[rname]
    P("%-14s all@5=%.1f%% all@15=%.1f%% all@30=%.1f%% any@30=%.1f%%" % (
        rname, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n), 100.0*anyr[rname]/max(1,n)))
P("FOUNDRY28_DONE %.0fs" % (time.time() - t0))
LOG.close()
