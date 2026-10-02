# -*- coding: utf-8 -*-
"""foundry12.py — 冲89%三层火箭: ①基线 ②去海(近重复塌缩) ③分头找(方面子查询并集)
靶: 严格all-gold@5(原文turn, 每片都要进)。纯向量零API。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry12_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

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
RAWN = [norm(rec_raw(m)) for m in MID]

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
C0 = l2n(X @ D.T)
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
_cw = np.load(HERE + "/cue_word_cache.npz", allow_pickle=True)
WV = _cw["V"]
WV = WV / np.maximum(np.linalg.norm(WV, axis=1, keepdims=True), 1e-9)
WORDS = [str(w).lower() for w in _cw["WORDS"]]
W2I = {w: i for i, w in enumerate(WORDS)}
P("loaded %.0fs" % (time.time() - t0))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

# ===== ①去海: 近重复塌缩 (贪心: cos>0.93的并成簇, 簇代表=最像题目质心的) =====
P("塌缩聚类中...")
sim_dd = D @ D.T
np.fill_diagonal(sim_dd, -9)
repr_of = np.arange(NR)
alive = np.ones(NR, dtype=bool)
order_deg = np.argsort(-sim_dd.max(axis=1))
for i in order_deg:
    if not alive[i]:
        continue
    grp = np.where((sim_dd[i] > 0.93) & alive)[0]
    for j in grp:
        alive[j] = False
        repr_of[j] = i
    alive[i] = True
REPR = np.unique(repr_of)
P("去海: %d → %d 条 (%.0fs)" % (NR, len(REPR), time.time() - t0))

def gold_set_repr(qa, mapping):
    return set(int(mapping[i]) for i in gold_set(qa))

# ===== ②方面子查询构造 =====
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
TIMEW = set(MONTHS) | {"when", "time", "year", "month", "week", "day", "date", "long", "ago"}
WHOw = {"who", "name", "person", "someone", "friend", "brother", "sister", "mother",
        "father", "husband", "wife", "dad", "mom", "cousin", "daughter", "son"}

def aspect_subq(qa):
    q = Q[qa]
    toks = [w for w in re.findall(r"[a-z]+", (q.get("question") or "").lower()) if len(w) > 3]
    time_t = [w for w in toks if w in TIMEW or w in W2I and False]
    time_v = [WV[W2I[w]] for w in toks if w in TIMEW and w in W2I]
    who_v = [WV[W2I[w]] for w in toks if w in WHOw and w in W2I]
    rest_v = [WV[W2I[w]] for w in toks if w not in TIMEW and w not in WHOw and w in W2I]
    subs = []
    if time_v:
        subs.append(("time", l2n(np.stack(time_v).sum(0, keepdims=True))[0]))
    if who_v:
        subs.append(("who", l2n(np.stack(who_v).sum(0, keepdims=True))[0]))
    if rest_v:
        subs.append(("rest", l2n(np.stack(rest_v).sum(0, keepdims=True))[0]))
    return subs

QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

def eval_variant(name, score_fn, mapping=None):
    a5 = a10 = a30 = n = 0
    for qa in IDS:
        G = gold_set(qa)
        if not G:
            continue
        if mapping is not None:
            G = set(mapping[i] for i in G)
        n += 1
        sc = score_fn(qa, G)
        order = sorted(sc, key=sc.get, reverse=True)
        ranks = [order.index(g) + 1 for g in G if g in order]
        if ranks and max(ranks) <= 5:
            a5 += 1
        if ranks and max(ranks) <= 10:
            a10 += 1
        if ranks and max(ranks) <= 30:
            a30 += 1
    P("%-22s all@5=%.1f%% all@10=%.1f%% all@30=%.1f%% (n=%d)" % (
        name, 100.0 * a5 / max(1, n), 100.0 * a10 / max(1, n), 100.0 * a30 / max(1, n), n))
    return 100.0 * a5 / max(1, n)

# 基线(塌缩后重映射)
def sc_base(qa, G):
    qi = IDX[qa]
    return {r: float(C0[qi][r]) for r in REPR if r in G or True}
def sc_base_fast(qa, G):
    qi = IDX[qa]
    sc = {}
    for r in REPR:
        sc[int(r)] = float(C0[qi][r])
    return sc

P("\n===== 阶梯 =====")
v1 = eval_variant("S0 基线(去海+纯cos)", sc_base_fast)

def zs(d):
    a = np.asarray(d, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / (s + 1e-9)

def sc_dedupe_aspect(qa, G, w_aspect):
    qi = IDX[qa]
    base = zs(np.array([C0[qi][r] for r in REPR]))
    sc = {int(r): float(base[i]) for i, r in enumerate(REPR)}
    subs = aspect_subq(qa)
    for tag, sv in subs:
        cs = zs(QW @ sv)
        for i, r in enumerate(REPR):
            sc[int(r)] = sc.get(int(r), 0.0) + w_aspect * float(cs[i])
    return sc
v2 = eval_variant("S1 去海+方面子查询(w=0.5)", lambda qa, G: sc_dedupe_aspect(qa, G, 0.5))
v3 = eval_variant("S2 方面w=1.0", lambda qa, G: sc_dedupe_aspect(qa, G, 1.0))
P("\nFOUNDRY12_DONE %.0fs" % (time.time() - t0))
LOG.close()
