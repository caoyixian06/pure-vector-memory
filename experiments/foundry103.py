# -*- coding: utf-8 -*-
"""foundry103.py — LoCoMo的ed时态向量街区derive(用户问:时态在哪个维度存在):
高ed记录vs零ed记录的256/1024维Δ + 最强维 + 金噪投影差 + 与时间词街区(d251族)的重叠"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
QW = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
X = np.load(HERE + "/xz_cache.npz")["X"]

# ed密度分组(排除时间词干扰: 去掉含时间词的记录,让Δ纯ed)
TW = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
ed_dens = np.array([sum(1 for w in re.findall(r"[a-z']+", r.lower()) if w.endswith("ed")) / max(1, len(r.split())) for r in RAW])
no_tw = np.array([not TW.search(r) for r in RAW])
hi = ed_dens > np.quantile(ed_dens[no_tw], 0.9)
lo = (ed_dens == 0) & no_tw
hi = hi & no_tw
P("高ed(纯)=%d 零ed(纯)=%d %.0fs" % (hi.sum(), lo.sum(), time.time() - t0))

# 256街区
delta256 = QW[hi].mean(0) - QW[lo].mean(0)
street = np.where(np.abs(delta256) > 0.01)[0]
top10 = np.argsort(-np.abs(delta256))[:10]
P("\n===== LoCoMo的ed时态街区(256空间) =====")
P("街区维数(|Δ|>0.01): %d  最强维: d%d(Δ=%.3f)" % (len(street), top10[0], delta256[top10[0]]))
P("top10: %s" % ",".join("d%d" % x for x in top10))
# 与时间词街区的重叠
DATE_WORDS = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
is_dated = np.array([bool(DATE_WORDS.search(r)) for r in RAW])
tw_delta = QW[is_dated].mean(0) - QW[~is_dated].mean(0)
tw_street = set(np.where(np.abs(tw_delta) > 0.01)[0].tolist())
inter = tw_street & set(street.tolist())
P("与时间词街区(%d维,最强d251族)重叠: %d维 (重叠率%.1f%%)" % (
    len(tw_street), len(inter), 100.0 * len(inter) / max(1, len(street))))
P("时间词街区top5: %s" % ",".join("d%d" % x for x in np.argsort(-np.abs(tw_delta))[:5]))

# 1024街区
delta1024 = D[hi].mean(0) - D[lo].mean(0)
st1024 = np.where(np.abs(delta1024) > 0.01)[0]
P("\n===== 1024空间 =====")
P("街区维数: %d  最强维: d%d(Δ=%.3f)" % (len(st1024), np.argmax(np.abs(delta1024)), np.abs(delta1024).max()))

# 金噪在ed方向的投影差(全库条件)
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))
u256 = delta256 / max(np.linalg.norm(delta256), 1e-9)
pg, pn = [], []
rng = np.random.RandomState(0)
for qi_ in range(0, len(IDS), 3):
    qa = IDS[qi_]
    g = gold_set(qa)
    if not g:
        continue
    ng = rng.choice([i for i in range(len(MID)) if i not in g], size=min(30, len(MID)), replace=False)
    for i in list(g)[:10]:
        pg.append(float(QW[i] @ u256))
    for i in ng:
        pn.append(float(QW[i] @ u256))
from bisect import bisect_left
def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    return (sum(bisect_left(allv, v) + 1 for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
P("\n金vs噪在ed方向的投影AUC: %.3f (金=%.4f 噪=%.4f)" % (auc(pg, pn), np.mean(pg), np.mean(pn)))
P("done %.0fs" % (time.time() - t0))
