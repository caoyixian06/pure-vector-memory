# -*- coding: utf-8 -*-
"""foundry112.py — 用户词级计数思路(会话内turn排序):
句得分 = 句中"与问题词高cos"的词数(阈值=库词对cos分位,零标签)
三臂: a句cos基线 / b词计数 / c等权z融合。验证=会话内top-k含金(对错)"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry112_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
HERE = "C:/locomo_refined/memsys"
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
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))
# 词向量表(LoCoMo word_vecs)
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WH.keys())
WV = l2n(np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
P("loaded %.0fs 词表=%d" % (time.time() - t0, len(WORDS)))

# 自适应阈值: 随机词对cos的95%分位(零标签)
rng0 = np.random.RandomState(0)
a_ = WV[rng0.choice(len(WV), 300, replace=False)]
b_ = WV[rng0.choice(len(WV), 300, replace=False)]
S = a_ @ b_.T
TAU = float(np.quantile(S, 0.95))
P("词对cos阈值(95%%分位)=%.3f" % TAU)

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

def zs(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)

res = {k: {1: 0, 3: 0, 5: 0} for k in ("sentcos", "worddens", "fuse")}
n = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    cq_all = D @ X[k_i]
    # 会话定位(cos第一的会话)
    sess_score = {}
    for cv in UCONVS:
        sess_score[cv] = float(cq_all[CONVKEY == cv].max())
    top_conv = max(sess_score, key=sess_score.get)
    rows = np.where(CONVKEY == top_conv)[0]
    if len(rows) < 5:
        continue
    # 问题词向量
    qws = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    qi_idx = [W2I[w] for w in qws if w in W2I]
    if not qi_idx:
        continue
    QV = WV[qi_idx]
    n += 1
    sc_cos, sc_wcnt = [], []
    for i in rows:
        sc_cos.append(float(D[i] @ X[k_i]))
        sws = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()) if len(w) > 2)
        sidx = [W2I[w] for w in sws if w in W2I]
        nw = max(1, len(sws))
        if sidx:
            M_ = (WV[sidx] @ QV.T).max(axis=1)
            sc_wcnt.append(float((M_ > TAU).sum()) / nw)
        else:
            sc_wcnt.append(0.0)
    sc_cos, sc_wcnt = np.asarray(sc_cos), np.asarray(sc_wcnt)
    fuse = zs(sc_cos) + zs(sc_wcnt)
    Gm = G & set(rows.tolist())
    for tag, sc in (("sentcos", sc_cos), ("worddens", sc_wcnt), ("fuse", fuse)):
        order = rows[np.argsort(-np.asarray(sc))]
        for k2 in (1, 3, 5):
            if Gm & set(order[:k2].tolist()):
                res[tag][k2] += 1
    if k_i % 400 == 0:
        P("  %d %.0fs" % (k_i, time.time() - t0))

P("\n===== 会话内turn排序三臂(n=%d, 会话=cos第一会话) =====" % n)
for tag in ("sentcos", "worddens", "fuse"):
    P("%-9s top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
        tag, 100.0 * res[tag][1] / n, 100.0 * res[tag][3] / n, 100.0 * res[tag][5] / n))
P("F112_DONE %.0fs" % (time.time() - t0))
