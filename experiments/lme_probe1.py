# -*- coding: utf-8 -*-
"""lme_probe1.py — 零标签验证#1/#2: 时间街区(两库对照) + 说话人可分性 (纯CPU, 不扰建库)
数据: LME前5块256(q0000-0004=81920条) + LME全库1024 + 源json文本 + LoCoMo库对照"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
HERE = "C:/locomo_refined/memsys"

# ===== LME Stage A 重解析(纯CPU) =====
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sess, sdate = {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    if h not in sess:
        sess[h] = turns
        sdate[h] = dt
hashes = list(sess.keys())
RAWS = []
for si, h in enumerate(hashes):
    dt = (sdate[h] or "")[:10]
    RAWS.append("[Session %d — %s]" % (si + 1, dt))
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
P("LME records=%d %.0fs" % (len(RAWS), time.time() - t0))

# ===== 验证#1: 时间街区 (LME前81920条 vs LoCoMo全库) =====
V5 = l2n(np.concatenate([np.load(OUT + "/_v256_parts/q%04d.npy" % c) for c in range(5)]))
N5 = len(V5)
DATE_PAT = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
is_dated_lme = np.array([bool(DATE_PAT.search(r)) for r in RAWS[:N5]])
P("LME样本: 含日期=%d 不含=%d" % (is_dated_lme.sum(), (~is_dated_lme).sum()))
delta_lme = V5[is_dated_lme].mean(0) - V5[~is_dated_lme].mean(0)

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
RAW_LC = [rec_raw(m) for m in MID]
QW = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
is_dated_lc = np.array([bool(DATE_PAT.search(r)) for r in RAW_LC])
P("LoCoMo样本: 含日期=%d 不含=%d" % (is_dated_lc.sum(), (~is_dated_lc).sum()))
delta_lc = QW[is_dated_lc].mean(0) - QW[~is_dated_lc].mean(0)

th = 0.01
st_lme = set(np.where(np.abs(delta_lme) > th)[0].tolist())
st_lc = set(np.where(np.abs(delta_lc) > th)[0].tolist())
P("\n===== 时间街区对照 (|Δ|>%.2f, hdr组-vs-raw组) =====" % th)
P("LoCoMo街区: %d 维  最强维 d%d(Δ=%.3f)" % (len(st_lc), int(np.argmax(np.abs(delta_lc))), np.abs(delta_lc).max()))
P("LME街区:   %d 维  最强维 d%d(Δ=%.3f)" % (len(st_lme), int(np.argmax(np.abs(delta_lme))), np.abs(delta_lme).max()))
inter = st_lme & st_lc
P("交集: %d 维  (LoCoMo街区内LME命中=%.1f%%, LME街区内LoCoMo命中=%.1f%%)" % (
    len(inter), 100.0 * len(inter) / max(1, len(st_lc)), 100.0 * len(inter) / max(1, len(st_lme))))
r = np.corrcoef(delta_lme, delta_lc)[0, 1]
P("Δ向量全程相关系数: %.3f" % r)
top10_lc = np.argsort(-np.abs(delta_lc))[:10]
P("LoCoMo top10维: %s" % ",".join("d%d" % x for x in top10_lc))
top10_lme = np.argsort(-np.abs(delta_lme))[:10]
P("LME    top10维: %s" % ",".join("d%d" % x for x in top10_lme))

# ===== 验证#2: 说话人可分性 (LME 1024全库, user-vs-assistant留一最近邻) =====
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
P("\nLME 1024全库: %s %.0fs" % (str(D.shape), time.time() - t0))
labels = np.array([0 if r.startswith("user:") else (1 if r.startswith("assistant:") else -1) for r in RAWS])
rng = np.random.RandomState(0)
idx = rng.choice(np.where(labels >= 0)[0], size=10000, replace=False)
sub = D[idx]
lab = labels[idx]
sims = sub @ sub.T
np.fill_diagonal(sims, -9)
nn = np.argmax(sims, axis=1)
acc = (lab[nn] == lab).mean()
P("说话人留一最近邻(user vs assistant, n=10000): %.1f%%  (随机50%%)" % (100.0 * acc))
P("done %.0fs" % (time.time() - t0))
