# -*- coding: utf-8 -*-
"""foundry120.py — 问题向量形态挖掘(用户: 问题在向量的表现形式):
①几何特征(密度/距离/投影) ②聚类形态 ③形态×检索对错的关联(盲合规)"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry120_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

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
TIME_WORDS = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)

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
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
P("loaded %.0fs" % (time.time() - t0))

NQ = len(d)
is_dated = np.array([bool(TIME_WORDS.search(r)) for r in RAWS])
delta_t = D[is_dated].mean(0) - D[~is_dated].mean(0)
delta_t = delta_t / max(np.linalg.norm(delta_t), 1e-9)
# ===== ① 问题几何特征 =====
lib_centroid = l2n(D[NOHDR].mean(0, keepdims=True))[0]
FEAT = np.zeros((NQ, 5), dtype=np.float32)
for qi in range(NQ):
    qv = X[qi]
    hay_keys = set("lme-s" + sid2h[s][:12] for s in d[qi]["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    dd = D[domain] @ qv
    FEAT[qi, 0] = float(qv @ lib_centroid)          # 到库质心(全域相关度)
    FEAT[qi, 1] = float(np.sort(dd)[-100:].mean())   # 域内top100密度(话题热度)
    FEAT[qi, 2] = float(dd.max() - np.sort(dd)[-50:].mean())  # 头部锐度(top1-top50差)
    FEAT[qi, 3] = float(np.linalg.norm(np.load(OUT + "/q_bge1024.npy")[qi]))  # 范数(用原始未归一)
    FEAT[qi, 4] = float(qv @ delta_t)
P("几何特征 %.0fs" % (time.time() - t0))

# ===== ② 聚类形态 =====
from sklearn.cluster import KMeans
Z = (FEAT - FEAT.mean(0)) / (FEAT.std(0) + 1e-9)
best = None
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z)
    sil = 0
    # 简易轮廓: 组内vs组间
    labels = km.labels_
    inw = np.mean([np.linalg.norm(Z[i] - km.cluster_centers_[labels[i]]) for i in range(NQ)])
    if best is None or inw < best[1]:
        best = (k, inw, labels)
k, inw, labels = best
P("聚类: 最优k=%d (组内距%.2f)" % (k, inw))

# ===== ③ 形态×检索对错(总分驱动, 盲合规; numpy向量化) =====
SIDs_arr = np.array(SIDS)
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)
res_by_cluster = {}
ok_qi = set()
for qi in range(NQ):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (d[qi].get("answer_session_ids") or []) if s in sid2h)
    if gold_h:
        ok_qi.add(qi)
# 向量化: 每题域内一次算
S5 = np.zeros(NQ, dtype=bool)
T5 = np.zeros(NQ, dtype=bool)
T5_valid = np.zeros(NQ, dtype=bool)
for qi in sorted(ok_qi):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (d[qi].get("answer_session_ids") or []) if s in sid2h)
    hay_keys = set("lme-s" + sid2h[s][:12] for s in d[qi]["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    cq = D[domain] @ X[qi]
    dsids = SIDs_arr[domain]
    # 会话max: numpy分组
    umap = {s: j for j, s in enumerate(sorted(set(dsids.tolist())))}
    uidx = np.array([umap[s] for s in dsids.tolist()])
    smax = np.full(len(umap), -9.0)
    np.maximum.at(smax, uidx, cq)
    uarr = np.array(sorted(umap))
    order_s = uarr[np.argsort(-smax)]
    S5[qi] = gold_h <= set(order_s[:5].tolist())
    # turn top5
    ans_raw = d[qi].get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if ans_stems:
        T5_valid[qi] = True
        top5 = domain[np.argsort(-cq)[:5]]
        hit = False
        for i in top5:
            if len(ans_stems & stems_of(RAWS[i])) / len(ans_stems) >= 0.5:
                hit = True
                break
        T5[qi] = hit

P("===== 问题向量形态 × 检索对错 =====")
P("簇 | 题数 | 会话@5 | turn@5 | 特征画像(z质心)")
for c in range(k):
    mask = (labels == c) & np.array([qi in ok_qi for qi in range(NQ)])
    n_ = int(mask.sum())
    s_ = int(S5[mask].sum())
    tmask = mask & T5_valid
    t_ = int(T5[tmask].sum())
    cz = Z[mask].mean(0)
    P("%2d | %4d | %5.1f%% | %5.1f%% | 密度%+.2f 锐度%+.2f 质心%+.2f 时间%+.2f 范数%+.2f" % (
        c, n_, 100.0 * s_ / max(1, n_), 100.0 * t_ / max(1, int(tmask.sum())),
        cz[0], cz[2], cz[1], cz[4], cz[3]))
P("特征: [域内top100密度, 头部锐度, 到库质心, 时间性投影, 范数] (z)")
P("F120_DONE %.0fs" % (time.time() - t0))
