# -*- coding: utf-8 -*-
"""foundry98.py — 全池穷举配对反查金(用户方案): 多金题高边端点/单金题最低平均边权"""
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

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
MID2I = {m: i for i, m in enumerate(MID)}
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
X = np.load(HERE + "/xz_cache.npz")["X"]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
C0 = l2n(X @ D.T)
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
FINAL = z_ck["FINAL"]

def base_key(m):
    mm = re.match(r"^(loco-conv-\d+)_(rbak(\d+)k|m(\d+))$", m)
    if mm:
        return mm.group(1) + "_" + (mm.group(3) or mm.group(4)).lstrip("0").zfill(2)
    return m
PAIR = {}
seen = {}
for i, m in enumerate(MID):
    k = base_key(m)
    if k in seen:
        PAIR[i] = seen[k]; PAIR[seen[k]] = i
    else:
        seen[k] = i

def gold_groups(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    groups = []
    for k in keys:
        hits = set(i for i in range(len(MID)) if k in RAWN[i])
        if not hits:
            continue
        for i in list(hits):
            tt = PAIR.get(i)
            if tt is not None:
                hits.add(tt)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

POOLSZ = 700
multi_hit = 0
multi_n = 0
multi_endp_hit = 0
single_hit = 0
single_n = 0
pos_edge, neg_edge = [], []
for k_i, qa in enumerate(IDS):
    grps = gold_groups(qa)
    if not grps:
        continue
    pool = sorted(set(np.argsort(-FINAL[k_i])[:250]) | set(np.argsort(-C0[k_i])[:250]))[:POOLSZ]
    V = QW[pool]
    S = V @ V.T
    np.fill_diagonal(S, -9)
    avg_edge = S.mean(axis=1)
    goldset = set()
    for g in grps:
        goldset |= set(pool) & g
    is_gold_arr = np.array([c in goldset for c in pool])
    if len(grps) >= 2:
        multi_n += 1
        # 高边端点: 边>0.85的对的端点集合
        hi = np.argwhere(S > 0.85)
        endp = set()
        for a2, b2 in hi:
            endp.add(a2)
            endp.add(b2)
        if endp:
            hit = sum(is_gold_arr[list(endp)]) / len(endp)
            multi_endp_hit += hit
            # 端点集合作为金预测的查准(端点里金占比)已计; 记成功题: 端点全部是金
            if all(is_gold_arr[list(endp)]):
                multi_hit += 1
    else:
        single_n += 1
        am = int(np.argmin(avg_edge))
        if is_gold_arr[am]:
            single_hit += 1
        pos_edge.append(avg_edge[is_gold_arr].mean())
        neg_edge.append(avg_edge[~is_gold_arr].mean())

P("多金题 n=%d: 高边(>0.85)端点全金率=%.1f%%  端点平均金纯度=%.1f%%" % (
    multi_n, 100.0 * multi_hit / max(1, multi_n), 100.0 * multi_endp_hit / max(1, multi_n)))
P("单金题 n=%d: 全池最低平均边权=金命中率=%.1f%% (随机%.1f%%)" % (
    single_n, 100.0 * single_hit / max(1, single_n), 100.0 / POOLSZ))
P("单金题 边权AUC(反向, 金avg边 vs 噪avg边): %.3f  金=%.3f 噪=%.3f" % (
    1 - auc(pos_edge, neg_edge), np.mean(pos_edge), np.mean(neg_edge)))
P("done %.0fs" % (time.time() - t0))
