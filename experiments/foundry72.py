# -*- coding: utf-8 -*-
"""foundry72.py — 头部吸引重排: 三团规律正确用法(金要聚), 贪心选5 + cat1失败样例"""
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
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
MID2I = {m: i for i, m in enumerate(MID)}
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
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
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v9.json", encoding="utf-8"))

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
            t = PAIR.get(i)
            if t is not None:
                hits.add(t)
        for g in groups:
            if g & hits:
                g |= hits
                break
        else:
            groups.append(hits)
    return groups

GS = {qa: gold_groups(qa) for qa in IDS}
GS = {qa: g for qa, g in GS.items() if g}
nG = len(GS)
P("loaded nG=%d %.0fs" % (nG, time.time() - t0))

def greedy_attract(seq, K=5, lam=1.0, headsz=12):
    """贪心吸引: top1锚定, 逐个加入 rank分+λ×与已选平均相似 最大者"""
    head = seq[:headsz]
    if len(head) <= K:
        return seq
    tk = {i: toks(RAW[i]) for i in head}
    # 相似度矩阵: z(jaccard)+z(1024cos)
    n = len(head)
    S = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = head[a], head[b]
            jac = len(tk[ia] & tk[ib]) / max(1, len(tk[ia] | tk[ib]))
            c1024 = float(D[ia] @ D[ib])
            S[a, b] = S[b, a] = jac + c1024
    z = (S - S.mean()) / (S.std() + 1e-9)
    rank_score = np.array([(headsz - r) / headsz for r in range(n)])
    chosen = [0]
    rest = set(range(1, n))
    while len(chosen) < K and rest:
        best, bv = None, -1e9
        for r in rest:
            coh = np.mean([z[r, c] for c in chosen])
            v = rank_score[r] + lam * coh
            if v > bv:
                bv, best = v, r
        chosen.append(best)
        rest.discard(best)
    order5 = [head[c] for c in chosen]
    tail = [i for i in head if i not in set(order5)]
    return order5 + tail + seq[headsz:]

# 基线 + 各λ
for lam in (0.0, 0.3, 1.0, 3.0):
    a5 = a3 = 0
    for qa, grps in GS.items():
        seq = [MID2I[m] for m in FORD.get(qa, [])]
        if lam == 0.0:
            new = seq
        else:
            new = greedy_attract(seq, K=5, lam=lam)
        if all(any(g in new[:5] for g in grp) for grp in grps):
            a5 += 1
        if all(any(g in new[:3] for g in grp) for grp in grps):
            a3 += 1
    P("λ=%.1f: @3=%.1f%%  @5=%.1f%%" % (lam, 100.0 * a3 / nG, 100.0 * a5 / nG))

# cat1失败样例
P("\n===== cat1失败样例 =====")
cnt = 0
for qa, grps in GS.items():
    if str(Q[qa].get("category")) != "1" or cnt >= 3:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    if all(any(g in seq[:5] for g in grp) for grp in grps):
        continue
    P("\n[%s] Q: %s" % (qa, Q[qa]["question"][:130]))
    P("  A: %s" % str(Q[qa].get("answer"))[:80])
    for r, i in enumerate(seq[:12]):
        tag = "GOLD" if any(i in grp for grp in grps) else "    "
        P("  %2d%s %s" % (r, tag, RAW[i][:100].replace("\n", " ")))
    cnt += 1
