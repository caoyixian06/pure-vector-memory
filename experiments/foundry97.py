# -*- coding: utf-8 -*-
"""foundry97.py — 孤立性信号验证(用户洞察: 金噪0.077<噪噪0.128→单金题金=孤立点)
测: v11前15中, 凝聚度最低候选是金的命中率/金凝聚度排名分布/AUC"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v11.json", encoding="utf-8"))

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

def core_group(qa, grps):
    ans = " ".join(str(x) for x in (Q[qa].get("answer") or []))
    aw = set(stem(w) for w in re.findall(r"[a-z]+", ans.lower()) if len(w) > 2)
    if not aw:
        return grps[0]
    best, bv = grps[0], -1.0
    for grp in grps:
        tw = set()
        for g in grp:
            tw |= set(stem(w) for w in re.findall(r"[a-z']+", RAW[g].lower()))
        cov = len(aw & tw) / len(aw)
        if cov > bv:
            bv, best = cov, grp
    return best

TOKC = [toks(r) for r in RAW]
single_rank = []      # 单金题: 金的凝聚度在15条中的名次(1=最孤立)
multi_rank = []
single_argmin_hit = 0
single_n = 0
pos_c, neg_c = [], []   # 单金题: 金vs噪凝聚度(AUC)
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    head = seq[:15]
    tk = [TOKC[i] for i in head]
    n = len(head)
    coh = np.zeros(n)
    for a2 in range(n):
        js = [len(tk[a2] & tk[b2]) / max(1, len(tk[a2] | tk[b2])) for b2 in range(n) if b2 != a2]
        coh[a2] = np.mean(js) if js else 0.0
    goldset = set()
    for g in grps:
        goldset |= set(head) & g
    cg = core_group(qa, grps) & set(head)
    is_single = (len(grps) == 1)
    if is_single:
        single_n += 1
        golds = [r for r, i in enumerate(head) if i in goldset]
        if golds:
            order = np.argsort(coh)   # 孤立→稠密
            rank_of_gold = [list(order).index(g) + 1 for g in golds]
            single_rank.extend(rank_of_gold)
            if order[0] in golds:
                single_argmin_hit += 1
            for g in golds:
                pos_c.append(coh[g])
            for r2, i in enumerate(head):
                if i not in goldset:
                    neg_c.append(coh[r2])
    else:
        golds = [r for r, i in enumerate(head) if i in goldset]
        if golds:
            order = np.argsort(coh)
            multi_rank.extend([list(order).index(g) + 1 for g in golds])

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

sr = np.array(single_rank)
mr = np.array(multi_rank)
P("单金题 n=%d (金在头部的题)" % single_n)
P("  金的孤立名次: 中位=%.0f/15  均值=%.1f  名次≤3占比=%.1f%%" % (
    np.median(sr), sr.mean(), 100.0 * (sr <= 3).mean()))
P("  最孤立(argmin)=金命中率: %.1f%%  (随机%.1f%%)" % (
    100.0 * single_argmin_hit / max(1, single_n), 100.0 / 15))
P("  凝聚度AUC(反向, 金vs噪): %.3f  金均值=%.3f 噪均值=%.3f" % (
    1 - auc(pos_c, neg_c), np.mean(pos_c), np.mean(neg_c)))
P("多金题对照: 金凝聚名次中位=%.0f/15 (多金应偏稠密端)" % (np.median(mr) if len(mr) else -1))
P("done %.0fs" % (time.time() - t0))
