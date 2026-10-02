# -*- coding: utf-8 -*-
"""foundry76.py — 结构层三快测: ①掉出金与头部锚索引距离 ②团预聚类排序 ③E指纹分层@5"""
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
RTOK = [toks(r) for r in RAW]
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v10.json", encoding="utf-8"))

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

GS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
nG = len(GS)
P("loaded nG=%d %.0fs" % (nG, time.time() - t0))

# ===== ① 掉出金与头部锚的索引距离 =====
dist_hist = {"0(同条)": 0, "±1": 0, "±2": 0, "±3-5": 0, "±6-10": 0, ">10/异会话": 0}
n_fail = 0
for qa, grps in GS.items():
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    pos = {i: r for r, i in enumerate(seq)}
    goldset = set()
    for grp in grps:
        goldset |= grp
    head_anchor = seq[:2]
    if all(any(pos.get(g, 999) < 5 for g in grp) for grp in grps):
        continue
    n_fail += 1
    for grp in grps:
        out_gold = [g for g in grp if pos.get(g, 999) >= 5]
        in_gold = [g for g in grp if pos.get(g, 999) < 5]
        if not out_gold or not in_gold:
            continue
        for og in out_gold:
            best = min((abs(og - ig) if CONVKEY[og] == CONVKEY[ig] else 999, ig) for ig in in_gold)[0]
            if best == 999:
                dist_hist[">10/异会话"] += 1
            elif best == 0:
                dist_hist["0(同条)"] += 1
            elif best <= 1:
                dist_hist["±1"] += 1
            elif best <= 2:
                dist_hist["±2"] += 1
            elif best <= 5:
                dist_hist["±3-5"] += 1
            elif best <= 10:
                dist_hist["±6-10"] += 1
            else:
                dist_hist[">10/异会话"] += 1
P("\n① 掉出金 vs 已进金的索引距离 (失败题=%d):" % n_fail)
P("  " + str(dist_hist))

# ===== ② 团预聚类排序: 池内Jaccard连通团, 团分=团内最高GBDT名次, top5=头部团展开 =====
# 用v10前35作池(已排), 团=前35内Jaccard>0.35连通, 重排: 团按最高名次排序, 窗口=团展开
def clan_rerank(seq, headsz=35, thr=0.35):
    head = seq[:headsz]
    n = len(head)
    tk = [RTOK[i] for i in head]
    adj = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            j = len(tk[a] & tk[b]) / max(1, len(tk[a] | tk[b]))
            if j > thr:
                adj[a].append(b); adj[b].append(a)
    seen_c = [False] * n
    clans = []
    for a in range(n):
        if seen_c[a]:
            continue
        stack = [a]; comp = []
        seen_c[a] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v2 in adj[u]:
                if not seen_c[v2]:
                    seen_c[v2] = True
                    stack.append(v2)
        clans.append(sorted(comp))
    clans.sort(key=lambda c: min(c))
    out = []
    for c in clans:
        out.extend(c)
    return out + seq[headsz:]

a5_base = 0
a5_clan = 0
for qa, grps in GS.items():
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    if all(any(g in seq[:5] for g in grp) for grp in grps):
        a5_base += 1
    seq2 = clan_rerank(seq)
    if all(any(g in seq2[:5] for g in grp) for grp in grps):
        a5_clan += 1
P("\n② 团预聚类重排: base@5=%.1f%%  clan@5=%.1f%%" % (100.0 * a5_base / nG, 100.0 * a5_clan / nG))

# ===== ③ E指纹分层: 问题首词的@5分布 =====
first_tot = {}
first_ok = {}
for qa, grps in GS.items():
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    f = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    first_tot[f] = first_tot.get(f, 0) + 1
    if all(any(g in seq[:5] for g in grp) for grp in grps):
        first_ok[f] = first_ok.get(f, 0) + 1
P("\n③ E指纹分层@5 (按题量排序):")
for f in sorted(first_tot, key=lambda x: -first_tot[x])[:10]:
    P("  %-8s %5.1f%%  (%d/%d)" % (f, 100.0 * first_ok.get(f, 0) / first_tot[f], first_ok.get(f, 0), first_tot[f]))
P("done %.0fs" % (time.time() - t0))
