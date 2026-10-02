# -*- coding: utf-8 -*-
"""foundry80.py — 金内分层: 多组题核心组vs上下文组的特征区分度"""
import io, json, re, sys
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
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
XSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your really cool wow thanks hey yeah okay ok just been being get got getting lot bit kind sort pretty much many some all also too very there their here have has had having not no yes but so as by be am will would can could should may might dont didnt".split())

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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v10.json", encoding="utf-8"))
MID2I = {m: i for i, m in enumerate(MID)}

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

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

# 核心组 vs 上下文组 特征对比 (多组题, 组代表=v10序内最优成员)
feat_names = ("new_ratio", "echo", "len", "num_ratio", "cap_ratio")
core_rows = []
ctx_rows = []
n_multi = 0
core_hit = 0
for qa in IDS:
    grps = gold_groups(qa)
    if len(grps) < 2:
        continue
    n_multi += 1
    cg = core_group(qa, grps)
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    pos = {i: r for r, i in enumerate(seq)}
    qt = toks(Q[qa]["question"])
    qts = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()))
    if any(pos.get(g, 999) < 5 for g in cg):
        core_hit += 1
    for grp in grps:
        cand = [g for g in grp if g in pos]
        if not cand:
            continue
        rep = min(cand, key=lambda g: pos[g])
        rt = toks(RAW[rep])
        rts = set(stem(w) for w in rt)
        new_ratio = len((rts - set(stem(w) for w in qt)) - XSTOP) / max(1, len(rts))
        echo = len(rt & qt) / max(1, len(rt))
        ln = len(RAW[rep].split()) / 40.0
        num_ratio = sum(1 for w in RAW[rep] if w.isdigit()) / max(1, len(RAW[rep].split()))
        cap_ratio = sum(1 for w in re.findall(r"[A-Z][a-z]+", RAW[rep][10:])) / max(1, len(RAW[rep].split()))
        row = (new_ratio, echo, ln, num_ratio, cap_ratio)
        (core_rows if grp is cg else ctx_rows).append(row)

P("多组题=%d  核心组@5命中率=%.1f%%" % (n_multi, 100.0 * core_hit / n_multi))
P("\n核心组(n=%d) vs 上下文组(n=%d) 特征区分度:" % (len(core_rows), len(ctx_rows)))
for ni, nm in enumerate(feat_names):
    a = [r[ni] for r in core_rows]
    b = [r[ni] for r in ctx_rows]
    P("  %-10s 核心=%.3f  上下文=%.3f  AUC=%.3f" % (nm, np.mean(a), np.mean(b), auc(a, b)))
