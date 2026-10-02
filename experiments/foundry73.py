# -*- coding: utf-8 -*-
"""foundry73.py — 反回声重排: new_ratio(内容增量)/echo(问题词覆盖) 区分度+头部重排"""
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
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your really cool wow thanks hey yeah okay ok just been being get got getting lot bit kind sort pretty much many some all also too very there their here have has had having not no yes but so as by be am were will would can could should may might do does dont didn".split())

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

GS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
nG = len(GS)

def feats(i, qt, qts):
    rt = toks(RAW[i])
    rts = set(stem(w) for w in rt)
    new_ratio = len((rts - qts) - QSTOP) / max(1, len(rts))
    echo = len(rt & qt) / max(1, len(rt))
    long_ratio = sum(1 for w in (rts - qts) - QSTOP if len(w) >= 6) / max(1, len(rts))
    return new_ratio, echo, long_ratio

# ===== 1) 区分度: 前5噪声(假头) vs 6-15名金(要提的) =====
pos_new, neg_new, pos_echo, neg_echo, pos_anti, neg_anti = [], [], [], [], [], []
for qa, grps in GS.items():
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    qt = toks(Q[qa]["question"])
    qts = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()))
    goldset = set()
    for grp in grps:
        goldset |= grp
    for r, i in enumerate(seq[:15]):
        nr, ec, lr = feats(i, qt, qts)
        anti = nr - ec
        if i in goldset:
            if r >= 5:
                pos_new.append(nr); pos_echo.append(ec); pos_anti.append(anti)
        else:
            if r < 5:
                neg_new.append(nr); neg_echo.append(ec); neg_anti.append(anti)

def auc(pos, neg):
    allv = pos + neg
    ranks = {v: r for r, v in enumerate(sorted(allv), 1)}
    rp = sum(ranks[v] for v in pos)
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

P("区分度 (正=6-15名要提的金 n=%d, 负=前5噪声 n=%d):" % (len(pos_new), len(neg_new)))
P("  new_ratio: 金=%.3f 噪=%.3f AUC=%.3f" % (np.mean(pos_new), np.mean(neg_new), auc(pos_new, neg_new)))
P("  echo:      金=%.3f 噪=%.3f AUC=%.3f" % (np.mean(pos_echo), np.mean(neg_echo), auc(pos_echo, neg_echo)))
P("  new-echo:  金=%.3f 噪=%.3f AUC=%.3f" % (np.mean(pos_anti), np.mean(neg_anti), auc(pos_anti, neg_anti)))

# ===== 2) 重排臂: head=前15, score = rank分 + λ·z(anti) =====
def rerank(seq, qt, qts, lam, headsz=15):
    head = seq[:headsz]
    n = len(head)
    fv = np.array([feats(i, qt, qts) for i in head])
    anti = fv[:, 0] - fv[:, 1]
    z = (anti - anti.mean()) / (anti.std() + 1e-9)
    rk = np.array([(headsz - r) / headsz for r in range(n)])
    sc = rk + lam * z
    order = list(np.argsort(-sc))
    new_head = [head[c] for c in order]
    return new_head + seq[headsz:]

for lam in (0.0, 0.3, 0.5, 1.0, 2.0):
    a5 = 0
    for qa, grps in GS.items():
        seq = [MID2I[m] for m in FORD.get(qa, [])]
        qt = toks(Q[qa]["question"])
        qts = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()))
        new = seq if lam == 0.0 else rerank(seq, qt, qts, lam)
        if all(any(g in new[:5] for g in grp) for grp in grps):
            a5 += 1
    P("λ=%.1f: @5=%.1f%%" % (lam, 100.0 * a5 / nG))
P("done %.0fs" % (time.time() - t0))
