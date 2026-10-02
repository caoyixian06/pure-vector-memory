# -*- coding: utf-8 -*-
"""foundry100.py — 规律穷举扫描器: 16候选特征, 头部条件(6-15金vs前5噪), LoCoMo筛选"""
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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
FORD = json.load(open(HERE + "/r41_final_order_v11.json", encoding="utf-8"))
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))

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

def speaker_of(i):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    return mm.group(1) if mm else None

CONV_ROWS = {}
for i in range(len(MID)):
    CONV_ROWS.setdefault(CONVKEY[i], []).append(i)
RELPOS = np.zeros(len(MID), dtype=np.float32)
for ck, rows in CONV_ROWS.items():
    n = len(rows)
    for p, i in enumerate(rows):
        RELPOS[i] = p / max(1, n - 1)
# kNN密度(粗: 采样1000锚点)
rng0 = np.random.RandomState(0)
anchors = rng0.choice(len(MID), 1000, replace=False)
simA = D[anchors] @ D.T
KNN_DENS = np.zeros(len(MID), dtype=np.float32)
for i in range(0, len(MID), 2000):
    chunk = np.arange(i, min(i + 2000, len(MID)))
    sims = D[chunk] @ D[anchors].T
    KNN_DENS[chunk] = np.sort(sims, axis=1)[:, -20:].mean(1)
VNORM = np.linalg.norm(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32), axis=1)
P("precalc %.0fs" % (time.time() - t0))

def feats(i, qtext, qcaps, qhas_num, q_past):
    r = RAW[i]
    words = r.split()
    nw = max(1, len(words))
    toks_ = re.findall(r"[a-z']+", r.lower())
    f = {}
    f["role_firstp"] = 1.0 if r.startswith("user:") or bool(speaker_of(i)) else 0.0
    f["num_dens"] = sum(1 for w in words if any(c.isdigit() for c in w)) / nw
    f["cap_dens"] = len(re.findall(r"[A-Z][a-z]+", r[10:])) / nw
    f["verb_dens"] = sum(1 for w in toks_ if w.endswith("ing") or w.endswith("ed")) / max(1, len(toks_))
    f["ttr"] = len(set(toks_)) / max(1, len(toks_))
    f["relpos"] = float(RELPOS[i])
    f["wlen"] = np.mean([len(w) for w in toks_]) / 10.0 if toks_ else 0.0
    f["excl"] = (r.count("!") + r.count("?")) / nw
    f["num_match"] = 1.0 if (qhas_num and any(c.isdigit() for c in r)) else 0.0
    rcaps = set(re.findall(r"[A-Z][a-z]+", r))
    f["cap_match"] = len(qcaps & rcaps) / max(1, len(qcaps)) if qcaps else 0.0
    f["past_match"] = 1.0 if (q_past and sum(1 for w in toks_ if w.endswith("ed")) > 0) else 0.0
    f["knn_dens"] = float(KNN_DENS[i])
    f["vnorm"] = float(VNORM[i])
    f["paren"] = 1.0 if ("(" in r or ")" in r) else 0.0
    f["quote"] = 1.0 if '"' in r else 0.0
    return f

FN = ["role_firstp", "num_dens", "cap_dens", "verb_dens", "ttr", "relpos", "wlen", "excl",
      "num_match", "cap_match", "past_match", "knn_dens", "vnorm", "paren", "quote"]
POS = {k: [] for k in FN}
NEG = {k: [] for k in FN}
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    goldset = set()
    for g in grps:
        goldset |= g
    qtext = Q[qa].get("question") or ""
    qcaps = set(re.findall(r"[A-Z][a-z]+", qtext))
    qhas_num = any(c.isdigit() for c in qtext)
    q_past = bool(re.search(r"\bdid\b|\bwas\b|\bwere\b|\bhave\b", qtext, re.I))
    for r_, i in enumerate(seq[:15]):
        fv = feats(i, qtext, qcaps, qhas_num, q_past)
        if i in goldset:
            if r_ >= 5:
                for k in FN:
                    POS[k].append(fv[k])
        else:
            if r_ < 5:
                for k in FN:
                    NEG[k].append(fv[k])

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

P("\n===== 穷举扫描(头部: 6-15金 n=%d vs 前5噪 n=%d) =====" % (len(POS[FN[0]]), len(NEG[FN[0]])))
rows = []
for k in FN:
    a = auc(POS[k], NEG[k])
    rows.append((max(a, 1 - a), a, k))
    P("%-12s AUC=%.3f  金=%.3f 噪=%.3f" % (k, a, np.mean(POS[k]), np.mean(NEG[k])))
rows.sort(reverse=True)
P("\n存活(AUC>0.6): %s" % ", ".join("%s(%.3f)" % (k, a) for m, a, k in rows if m > 0.6))
P("done %.0fs" % (time.time() - t0))
