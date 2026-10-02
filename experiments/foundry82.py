# -*- coding: utf-8 -*-
"""foundry82.py — 裸cos全库排序的学长口径数字: 量化检索层功能性贡献"""
import io, json, re, sys
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
X = np.load(HERE + "/xz_cache.npz")["X"]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
C0 = l2n(X @ D.T)
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
QW = np.zeros((len(MID), 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QWQ256 = l2n(Q256 @ QW.T)

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

GS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
nG = len(GS)

def post(seq, qa):
    myconv = "loco-" + qa.split("#")[0]
    out, used = [], set()
    for i in seq:
        if CONVKEY[i] != myconv:
            continue
        p = PAIR.get(i)
        if p is not None and p in used:
            continue
        out.append(i)
        used.add(i)
    return out

for tag, scorer in (("1024cos全库裸排", lambda k: C0[k]),
                    ("256通道全库裸排", lambda k: QWQ256[k]),
                    ("双通道均值裸排", lambda k: C0[k] + QWQ256[k])):
    all15 = core5 = 0
    for k, qa in enumerate(IDS):
        grps = GS.get(qa)
        if not grps:
            continue
        seq = post(list(np.argsort(-scorer(k))), qa)
        if all(any(g in seq[:15] for g in grp) for grp in grps):
            all15 += 1
        cg = core_group(qa, grps)
        if any(g in seq[:5] for g in cg):
            core5 += 1
    P("%s: 全部@15=%5.1f%%  核心@5=%5.1f%%" % (tag, 100.0 * all15 / nG, 100.0 * core5 / nG))
