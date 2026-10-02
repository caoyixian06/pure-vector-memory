# -*- coding: utf-8 -*-
"""foundry81.py — 检索层反哺: 词面计票第三通道并入池, 测池全金率(倒排表版)"""
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
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
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
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
QWQ256 = l2n(Q256 @ QW.T)

# 词干倒排表(过滤停用词)
W2R = {}
for i in range(NR):
    for w in set(stem(x) for x in re.findall(r"[a-z']+", RAW[i].lower())):
        if len(w) > 2 and w not in QSTOP:
            W2R.setdefault(w, []).append(i)
P("倒排表: %d词 %.0fs" % (len(W2R), time.time() - t0))

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
        hits = set(i for i in range(NR) if k in RAWN[i])
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
QSTEMS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
        QSTEMS[qa] = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
nG = len(GS)
P("nG=%d" % nG)

def lex_votes(qa):
    votes = np.zeros(NR, dtype=np.int32)
    for w in QSTEMS[qa]:
        for i in W2R.get(w, ()):
            votes[i] += 1
    return votes

LEX_TOP = {qa: set(np.argsort(-lex_votes(qa))[:350]) for qa in GS}
P("词面通道 %.0fs" % (time.time() - t0))

def pool_recall(use_lex, cap):
    ok = 0
    for k_i, qa in enumerate(IDS):
        grps = GS.get(qa)
        if not grps:
            continue
        pool = set(np.argsort(-C0[k_i])[:350]) | set(np.argsort(-QWQ256[k_i])[:350])
        if use_lex:
            pool |= LEX_TOP[qa]
        top30 = list(np.argsort(-C0[k_i])[:30])
        for i in top30:
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                    pool.add(j)
        pool = set(sorted(pool)[:cap])
        if all(any(g in pool for g in grp) for grp in grps):
            ok += 1
    return ok

for use_lex, cap, tag in ((False, 700, "双cos+邻域(cap700, 现行)"),
                          (True, 700, "三通道+邻域(cap700)"),
                          (True, 900, "三通道+邻域(cap900)")):
    ok = pool_recall(use_lex, cap)
    P("%s: 池全金 = %.1f%% (%d/%d)" % (tag, 100.0 * ok / nG, ok, nG))
P("done %.0fs" % (time.time() - t0))
