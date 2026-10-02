# -*- coding: utf-8 -*-
"""foundry71.py — @5错题解剖: 掉出金片的排名分布/题型分层/信号差/说话人匹配率"""
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
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

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
X = np.load(HERE + "/xz_cache.npz")["X"]
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
IDX = {q: i for i, q in enumerate(IDS)}
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

# 说话人表: 每会话从raw前缀统计
SPK = {}
for i in range(len(MID)):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    if mm:
        SPK.setdefault(CONVKEY[i], set()).add(mm.group(1))
def speaker_of(i):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    return mm.group(1) if mm else None

# 题型字段探测
k0 = IDS[0]
P("question字段: %s" % sorted(Q[k0].keys()))
CAT = {}
for qa in IDS:
    c = Q[qa].get("category") or Q[qa].get("type") or "?"
    CAT[qa] = str(c)[:12]

# ===== 解剖 =====
nG = 0
fail = 0
rank_hist = {"6-10": 0, "11-15": 0, "16-35": 0, ">35": 0}
cat_tot = {}
cat_fail = {}
spk_hit_gold = 0
spk_tot_gold = 0
spk_hit_noise = 0
spk_tot_noise = 0
fail_sig = []   # (g1, dqc, vqc, jac, lr) 失败掉出金
ok_sig = []     # 进窗金
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    nG += 1
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    pos = {i: r for r, i in enumerate(seq)}
    qi = IDX[qa]
    q = Q[qa]
    qtok = toks(q["question"])
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    qtext = q["question"].lower()
    conv_speakers = SPK.get(CONVKEY[seq[0]] if seq else "x", set())
    q_spk = set(s for s in conv_speakers if s and s.lower() in qtext)
    passed = all(any(pos.get(g) is not None and pos[g] < 5 for g in grp) for grp in grps)
    # 信号计算: 每组的最优代表
    all_best = []
    for grp in grps:
        cand = [g for g in grp if g in pos]
        if cand:
            all_best.append(min(cand, key=lambda g: pos[g]))
    for g in all_best:
        s = speaker_of(g)
        if s is not None:
            spk_tot_gold += 1
            if (not q_spk) or s in q_spk:
                spk_hit_gold += 1
        if pos.get(g, 999) < 5:
            rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[g].lower()))
            inter = qtok & toks(RAW[g])
            ok_sig.append((len(qstems & rstems) / max(1, len(qstems)),
                           float(D[g] @ X[qi]), float(Q256[qi] @ QW[g]),
                           len(inter) / max(1, len(qtok | toks(RAW[g])))))
        else:
            r = pos.get(g, 999)
            if r <= 10:
                rank_hist["6-10"] += 1
            elif r <= 15:
                rank_hist["11-15"] += 1
            elif r <= 35:
                rank_hist["16-35"] += 1
            else:
                rank_hist[">35"] += 1
            rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[g].lower()))
            inter = qtok & toks(RAW[g])
            fail_sig.append((len(qstems & rstems) / max(1, len(qstems)),
                             float(D[g] @ X[qi]), float(Q256[qi] @ QW[g]),
                             len(inter) / max(1, len(qtok | toks(RAW[g])))))
    # 头部噪声说话人
    for i in seq[:5]:
        if not any(i in grp for grp in grps):
            s = speaker_of(i)
            if s is not None:
                spk_tot_noise += 1
                if (not q_spk) or s in q_spk:
                    spk_hit_noise += 1
    cat_tot[CAT[qa]] = cat_tot.get(CAT[qa], 0) + 1
    if not passed:
        fail += 1
        cat_fail[CAT[qa]] = cat_fail.get(CAT[qa], 0) + 1

P("\nnG=%d  @5失败=%d (%.1f%%)" % (nG, fail, 100.0 * fail / nG))
P("掉出金片排名分布: " + str(rank_hist))
P("\n题型失败率:")
for c in sorted(cat_tot, key=lambda x: -cat_fail.get(x, 0)):
    P("  %-14s %5.1f%%  (%d/%d)" % (c, 100.0 * cat_fail.get(c, 0) / cat_tot[c], cat_fail.get(c, 0), cat_tot[c]))
P("\n信号对比 (进窗金 n=%d vs 掉出金 n=%d):" % (len(ok_sig), len(fail_sig)))
for ni, nm in enumerate(("G1词干", "1024cos", "256cos", "Jaccard")):
    a = np.array([x[ni] for x in ok_sig])
    b = np.array([x[ni] for x in fail_sig])
    P("  %-9s 金@5=%.3f  掉出=%.3f  Δ=%.3f" % (nm, a.mean(), b.mean(), b.mean() - a.mean()))
P("\n说话人匹配: 金片 %d/%d=%.1f%%   头部噪声 %d/%d=%.1f%%" % (
    spk_hit_gold, spk_tot_gold, 100.0 * spk_hit_gold / max(1, spk_tot_gold),
    spk_hit_noise, spk_tot_noise, 100.0 * spk_hit_noise / max(1, spk_tot_noise)))
