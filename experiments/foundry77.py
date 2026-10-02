# -*- coding: utf-8 -*-
"""foundry77.py — 影子结构验证: 问题↔对话内追问句(影子), 金=影子的下一条回答?
测: 池内非金top1(影子)的问句率/下一条金命中率/±1金率; prev-cos特征区分度"""
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
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = np.load(HERE + "/xz_cache.npz")["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
C0 = l2n(X @ D.T)
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
P("loaded nG=%d %ds" % (nG, int(time.time() - t0)))

def speaker_of(i):
    mm = re.match(r"^([A-Z][a-z]+)\s*:", RAW[i])
    return mm.group(1) if mm else None

# ===== 影子结构验证 =====
# 影子 = 全会话内(排除金及其孪生)与问题1024cos最高的记录
stat = {"n": 0, "shadow_is_q": 0, "next_is_gold": 0, "prev_is_gold": 0, "pm1_gold": 0,
        "shadow_diff_spk": 0, "next_of_failq_gold": 0, "n_fail": 0}
# prev-cos 特征区分度: 金 vs v10前5噪声
pos_pc, neg_pc = [], []
for qa, grps in GS.items():
    qi = IDS.index(qa)
    qv = X[qi]
    goldset = set()
    for grp in grps:
        goldset |= grp
    myconv = "loco-" + qa.split("#")[0]
    conv_idx = [i for i in range(len(MID)) if CONVKEY[i] == myconv and "_rbak" not in MID[i]]
    if not conv_idx:
        continue
    sims = D[conv_idx] @ qv
    order = sorted(range(len(conv_idx)), key=lambda r: -sims[r])
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    pos5 = {i: r for r, i in enumerate(seq)}
    passed = all(any(pos5.get(g, 999) < 5 for g in grp) for grp in grps)
    # 影子 = 非金 top1
    shadow = None
    for r in order:
        i = conv_idx[r]
        if i not in goldset:
            shadow = i
            break
    if shadow is None:
        continue
    stat["n"] += 1
    if RAW[shadow].rstrip().endswith("?"):
        stat["shadow_is_q"] += 1
    ss = speaker_of(shadow)
    # 下一跳: m序号+1 同会话
    nxt = shadow + 1 if shadow + 1 < len(MID) and CONVKEY[shadow + 1] == myconv else None
    prv = shadow - 1 if shadow - 1 >= 0 and CONVKEY[shadow - 1] == myconv else None
    if nxt is not None:
        if nxt in goldset or (PAIR.get(nxt) is not None and PAIR[nxt] in goldset):
            stat["next_is_gold"] += 1
            if not passed:
                stat["next_of_failq_gold"] += 1
    if prv is not None and (prv in goldset or (PAIR.get(prv) is not None and PAIR[prv] in goldset)):
        stat["prev_is_gold"] += 1
    if (nxt is not None and (nxt in goldset)) or (prv is not None and (prv in goldset)):
        stat["pm1_gold"] += 1
    if not passed:
        stat["n_fail"] += 1
    if ss is not None and speaker_of(nxt if nxt is not None else shadow) != ss:
        stat["shadow_diff_spk"] += 1
    # prev-cos: 金 vs 前5噪声 (各自前一条同会话记录的1024cos)
    for g in goldset:
        pg = g - 1 if g - 1 >= 0 and CONVKEY[g - 1] == CONVKEY[g] else None
        pos_pc.append(float(D[pg] @ qv) if pg is not None else 0.0)
    for r, i in enumerate(seq[:5]):
        if i in goldset:
            continue
        pi = i - 1 if i - 1 >= 0 and CONVKEY[i - 1] == CONVKEY[i] else None
        neg_pc.append(float(D[pi] @ qv) if pi is not None else 0.0)

n = stat["n"]
P("\n影子结构 (影子=会话内非金cos top1, n=%d):" % n)
P("  影子是问句:        %5.1f%%" % (100.0 * stat["shadow_is_q"] / n))
P("  影子的下一条=金:   %5.1f%%" % (100.0 * stat["next_is_gold"] / n))
P("  影子的前一条=金:   %5.1f%%" % (100.0 * stat["prev_is_gold"] / n))
P("  影子±1含金:        %5.1f%%" % (100.0 * stat["pm1_gold"] / n))
P("  影子下一跳异说话人: %5.1f%%" % (100.0 * stat["shadow_diff_spk"] / n))
P("  (@5失败题中影子下一条=金: %d/%d)" % (stat["next_of_failq_gold"], stat["n_fail"]))

def auc(pos, neg):
    allv = list(pos) + list(neg)
    ranks = {}
    for r, v in enumerate(sorted(allv), 1):
        ranks[v] = ranks.get(v, 0)
    # 处理并列: 简化版
    import statistics
    sv = sorted(allv)
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(sv, v) + 1 + 0.5 * (bs.bisect_right(sv, v) - bs.bisect_left(sv, v))
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

P("\nprev-cos特征 (本条金/噪 vs 其前一条与问题的1024cos):")
P("  金 prev-cos=%.3f (n=%d)  前五噪 prev-cos=%.3f (n=%d)  AUC=%.3f" % (
    np.mean(pos_pc), len(pos_pc), np.mean(neg_pc), len(neg_pc), auc(pos_pc, neg_pc)))
P("done %.0fs" % (time.time() - t0))
