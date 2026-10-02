# -*- coding: utf-8 -*-
"""foundry96.py — 用户向量路由架构: 问题轴向定向→条件启用通道(凝聚度只在多金, 时间投影只在时间题)
后处理版: v11终序 + 路由提权, 对照v11=80.5/78.2"""
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
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

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
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
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

GS = {}
for qa in IDS:
    g = gold_groups(qa)
    if g:
        GS[qa] = g
nG = len(GS)
P("loaded %.0fs" % (time.time() - t0))

# ===== 轴构造(零标签) =====
DATE_WORDS = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
is_dated = np.array([bool(DATE_WORDS.search(r)) for r in RAW])
DELTA_T = QW[is_dated].mean(0) - QW[~is_dated].mean(0)
DELTA_T = DELTA_T / max(np.linalg.norm(DELTA_T), 1e-9)
REC_TPROJ = QW @ DELTA_T
# 人名轴: 库内说话人名(专名大写开头)的256向量平均
SPK = set()
for r in RAW:
    mm = re.match(r"^([A-Z][a-z]+)\s*:", r)
    if mm:
        SPK.add(mm.group(1))
WHd = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WHd.keys())
WV = l2n(np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
def axis_of(words):
    idxs = [W2I[w] for w in words if w in W2I]
    if len(idxs) < 2:
        return None
    v = WV[idxs].mean(0)
    return v / max(np.linalg.norm(v), 1e-9)
NAME_AX = axis_of([w.lower() for w in SPK])
P("说话人=%d 人名轴=%s 时间街区dated=%d %.0fs" % (len(SPK), NAME_AX is not None, is_dated.sum(), time.time() - t0))

def q_axis_proj(qa, ax):
    if ax is None:
        return 0.0
    qv = Q256[IDS.index(qa)]
    return float(qv @ ax)

# ===== 路由后处理 =====
def route_score(qa, seq):
    """问题定向→分支提权"""
    qv = Q256[IDS.index(qa)]
    q_t = float(qv @ DELTA_T)                    # 时间性
    q_n = float(qv @ NAME_AX) if NAME_AX is not None else 0.0  # 人名性
    qtext = (Q[qa].get("question") or "").lower()
    multi_hint = 1.0 if re.search(r"\ball\b|\bevery\b|how many|list|each", qtext) else 0.0
    head = seq[:15]
    tok = [toks(RAW[i]) for i in head]
    n = len(head)
    coh = np.zeros(n)
    for a2 in range(n):
        js = [len(tok[a2] & tok[b2]) / max(1, len(tok[a2] | tok[b2])) for b2 in range(n) if b2 != a2]
        coh[a2] = np.mean(js) if js else 0.0
    tp = REC_TPROJ[head]
    # z归一
    zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    bonus = np.zeros(n)
    bonus += max(0.0, q_t) * zs(tp) * 1.0          # 时间性问题→时间投影提权
    bonus += multi_hint * zs(coh) * 1.0            # 多金题→凝聚度启用(单金题不加=用户批评的修正)
    bonus += max(0.0, q_n) * 0.0                   # 人名性→(说话人匹配已在ranker特征, 暂不动)
    new_order_idx = list(np.argsort(-(np.arange(n) * 0.0 + bonus)))
    # 稳定融合: 原名次权重+bonus
    rank_sc = np.array([(n - r) / n for r in range(n)])
    mix = rank_sc + 0.35 * zs(bonus) + bonus * 0.0
    new_head = [head[i] for i in np.argsort(-mix)]
    return new_head + head[len(new_head):] + seq[15:] if False else new_head + [h for h in head if h not in set(new_head)] + seq[15:]

def score_with(process):
    a15 = c5 = 0
    for qa, grps in GS.items():
        seq = [MID2I[m] for m in FORD.get(qa, [])]
        seq2 = process(qa, seq) if process else seq
        if all(any(g in seq2[:15] for g in grp) for grp in grps):
            a15 += 1
        cg = core_group(qa, grps)
        if any(g in seq2[:5] for g in cg):
            c5 += 1
    return 100.0 * a15 / nG, 100.0 * c5 / nG

a, c = score_with(None)
P("v11原序:      all15=%5.1f%%  core5=%5.1f%%" % (a, c))
a, c = score_with(route_score)
P("v16向量路由:  all15=%5.1f%%  core5=%5.1f%%" % (a, c))
P("done %.0fs" % (time.time() - t0))
