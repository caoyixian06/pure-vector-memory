# -*- coding: utf-8 -*-
"""foundry90.py — 反向规律边界解剖: 55个"金>=噪"题 vs 133个"噪>金"题的结构对比"""
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
WHd = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WHd.keys())
WV = l2n(np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)[:len(WORDS)])
W2I = {w: i for i, w in enumerate(WORDS)}
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

groups_A = []   # 噪>金 (反向成立)
groups_B = []   # 金>=噪 (反向失效)
for qa in IDS:
    grps = gold_groups(qa)
    if not grps:
        continue
    seq = [MID2I[m] for m in FORD.get(qa, [])]
    goldset = set()
    for g in grps:
        goldset |= g
    qws = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    qidx = np.array([W2I[w] for w in qws if w in W2I], dtype=np.int64)
    if len(qidx) == 0:
        continue
    qv = WV[qidx]
    pos_sa, neg_sa = [], []
    gold_g1 = []
    for r, i in enumerate(seq[:15]):
        ws_c = set(stem(w) for w in re.findall(r"[a-z']+", RAW[i].lower()) if len(w) > 2)
        ci = np.array([W2I[w] for w in ws_c if w in W2I], dtype=np.int64)
        if len(ci) == 0:
            continue
        sa = float((qv @ WV[ci].T).max(axis=1).mean())
        if i in goldset:
            if r >= 5:
                pos_sa.append(sa)
            gold_g1.append(len(qws & ws_c) / max(1, len(qws)))
        else:
            if r < 5:
                neg_sa.append(sa)
    if not pos_sa or not neg_sa:
        continue
    first = (Q[qa].get("question") or "x").strip().lower().split()[0]
    cat = str(Q[qa].get("category"))
    passed = all(any(g in seq[:5] for g in grp) for grp in grps)
    row = (first, cat, len(grps), float(np.mean(gold_g1)), passed, np.mean(neg_sa) - np.mean(pos_sa))
    (groups_A if np.mean(neg_sa) > np.mean(pos_sa) else groups_B).append(row)

P("反向成立组 n=%d  失效组 n=%d" % (len(groups_A), len(groups_B)))
for nm, G in (("噪>金(成立)", groups_A), ("金>=噪(失效)", groups_B)):
    firsts = {}
    for r in G:
        firsts[r[0]] = firsts.get(r[0], 0) + 1
    top3 = sorted(firsts.items(), key=lambda x: -x[1])[:3]
    P("\n[%s] 首词top3=%s  when占比=%.1f%%" % (
        nm, str(top3), 100.0 * sum(1 for r in G if r[0] == "when") / len(G)))
    P("  平均金G1=%.3f  单组占比=%.1f%%  @5通过率=%.1f%%  平均(噪-金)=%.4f" % (
        np.mean([r[3] for r in G]), 100.0 * sum(1 for r in G if r[2] == 1) / len(G),
        100.0 * sum(1 for r in G if r[4]) / len(G), np.mean([r[5] for r in G])))
# G1分界: 失效组的金G1是否系统性更高
gA = [r[3] for r in groups_A]
gB = [r[3] for r in groups_B]
P("\n金G1: 成立组=%.3f  失效组=%.3f  (失效组的金词面呼应更强?)" % (np.mean(gA), np.mean(gB)))
hi = [r for r in groups_A + groups_B if r[3] > 0.5]
lo = [r for r in groups_A + groups_B if r[3] <= 0.5]
P("金G1>0.5的题: 失效占比=%.1f%% (n=%d);  金G1<=0.5: 失效占比=%.1f%% (n=%d)" % (
    100.0 * sum(1 for r in hi if r in groups_B) / max(1, len(hi)), len(hi),
    100.0 * sum(1 for r in lo if r in groups_B) / max(1, len(lo)), len(lo)))
P("done %.0fs" % (time.time() - t0))
