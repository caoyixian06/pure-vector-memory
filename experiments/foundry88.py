# -*- coding: utf-8 -*-
"""foundry88.py — 软配对反向差的规律性检验: 逐题稳定性/独立性/组合判别力"""
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
XSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your really cool wow thanks hey yeah okay ok just been being get got getting lot bit kind sort pretty much many some all also too very there their here have has had having not no yes but so as by be am will would can could should may might dont didnt".split())

t0 = time.time()
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WORDS = list(WH.keys())
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

CV_CACHE = {}
def cand_words_idx(i):
    if i not in CV_CACHE:
        ws = set(stem(w) for w in re.findall(r"[a-z']+", RAW[i].lower()) if len(w) > 2)
        CV_CACHE[i] = np.array([W2I[w] for w in ws if w in W2I], dtype=np.int64)
    return CV_CACHE[i]

rows = []   # (sa, new_ratio, label, qa)
n_pair = n_noise_win = 0
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
    qt = toks(Q[qa]["question"])
    qts = set(stem(w) for w in qt)
    pos_sa = []
    neg_sa = []
    for r, i in enumerate(seq[:15]):
        cidx = cand_words_idx(i)
        if len(cidx) == 0:
            continue
        sa = float((qv @ WV[cidx].T).max(axis=1).mean())
        rt = toks(RAW[i])
        rts = set(stem(w) for w in rt)
        new_ratio = len((rts - qts) - XSTOP) / max(1, len(rts))
        lab = 1 if (i in goldset and r >= 5) else (0 if (i not in goldset and r < 5) else -1)
        if lab >= 0:
            rows.append((sa, new_ratio, lab))
        if lab == 1:
            pos_sa.append(sa)
        elif lab == 0:
            neg_sa.append(sa)
    if pos_sa and neg_sa:
        n_pair += 1
        if np.mean(neg_sa) > np.mean(pos_sa):
            n_noise_win += 1

arr = np.array(rows)
P("逐题配对检验: 噪>金的题 = %d/%d (%.1f%%)  [随机50%%]" % (n_noise_win, n_pair, 100.0 * n_noise_win / max(1, n_pair)))
P("金sa=%.3f 噪sa=%.3f  金new_ratio=%.3f 噪new_ratio=%.3f" % (
    arr[arr[:, 2] == 1][:, 0].mean(), arr[arr[:, 2] == 0][:, 0].mean(),
    arr[arr[:, 2] == 1][:, 1].mean(), arr[arr[:, 2] == 0][:, 1].mean()))
c = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
P("软配对 vs new_ratio 相关系数: %.3f" % c)

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

sa_p = arr[arr[:, 2] == 1][:, 0]; sa_n = arr[arr[:, 2] == 0][:, 0]
nr_p = arr[arr[:, 2] == 1][:, 1]; nr_n = arr[arr[:, 2] == 0][:, 1]
P("单信号AUC: soft(反向)=%.3f  new_ratio=%.3f" % (1 - auc(sa_p, sa_n), auc(nr_p, nr_n)))
# 组合: -soft + new_ratio 简单等权和
zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)
comb = zs(-arr[:, 0]) + zs(arr[:, 1])
comb_p = comb[arr[:, 2] == 1]; comb_n = comb[arr[:, 2] == 0]
P("组合(-soft + new_ratio) AUC: %.3f" % auc(comb_p, comb_n))
P("done %.0fs" % (time.time() - t0))
