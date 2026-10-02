# -*- coding: utf-8 -*-
"""rank_shift.py — 窗内名次爬升的正确度量
对50错+50对: 基线vs三路先验, 量证据名次分布(中位/top5率/top1率/平均名次)
分错题/对题两看
"""
import io, json, os, sys, re, random
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def toks_list(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if len(w) > 2]

# 复用three_prior的priors逻辑(重新内联)
zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None

atoms = []
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") != "raw":
        continue
    s = (r.get("raw") or "").strip()
    if s.startswith("[Session"):
        continue
    sess = r.get("session_id") or ""
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if len(words) <= 5 and re.search(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank)\b", p, re.I):
            continue
        atoms.append((p, sess, r.get("memory_id")))
A2WV = []
sess2idx = {}
for k2, (p, sess, mid) in enumerate(atoms):
    vs = [wv(w) for w in toks_list(p)]
    vs = [v for v in vs if v is not None]
    A2WV.append(np.stack(vs) if vs else np.zeros((1, 256), dtype=np.float32))
    sess2idx.setdefault(sess, []).append(k2)
A2HOST = [a[2] for a in atoms]
A2TOK = [toks_set(a[0]) for a in atoms]
print("原子句:", len(atoms), flush=True)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
MID2I = {m: i for i, m in enumerate(MID)}
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits:
        targets[i] = hits
keys = sorted(targets.keys())
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
bQ = emb([r["question"] for r in rows])
bQ = l2n(bQ)
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQu = l2n(bQ + u2)
SBS = (bQu @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# Δ̂近邻库(前半训练)
train_deltas = {}
for i in keys[:len(keys) // 2]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:6]
    if not ev or len(no) < 3:
        continue
    di = D[ev].mean(0) - D[no].mean(0)
    n2 = np.linalg.norm(di) + 1e-9
    train_deltas[i] = di / n2
train_ids = list(train_deltas.keys())
TRM = np.stack([bQu[i2] for i2 in train_ids])
TRD = np.stack([train_deltas[i2] for i2 in train_ids])

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok

STOPW2 = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())

def priors(i):
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sid if False else sess, [])
    if not idxs:
        return {}, {}, {}
    sims = TRM @ bQu[i]
    nn = np.argsort(-sims)[:10]
    dhat = TRD[nn].mean(0)
    dhat /= np.linalg.norm(dhat) + 1e-9
    p1 = {}
    for k in idxs:
        j = MID2I.get(A2HOST[k])
        if j is not None:
            p1[j] = float(D[j] @ dhat)
    qws = [wv(w) for w in toks_list(rows[i]["question"]) if w not in STOPW2]
    qws = [v for v in qws if v is not None]
    Qm = np.stack(qws) if qws else None
    simsA = np.zeros(len(idxs), dtype=np.float32)
    for k2, k in enumerate(idxs):
        wvm = A2WV[k]
        if wvm.shape[0] == 1 and wvm[0].sum() == 0 or Qm is None:
            continue
        simsA[k2] = float((Qm @ wvm.T).max(axis=1).mean())
    p2 = {}
    top_a = np.argsort(-simsA)[:5]
    for oi in top_a:
        k = idxs[oi]
        for nb in (k - 1, k + 1):
            if 0 <= nb < len(atoms) and atoms[nb][1] == atoms[k][1]:
                j = MID2I.get(A2HOST[nb])
                if j is not None:
                    p2[j] = max(p2.get(j, 0), float(simsA[oi]))
    qtok = toks_set(rows[i]["question"])
    p3 = {}
    for k in idxs:
        j = MID2I.get(A2HOST[k])
        if j is not None:
            p3[j] = float(len(A2TOK[k] & qtok))
    return p1, p2, p3

CACHE = {}
for c, i in enumerate(sel):
    CACHE[i] = priors(i)

# Δ全局方向 + 近邻Δ̂库
evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ_D = (D @ DELTA).astype(np.float32)
print("priors cached", flush=True)

def rank_stats(order_fn, sel_list, tag):
    ranks_miss, ranks_ok = [], []
    top1_hit = top5_hit = cnt = 0
    for i in sel_list:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        allh = hs | tws
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        rk = min(pos[h2] for h2 in allh) + 1
        cnt += 1
        if rk <= 1 and allh and (order[0] in allh):
            top1_hit += 1
        if rk <= 5:
            top5_hit += 1
        (ranks_ok if rows[i].get("llm_score") == 1 else ranks_miss).append(rk)
    rm = np.array(ranks_miss) if ranks_miss else np.array([0])
    ro = np.array(ranks_ok) if ranks_ok else np.array([0])
    print("  %-22s 错题名次中位%.0f(前5率%.0f%%) 对题名次中位%.0f(前5率%.0f%%) top1=%d" % (
        tag, np.median(rm), 100 * (rm <= 5).mean(), np.median(ro), 100 * (ro <= 5).mean(), top1_hit), flush=True)

def base_order(i):
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j2, -7.0) for j2 in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return np.argsort(-newf)

def prior_order(i, w1, w2, w3, w_delta=0.6):
    b = base_order(i)
    newf = np.zeros(N, dtype=np.float32)
    newf[b[:50]] = np.linspace(50, 1, 50)
    p1, p2, p3 = CACHE[i]
    for j, v in p1.items():
        newf[j] += w1 * v * 3
    for j, v in p2.items():
        newf[j] += w2 * v
    for j, v in p3.items():
        newf[j] += w3 * v
    newf += w_delta * zs(PROJ_D)
    return np.argsort(-newf)

print("== 窗内名次爬升正确度量(50错+50对) ==", flush=True)
rank_stats(base_order, sel, "基线r37")
rank_stats(lambda i: prior_order(i, 1, 2, 0, 0), sel, "P1+P2 (无Δ)")
rank_stats(lambda i: prior_order(i, 0, 2, 2, 0), sel, "P2+P3 (无Δ)")
rank_stats(lambda i: prior_order(i, 1, 2, 2, 0), sel, "P1+P2+P3 (无Δ)")
rank_stats(lambda i: prior_order(i, 1, 2, 2, 0.6), sel, "全开+Δ0.6")
rank_stats(lambda i: prior_order(i, 1, 2, 2, 1.2), sel, "全开+Δ1.2")
print("RANKSHIFT_DONE", flush=True)
