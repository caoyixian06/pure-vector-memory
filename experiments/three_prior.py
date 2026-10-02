# -*- coding: utf-8 -*-
"""three_prior.py — 三路先验合体重排(50错+50对)
P1 方向先验: Δ̂(近邻k=10推导)投影
P2 位置先验: 邻句(种子句的+1/-1邻接句)
P3 内容先验: 题目词覆盖(词干)
基线: r37; 单路; 双路; 三路全开
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

# 原子句库
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
print("原子句:", len(atoms), flush=True)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

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

# P1 训练Δ库(前半)
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
print("训练Δ:", len(train_ids), flush=True)

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok

def priors(i):
    """返回 (P1方向投影dict, P2邻句dict, P3词覆盖dict) 都在原子句宿主记录上"""
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    if not idxs:
        return {}, {}, {}
    # P1: Δ̂
    sims = TRM @ bQu[i]
    nn = np.argsort(-sims)[:10]
    dhat = TRD[nn].mean(0)
    dhat /= np.linalg.norm(dhat) + 1e-9
    p1 = {}
    for k in idxs:
        j = MID2I.get(A2HOST[k])
        if j is not None:
            p1[j] = float(D[j] @ dhat)
    # P2: 邻句(基于句级余弦种子)
    simsA = A2WV_ALIGN(i, idxs)
    p2 = {}
    top_a = np.argsort(-simsA)[:5]
    for oi in top_a:
        k = idxs[oi]
        for nb in (k - 1, k + 1):
            if 0 <= nb < len(atoms) and atoms[nb][1] == atoms[k][1]:
                j = MID2I.get(A2HOST[nb])
                if j is not None:
                    p2[j] = max(p2.get(j, 0), float(simsA[oi]))
    # P3: 词覆盖
    qtok = toks_set(rows[i]["question"])
    p3 = {}
    for k in idxs:
        j = MID2I.get(A2HOST[k])
        if j is not None:
            p3[j] = float(len(A2WV[k2idx(k)] if False else []) ) if False else float(len(toks_set(atoms[k][0]) & qtok))
    return p1, p2, p3

def A2WV_ALIGN(i, idxs):
    qws = [wv(w) for w in toks_list(rows[i]["question"]) if w not in STOPW2]
    qws = [v for v in qws if v is not None]
    if not qws:
        return np.zeros(len(idxs), dtype=np.float32)
    Qm = np.stack(qws)
    out = np.zeros(len(idxs), dtype=np.float32)
    for k2, k in enumerate(idxs):
        wvm = A2WV[k]
        if wvm.shape[0] == 1 and wvm[0].sum() == 0:
            continue
        out[k2] = float((Qm @ wvm.T).max(axis=1).mean())
    return out

STOPW2 = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())

def k2idx(k):
    return k

def ev(order_fn, sel_list, tag):
    m25 = o5 = cnt = 0
    for i in sel_list:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        allh = list(hs) + list(tws)
        if not all(h2 in pos for h2 in allh):
            continue
        rk = min(pos[h2] for h2 in allh) + 1
        cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-24s 错题进25 %2d/50  对题进5 %2d/50" % (tag, m25, o5), flush=True)

def base(i):
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j2, -7.0) for j2 in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return np.argsort(-newf), newf, top

print("== 三路先验合成(50错+50对) ==", flush=True)
# 预计算
CACHE = {}
for c, i in enumerate(sel):
    p1, p2, p3 = priors(i)
    CACHE[i] = (p1, p2, p3)
    if c % 20 == 0:
        print("  priors", c, flush=True)

ev(lambda i: base(i)[0], sel, "基线r37")
def with_prior(i, w1, w2, w3):
    b, newf, top = base(i)
    p1, p2, p3 = CACHE[i]
    newf2 = newf.copy()
    for j, v in p1.items():
        newf2[j] += w1 * v * 3
    for j, v in p2.items():
        newf2[j] += w2 * v
    for j, v in p3.items():
        newf2[j] += w3 * v
    return np.argsort(-newf2)
ev(lambda i: with_prior(i, 0, 2, 0), sel, "P2邻句单路")
ev(lambda i: with_prior(i, 0, 0, 2), sel, "P3覆盖单路")
ev(lambda i: with_prior(i, 1, 2, 0), sel, "P1+P2")
ev(lambda i: with_prior(i, 1, 2, 2), sel, "P1+P2+P3全开")
ev(lambda i: with_prior(i, 1, 0, 2), sel, "P1+P3")
print("TRI_DONE", flush=True)
