# -*- coding: utf-8 -*-
"""dualspace.py — 用户设计落地: 256侦察→1024主力互证检索
50错题+50对题(从r37各抽), 双条件对照:
  A: r37原架构排序(融合+精排)
  B: A + 双空间互证(问题词256近邻→原子句命中→宿主1024加成)
指标: 错题证据进25(救回数) / 对题证据保持在窗(砸坏数) / 均名次
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

# 原子句库(v2无过滤版)
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
A2VEC = emb([a[0] for a in atoms])
A2SESS = [a[1] for a in atoms]
A2HOST = [a[2] for a in atoms]
A2TOK = [toks_set(a[0]) for a in atoms]
sess2idx = {}
for idx, a in enumerate(atoms):
    sess2idx.setdefault(a[1], []).append(idx)
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
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# 50错+50对
rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

# 256侦察: 问题词向量 → 原子句词集合的语义命中 → 宿主记录加成
def recon_boost(i):
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    if not idxs:
        return {}
    qwords = [w for w in re.findall(r"[a-z']+", rows[i]["question"].lower()) if len(w) > 3]
    boost = {}
    for w in qwords:
        v = None
        i2 = None
        # 词向量: 从word_vecs查
        # (用WH键查)
        pass
    return boost

# 更直接的256侦察: 问题向量(256 qwen)→原子句需要256向量——改为: 问题词干→原子句词干覆盖(词级)
# + 问题1024→原子句1024(句级) 两级侦察
def dualspace_order(i, boost_w=3.0):
    # A: r37原架构
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    orderA = np.argsort(-newf)
    # B: 双空间互证
    # 侦察1(句级1024): 原子句句向量 vs 问题向量, 命中top8原子句
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    boost = {}
    if idxs:
        sims = A2VEC[idxs] @ bQ[i]
        qtok = toks_set(rows[i]["question"])
        for oi in np.argsort(-sims)[:8]:
            at_t = A2TOK[idxs[oi]]
            # 侦察2(词级256近似): 词干覆盖 + 词面向量近邻(问题词vs原子句词, 用词向量)
            overlap = len(at_t & qtok)
            # 词向量近邻: 原子句词向量与问题词向量的最大点积(需词向量; 简化: 用原子句向量与问题向量差)
            host_mid = A2HOST[idxs[oi]]
            j = MID2I.get(host_mid)
            if j is None:
                continue
            score = float(sims[oi]) + 0.3 * overlap
            boost[j] = max(boost.get(j, 0), score)
    newf2 = newf.copy()
    for j, s2 in boost.items():
        newf2[j] = newf2[j] + boost_w * s2
    return np.argsort(-newf2), np.argsort(-newf)

def ev(order_fn, sel_list, tag):
    m25 = o5 = cnt = 0
    rsum = 0
    for i in sel_list:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        allh = list(hs) + list(tws)
        if not all(h2 in pos for h2 in allh):
            continue
        rk = min(pos[h2] for h2 in allh) + 1
        rsum += rk; cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-22s 错题进25 %2d/50  对题进5 %2d/50  均名次%.0f (n=%d)" % (
        tag, m25, o5, rsum / max(1, cnt), cnt), flush=True)
    return m25, o5

print("== 50错+50对 对照实验 ==", flush=True)
ev(lambda i: dualspace_order(i)[1], sel, "A: r37原架构")
ev(lambda i: dualspace_order(i)[0], sel, "B: +双空间互证w=3")
print("  -- 错题子集 --", flush=True)
ev(lambda i: dualspace_order(i)[1], sel_miss, "A: r37(仅错题)")
ev(lambda i: dualspace_order(i)[0], sel_miss, "B: 互证(仅错题)")
print("  -- 对题子集 --", flush=True)
ev(lambda i: dualspace_order(i)[1], sel_ok, "A: r37(仅对题)")
ev(lambda i: dualspace_order(i)[0], sel_ok, "B: 互证(仅对题)")
for w in (1.5, 6.0):
    ev(lambda i, ww=w: dualspace_order(i, ww)[0], sel, "B: 互证w=%.1f" % w)
print("DUALSPACE_DONE", flush=True)
