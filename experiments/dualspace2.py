# -*- coding: utf-8 -*-
"""dualspace2.py — 双空间互证·词级侦察修正版
步骤:
S0 原子句逐词256向量预存(名词动词保留)
S1 去相关前置检验: 词级侦察分 vs 主通道分 的相关(必须<0.5)
S2 50错+50对: A原架构 / B=主通道+真词级侦察
S3 剂量扫描
"""
import io, json, os, sys, re, random
import time
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
def toks_list(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if len(w) > 2]
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

# 词向量表
zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
STOPW = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None

# S0: 原子句逐词向量
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
print("原子句:", len(atoms), flush=True)
# 逐词向量(内存注意: 7668句×~8词×256≈1500万float32≈60MB, 可以)
A2WV = []
sess2idx = {}
for k2, (p, sess, mid) in enumerate(atoms):
    vs = [wv(w) for w in toks_list(p)]
    vs = [v for v in vs if v is not None]
    A2WV.append(np.stack(vs) if vs else np.zeros((1, 256), dtype=np.float32))
    sess2idx.setdefault(sess, []).append(k2)
print("逐词向量预存完成", flush=True)

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

# 真词级侦察: 每题 问题词向量 → 每原子句的词级对齐分(逐词最大点积的均值)
def scout_score(i, sess_idx):
    qws = [wv(w) for w in toks_list(rows[i]["question"]) if w not in STOPW]
    qws = [v for v in qws if v is not None]
    if not qws:
        return {}
    Qm = np.stack(qws)
    out = {}
    for k in sess_idx:
        wv_m = A2WV[k]
        if wv_m.shape[0] == 1 and wv_m[0].sum() == 0:
            continue
        Am = Qm @ wv_m.T          # (nq, nc)
        out[k] = float(Am.max(axis=1).mean())   # 每个问题词的最佳对齐均值
    return out

# 样本与对照
rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

# S1: 去相关检验
sc_out, main_out = [], []
t0 = time.time()
for c, i in enumerate(sel):
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    sc = scout_score(i, idxs)
    if not sc:
        continue
    main = SBS[i]
    for k, v in sc.items():
        j = MID2I.get(atoms[k][2])
        if j is not None:
            sc_out.append(v)
            main_out.append(main[j])
    if c % 20 == 0:
        print("  scout", c, "%.0fs" % (time.time() - t0), flush=True)
cc = np.corrcoef(np.array(sc_out), np.array(main_out))[0, 1]
print("S1 去相关检验: cos/pearson(侦察分, 主分)=%.3f (须<0.5)" % cc, flush=True)

# S2: 对照实验
def order_dual(i, w, scout_cache):
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc_arr = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc_arr)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    sc = scout_cache.get(i, {})
    if sc and w > 0:
        zv = zs(np.array(list(sc.values()) + [0.0]))[:-1] if len(sc) > 1 else np.array(list(sc.values()))
        for k2, (k, v) in enumerate(sc.items()):
            j = MID2I.get(atoms[k][2])
            if j is not None:
                newf[j] = newf[j] + w * zv[min(k2, len(zv) - 1)]
    return np.argsort(-newf)

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
    print("  %-20s 错题进25 %2d/50  对题进5 %2d/50  均名次%.0f" % (tag, m25, o5, rsum / max(1, cnt)), flush=True)

scout_cache = {}
for i in sel:
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    scout_cache[i] = scout_score(i, sess2idx.get(sess, []))
print("== S2 对照(真词级侦察) ==", flush=True)
ev(lambda i: np.argsort(-(zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i]))), sel, "A: r37原架构")
for w in (2.0, 4.0):
    ev(lambda i, ww=w: order_dual(i, ww, scout_cache), sel, "B: 词级侦察w=%.1f" % w)
print("DUAL2_DONE", flush=True)
