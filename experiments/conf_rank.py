# -*- coding: utf-8 -*-
"""conf_rank.py — 标注置信度重排(50错+50对)
置信度 = 三信号合成分(原子boost + Δ投影 + 题词覆盖), 每题top50内计算
排序 = r37底盘 + w × 置信度, w剂量扫描
指标: 错题证据进25(救回) / 对题证据进5保持 / 对题证据名次中位(降名次代价)
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

# 原子句域(同r38)
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
sess2idx = {}
for k2, a in enumerate(atoms):
    sess2idx.setdefault(a[1], []).append(k2)
print("原子句:", len(atoms), flush=True)

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

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

def conf_scores(i):
    """三信号合成置信度: 原子boost(0-1) + Δ投影归一 + 题词覆盖(0-1), 各0.33权重"""
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    qtok = toks_set(rows[i]["question"])
    atom_b = {}
    if idxs:
        sims = A2VEC[idxs] @ bQ[i]
        for oi in np.argsort(-sims)[:6]:
            j = MID2I.get(atoms[idxs[oi]][2])
            if j is not None:
                atom_b[j] = max(atom_b.get(j, 0), float(sims[oi]))
    out = {}
    top = np.argsort(-(zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])))[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    rer_med = np.median([rer_map.get(j, -7.0) for j in top])
    for j in top:
        ab = atom_b.get(j, 0.0) / max(1e-9, max(atom_b.values()) if atom_b else 1)
        dp = float(PROJ_D[j])
        cov = len(toks_set(TEXTS[j]) & qtok) / max(1, len(qtok))
        # Δ归一: >0为证据腔
        dpn = max(0.0, min(1.0, dp + 0.2)) * 0.15
        isq = 0.3 if TEXTS[j].rstrip().endswith('?') else 1.0
        out[j] = 0.5 * min(1.0, cov) + 0.35 * ab * isq + dpn
    return out, rer_map, top, rer_med

PROJ_D = (D @ (D[np.array([j for i in keys[:100] for j in TOP50[i][:4]])].mean(0))).astype(np.float32)
PROJ_D = PROJ_D / (np.linalg.norm(PROJ_D) + 1e-9)

def ev(order_fn, sel_list, tag):
    m25 = o5 = cnt = 0
    medranks = []
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
        medranks.append(rk)
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    import numpy as _np
    med = _np.median(medranks) if medranks else -1
    print("  %-24s 错题进25 %2d/50  对题进5 %2d/50  名次中位%.0f (n=%d)" % (
        tag, m25, o5, med, cnt), flush=True)

def base_order(i):
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j2, -7.0) for j2 in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return np.argsort(-newf)

print("== 标注置信度重排(50错+50对) ==", flush=True)
ev(lambda i: base_order(i), sel, "A: r37基线")
for w in (1.0, 2.0, 4.0, 8.0):
    def conf_order(i, ww=w):
        cs, rer_map, top, rm = conf_scores(i)
        b = base_order(i)
        newf = np.zeros(N, dtype=np.float32)
        newf[b[:50]] = np.linspace(50, 1, 50)
        for j in top:
            newf[j] += ww * 3.0 * cs[j]
        return np.argsort(-newf)
    ev(lambda i, ff=conf_order: ff(i), sel, "B: 置信度w=%.1f" % w)
print("CONF_DONE", flush=True)
