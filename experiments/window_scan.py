# -*- coding: utf-8 -*-
"""window_scan.py — 窗口大小扫描: 5/8/10/15/20/25
测: ①错题证据入窗率 ②对题证据入窗率 ③窗内噪声条数(GLM注意力负担)
④ 模拟读取成功率: 窗内证据排位×窗大小的联合指标
关键: 用r39全家福排序(当前最优)做基础
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

# 原子句(v2)
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

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

# r39式排序(含原子句域)
def r39_order(i):
    q = rows[i]
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    sid = q.get("sample_id") or ("conv-" + str(q.get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess2idx.get(sess, [])
    qtok = toks_set(q["question"])
    atom_b = {}
    if idxs:
        sims = A2VEC[idxs] @ bQ[i]
        cov = np.array([len(A2TOKX[idxs[k2]] & qtok) if False else 0 for k2 in range(len(idxs))], dtype=np.float32)
        for oi in np.argsort(-sims)[:6]:
            j = MID2I.get(atoms[idxs[oi]][2])
            if j is not None:
                atom_b[j] = max(atom_b.get(j, 0), float(sims[oi]))
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j2, -7.0) for j2 in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    for j, ab in atom_b.items():
        if j < N:
            newf[j] += 1.5 * ab
    return np.argsort(-newf)
A2TOKX = [toks_set(a[0]) for a in atoms]
ORD = {i: r39_order(i) for i in sel}
print("排序完成", flush=True)

print("== 窗口扫描(r39排序) ==", flush=True)
print("窗口 | 错题证据入窗 | 对题证据入窗 | 平均噪声条数(错题) | 平均证据条数", flush=True)
for W in (5, 8, 10, 15, 20, 25):
    m_in = o_in = cnt_m = cnt_o = 0
    noise = evc = 0
    for i in sel:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        allh = hs | tws
        order = ORD[i]
        win = set(order[:W].tolist())
        found = bool(allh & win)
        noise_n = len(win - allh)
        if rows[i].get("llm_score") == 1:
            o_in += found
            cnt_o += 1
            evc += len(allh & win)
        else:
            m_in += found
            cnt_m += 1
            noise += noise_n
    print("  %3d | %2d/%d (%.0f%%) | %2d/%d (%.0f%%) | %.1f | %.2f" % (
        W, m_in, cnt_m, 100 * m_in / max(1, cnt_m), o_in, cnt_o, 100 * o_in / max(1, cnt_o),
        noise / max(1, cnt_m), evc / max(1, cnt_o)), flush=True)
print("WINDOWSCAN_DONE", flush=True)
