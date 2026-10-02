# -*- coding: utf-8 -*-
"""why_conf_drop.py — 对题证据置信度为什么会降低: 逐题解剖
H1 置信度的三成分里, 对题证据在哪一项上输给了噪声?
H2 被挤出的对题证据: 挤掉它的是什么样的句子(高置信噪声长什么样)?
H3 置信度的归一化缺陷: max归一化导致每题内部相对化, 弱信号题全部失真?
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
PROJ_D = (D @ (D[np.array([j for i in keys[:100] for j in TOP50[i][:4]])].mean(0))).astype(np.float32)
PROJ_D /= (np.linalg.norm(PROJ_D) + 1e-9)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)

def conf_decomp(i):
    """返回 对题证据在置信度三成分上的分解 + 被挤掉的机制"""
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
    ab_max = max(atom_b.values()) if atom_b else 1.0
    top = np.argsort(-(zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])))[:50]
    info = []
    for j in top:
        ab = atom_b.get(j, 0.0) / max(1e-9, ab_max)
        dp = float(PROJ_D[j])
        dpn = max(0.0, min(1.0, dp + 0.2))
        cov = len(toks_set(TEXTS[j]) & qtok) / max(1, len(qtok))
        conf = 0.4 * ab + 0.3 * dpn + 0.3 * min(1.0, cov)
        isev = j in hs if False else None
        info.append((j, conf, ab, dpn, cov))
    return info, top

# H1: 对题的 证据成分 vs 噪声成分 (对题子集)
print("== H1 对题: 三成分分解(证据句 vs 池内噪声) ==", flush=True)
from collections import defaultdict
comp = defaultdict(lambda: ([], []))  # 成分名 -> (证据侧, 噪声侧)
n_ok = 0
for i in sel_ok:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    info, top = conf_decomp(i)
    for j, conf, ab, dpn, cov in info[:20]:
        ise = j in hs or j in tws
        for nm, v in (("原子boost", ab), ("Δ腔", dpn), ("题词覆盖", cov), ("总conf", conf)):
            comp[nm][0 if ise else 1].append(v)
        n_ok += 0
print("  成分        证据句均值   噪声句均值   差")
for nm in ("原子boost", "Δ腔", "题词覆盖", "总conf"):
    e, n2 = comp[nm]
    print("  %-10s %.3f      %.3f      %+.3f" % (nm, np.mean(e), np.mean(n2), np.mean(e) - np.mean(n2)), flush=True)

# H2: 被挤出的对题证据长什么样(高置信噪声解剖)
print()
print("== H2 被挤出案例(对题, w=8时证据掉出top5) ==", flush=True)
shown = 0
for i in sel_ok:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    info, top = conf_decomp(i)
    info_sorted = sorted(info, key=lambda x: -x[1])
    ev_in = [x for x in info_sorted if (x[0] in hs or x[0] in tws)]
    if not ev_in:
        continue
    ev_best = ev_in[0]
    rank_in = info_sorted.index(ev_best) + 1
    if rank_in <= 2 or shown >= 3:
        continue
    shown += 1
    top1 = info_sorted[0]
    print("  Q:", rows[i]["question"][:60], flush=True)
    print("    证据句 rank%d conf=%.2f (atom%.2f Δ%.2f cov%.2f):" % (
        rank_in, ev_best[1], ev_best[2], ev_best[3], ev_best[4]), TEXTS[ev_best[0]][:80], flush=True)
    print("    top1噪声 conf=%.2f (atom%.2f Δ%.2f cov%.2f):" % (
        top1[1], top1[2], top1[3], top1[4]), TEXTS[top1[0]][:80], flush=True)
