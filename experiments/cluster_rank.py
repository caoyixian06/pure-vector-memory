# -*- coding: utf-8 -*-
"""cluster_rank.py — 簇信号无答案重排(50错+50对)
信号(全部不含答案信息):
  S1 相邻性: 候选与top3种子同turn相邻或同session
  S2 向量互聚: 候选与top3种子的cos
  S3 实体桥: 候选与种子的实体词共现
  S4 词干覆盖(问题词) — 问题侧合法信号
合成: rank提升 = f(S1,S2,S3,S4), 50错+50对: 错题进25/对题进5/均名次
对照: r37基线
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
def toks_set(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

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
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

def cluster_order(i, w_cluster=2.0, w_pen=0.5):
    # A: r37基线排序
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    order = np.argsort(-newf)
    # 簇信号: 种子=新序top3
    seeds = order[:3]
    seed_vec = D[seeds]
    seed_tok = [toks_set(TEXTS[j]) for j in seeds]
    seed_ent = [set(re.findall(r"\b[A-Z][a-z]{2,}\b", TEXTS[j])) for j in seeds]
    newf2 = newf.copy()
    qtok = toks_set(rows[i]["question"])
    for r0, j in enumerate(order):
        t = TEXTS[j]
        s1 = 1.0 if any(CONVKEY[j] == CONVKEY[s0] for s0 in seeds) else 0.0
        s2 = float(np.max(seed_vec @ D[j]))
        s3 = 0.0
        jt = set(re.findall(r"\b[A-Z][a-z]{2,}\b", t))
        for se in seed_ent:
            if jt & se:
                s3 = 1.0
                break
        s4 = len(toks_set(t) & qtok) * 0.2
        if r0 >= 3:  # 种子本身不动
            newf2[j] = newf2[j] + w_cluster * (0.6 * s1 + 0.8 * (s2 - 0.65) + 0.4 * s3) + w_pen * (1.0 if t.rstrip().endswith("?") or COUR.search(t) else 0.0) * -1
    return np.argsort(-newf2)
COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)

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
    print("  %-22s 错题进25 %2d/50  对题进5 %2d/50  均名次%.0f" % (tag, m25, o5, rsum / max(1, cnt)), flush=True)

print("== 簇信号无答案重排(50错+50对) ==", flush=True)
ev(lambda i: np.argsort(-(zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i]))), sel, "A: r37基线")
for w in (1.0, 2.0, 4.0):
    ev(lambda i, ww=w: cluster_order(i, ww), sel, "B: 簇信号w=%.1f" % w)
print("CLUSTERRANK_DONE", flush=True)
