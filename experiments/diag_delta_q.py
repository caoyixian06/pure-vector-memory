# -*- coding: utf-8 -*-
"""diag_delta_q.py — Δ与问题的关联: Δ带不带'问题侧'信息?
P1 Δ的问题响应: 各题的(证据-噪声)逐题Δ_i 与 全局Δ 的关系; 逐题Δ_i与该题问题向量q_i的结构相关
P2 问题调制: score = Δ·rec + Σ_k w_k * (Δ_k*q_k) * rec_k  — Δ×q 交互项是否增益(拆半)
P3 256维空间同实验: qwen库向量挖Δ256, AUC对比1024版
P4 Δ与问题余弦 vs 噪声与问题余弦: Δ这个方向本身是不是'更像问题'的方向
"""
import io, json, os, sys, re, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
bQ = emb(Q)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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
MID2I = {m: i for i, m in enumerate(MID)}
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50 = z1["TOP50"]

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
print("matched:", len(targets), flush=True)

keys = sorted(targets.keys())
half = len(keys) // 2
def per_item(i):
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    return ev, no

# ---- P1: 逐题Δ_i vs 全局Δ; Δ_i 与 q_i ----
Dg = []
qi_delta_cos = []
for i in keys:
    ev, no = per_item(i)
    if not ev or len(no) < 3:
        continue
    di = D[ev].mean(0) - D[no].mean(0)
    Dg.append(di)
    qi_delta_cos.append(float(l2n(di[None])[0] @ bQ[i]))
Dg = np.stack(Dg)
Dg = l2n(Dg)
deltaG = l2n(Dg.mean(0)[None])[0]
print("P1a 逐题Δ_i与全局Δ cos: 均值%.3f (一致性)" % float(np.mean(Dg @ deltaG)), flush=True)
print("P1b 逐题Δ_i(归一化)与本题问题q cos: 均值%.3f (Δ带问题信息吗)" % float(np.mean(qi_delta_cos)), flush=True)
rng = np.random.default_rng(1)
qc_rand = []
for k in range(500):
    i = keys[rng.integers(0, len(keys))]
    j = keys[rng.integers(0, len(keys))]
    if i == j:
        continue
    ev, no = per_item(i)
    if not ev or len(no) < 3:
        continue
    di = l2n((D[ev].mean(0) - D[no].mean(0))[None])[0]
    qc_rand.append(float(di @ bQ[j]))
print("    对照: Δ_i与'别人家问题' cos=%.3f (差异即问题特异信息)" % float(np.mean(qc_rand)), flush=True)

# ---- P2: 交互项增益(拆半) ----
def auc(x, p, q):
    r = np.argsort(np.concatenate([x[p], x[q]]))
    ranks = np.empty_like(r, dtype=np.float64)
    ranks[r] = np.arange(1, len(r) + 1)
    return (ranks[:p.sum()].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * q.sum())

def collect(sel_keys):
    scores = {"delta": [], "dq": [], "qcos": [], "comb": []}
    labs = []
    for i in sel_keys:
        ev, no = per_item(i)
        if not ev or len(no) < 3:
            continue
        qv = bQ[i]
        for j in ev:
            scores["delta"].append(float(D[j] @ deltaG))
            scores["dq"].append(float((D[j] * qv) @ (deltaG * qv)))
            scores["qcos"].append(float(D[j] @ qv))
            scores["comb"].append(float(D[j] @ deltaG) + float((D[j] * qv) @ (deltaG * qv)))
            labs.append(1)
        for j in no[:8]:
            scores["delta"].append(float(D[j] @ deltaG))
            scores["dq"].append(float((D[j] * qv) @ (deltaG * qv)))
            scores["qcos"].append(float(D[j] @ qv))
            scores["comb"].append(float(D[j] @ deltaG) + float((D[j] * qv) @ (deltaG * qv)))
            labs.append(0)
    labs = np.array(labs, dtype=bool)
    return scores, labs

sA, lA = collect(keys[:half])
sB, lB = collect(keys[half:])
print("P2 拆半(A半算交互权重→直接B半测; 权重只有加法系数, 保守直接测):", flush=True)
for k in ("delta", "dq", "qcos", "comb"):
    print("  %-6s B半AUC=%.3f" % (k, auc(np.array(sB[k]), lB, ~lB)), flush=True)

# ---- P3: 256维空间的Δ ----
def ollama_embed(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
    return l2n(np.concatenate(out))

QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
wQ = ollama_embed(Q)
ev_rows, no_rows = [], []
for i in keys[:half]:
    ev, no = per_item(i)
    ev_rows += ev
    no_rows += no[:8]
delta256 = QW[np.array(ev_rows)].mean(0) - QW[np.array(no_rows)].mean(0)
delta256 /= np.linalg.norm(delta256)
evB, noB = [], []
for i in keys[half:]:
    ev, no = per_item(i)
    evB += ev
    noB += no[:8]
sev = QW[np.array(evB)] @ delta256
sno = QW[np.array(noB)] @ delta256
print("P3 256维Δ(留出): EVID分%.3f NOISE分%.3f AUC=%.3f  (1024维Δ=0.800)" % (
    sev.mean(), sno.mean(),
    auc(np.concatenate([sev, sno]), np.arange(len(sev)), np.arange(len(sev), len(sev) + len(sno)))), flush=True)

# ---- P4: Δ方向本身离问题近还是远 ----
print("P4 cos(Δ, 问题质心)=%.3f ; cos(Δ, 陈述质心)=%.3f" % (
    float(deltaG @ bQ.mean(0)), float(deltaG @ D[[i for i in range(N) if KIND[i] == 'raw'][:3000]].mean(0))), flush=True)
