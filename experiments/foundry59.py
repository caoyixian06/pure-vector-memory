# -*- coding: utf-8 -*-
"""foundry59.py — 隐含表达型的cos优势普适性检验(用户: "cos微弱优势是不是全部存在")
定义"隐含表达型": 真金与问题的词面交集(G1呼应/Jaccard)低于阈值
测: 这类题里, 256cos/1024cos/avgcos 的"真金>假金"胜率
  - 若显著>50%: cos优势普适存在 → 加权可用(弱但一致的信号)
  - 若≈50%: 无规律 → 词面全灭时向量也灭, 真正的盲区
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry59_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

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

MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
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
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

def g1_of(qa, c):
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
    return len(qstems & rstems) / max(1, len(qstems))

# 收集: 头部30名内 每题的 (真金cos, 假金cos最大, G1) — 分层统计
buckets = {"G1<0.1(全隐含)": [], "0.1<=G1<0.25": [], "G1>=0.25(有词面)": []}
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    G = GSETS[k_i]
    if not G:
        continue
    order = np.argsort(-FINAL[qi])
    head = list(order[:30])
    golds_h = [c for c in head if c in G]
    fakes_h = [c for c in head if c not in G]
    if not golds_h or not fakes_h:
        continue
    for g in golds_h:
        g1 = g1_of(qa, g)
        v_g = float(Q256[qi] @ QW[g])
        d_g = float(D[g] @ X[qi])
        # 该题头部假金的cos均值与最大
        v_f = max(float(Q256[qi] @ QW[f]) for f in fakes_h[:15])
        d_f = max(float(D[f] @ X[qi]) for f in fakes_h[:15])
        if g1 < 0.1:
            buckets["G1<0.1(全隐含)"].append((v_g, d_g, v_f, d_f))
        elif g1 < 0.25:
            buckets["0.1<=G1<0.25"].append((v_g, d_g, v_f, d_f))
        else:
            buckets["G1>=0.25(有词面)"].append((v_g, d_g, v_f, d_f))

P("===== cos优势的分层普适性 (真金 vs 头部最强假金) =====")
for name, arr in buckets.items():
    if not arr:
        P("%-18s n=0" % name)
        continue
    A = np.array(arr)
    win256 = float(np.mean(A[:, 0] > A[:, 2]))
    win1024 = float(np.mean(A[:, 1] > A[:, 3]))
    winavg = float(np.mean((A[:, 0] + A[:, 1]) / 2 > (A[:, 2] + A[:, 3]) / 2))
    margin256 = float(np.mean(A[:, 0] - A[:, 2]))
    P("%-18s n=%4d | 256胜率=%.1f%% 1024胜率=%.1f%% avg胜率=%.1f%% | 256优势均值=%.3f" % (
        name, len(arr), 100 * win256, 100 * win1024, 100 * winavg, margin256))
P("\n判读: 胜率>55%=规律存在可加权; 50%±3%=无规律真盲区")
P("F59_DONE")
LOG.close()
