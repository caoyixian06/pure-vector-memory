# -*- coding: utf-8 -*-
"""diag_surgery_net.py — 维度手术净值测试: 对全部1382题施术, 数捞回与砸坏
q' = normalize(q + α*(答案质心-问题质心)), 对比对题/错题的证据名次变化
"""
import io, json, os, sys, re
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
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

Q = [r["question"] for r in rows]
A = [str(r["answer"][0]) for r in rows]
bQ = emb(Q); bA = emb(A)
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        rr = json.loads(l)
        REC[rr.get("memory_id")] = rr
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

qc = bQ.mean(0); qc /= np.linalg.norm(qc)
ac = bA.mean(0); ac /= np.linalg.norm(ac)
u = ac - qc

# 每题的证据索引
targets = []
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        targets.append((i, hits))
print("可匹配证据的题:", len(targets), flush=True)

ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])

def ranks_for(alpha):
    out = []
    for i, hits in targets:
        qp = bQ[i] + alpha * u
        qp /= np.linalg.norm(qp)
        s = D @ qp
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        out.append(min(pos[h] for h in hits) + 1)
    return np.array(out)

print("%-6s | 错题证据进25 | 对题证据进5 | 对题证据进25 | 对题被砸出5 | 对题被砸出25 | 全体平均名次" % "α", flush=True)
base5 = base25 = None
for alpha in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
    rk = ranks_for(alpha)
    m_ok = rk[ok]; m_mi = rk[~ok]
    if alpha == 0.0:
        base5 = m_ok <= 5; base25 = m_ok <= 25
    broken5 = int((base5 & (m_ok > 5)).sum())
    broken25 = int((base25 & (m_ok > 25)).sum())
    print("%.2f  | %3d/%d  | %3d  | %3d  | %3d  | %3d  | %.0f" % (
        alpha, (m_mi <= 25).sum(), (~ok).sum(), (m_ok <= 5).sum(), (m_ok <= 25).sum(),
        broken5, broken25, rk.mean()), flush=True)
