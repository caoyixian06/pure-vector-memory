# -*- coding: utf-8 -*-
"""diag_surgery_split.py — 手术方向u的防泄漏验证
①拆半: 用一半会话的答案算u, 施在另一半会话上(交叉两向)
②无泄漏变体: u2=库内陈述句质心-问题质心(不用任何金答案)
指标: 错题证据进25 / 对题证据进5 / 平均名次 (只看被施术的那一半)
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

# 会话奇偶拆半
def half_of(i):
    sid = rows[i].get("sample_id") or ""
    try:
        return int(re.sub(r"\D", "", sid)) % 2
    except Exception:
        return 0
ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])
half = np.array([half_of(i) for i, _ in targets])

def centroid(M):
    c = M.mean(0)
    return c / np.linalg.norm(c)

def ev_ranks(idxs_mask, alpha, u):
    rk = []
    sel = np.where(idxs_mask)[0]
    for k in sel:
        i, hits = targets[k]
        qp = bQ[i] + alpha * u
        qp /= np.linalg.norm(qp)
        s = D @ qp
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk.append(min(pos[h] for h in hits) + 1)
    return np.array(rk), ok[sel]

def report(tag, mask, u, alpha=1.0):
    r0, o = ev_ranks(mask, 0.0, u)
    r1, _ = ev_ranks(mask, alpha, u)
    print("%-28s 基线: 错题进25 %d/%d 对题进5 %d 均名次%.0f" % (
        tag, (r0[~o] <= 25).sum(), (~o).sum(), (r0[o] <= 5).sum(), r0.mean()), flush=True)
    print("%-28s α=%.1f: 错题进25 %d/%d 对题进5 %d 均名次%.0f" % (
        "", alpha, (r1[~o] <= 25).sum(), (~o).sum(), (r1[o] <= 5).sum(), r1.mean()), flush=True)

qA = [j for j, (i, h) in enumerate(targets) if half[j] == 0]
qB = [j for j, (i, h) in enumerate(targets) if half[j] == 1]
a_half0 = bA[[i for i, _ in targets if half_of(i) == 0]]
a_half1 = bA[[i for i, _ in targets if half_of(i) == 1]]
q_half0 = bQ[[i for i, _ in targets if half_of(i) == 0]]
q_half1 = bQ[[i for i, _ in targets if half_of(i) == 1]]

print("== 拆半交叉(只用另一半的答案算方向) ==", flush=True)
u_A = centroid(a_half0) - centroid(q_half0)
report("方向来自会话偶数半→施于奇数半", half == 1, u_A)
u_B = centroid(a_half1) - centroid(q_half1)
report("方向来自会话奇数半→施于偶数半", half == 0, u_B)

print("== 无泄漏变体(u2=库内raw陈述质心-全部问题质心, 不用金答案) ==", flush=True)
raw_idx = [i for i in range(len(MID)) if KIND[i] == "raw"]
stmt_c = centroid(D[raw_idx[:3000]])
qc_all = centroid(bQ)
u2 = stmt_c - qc_all
report("u2 施于全体", np.ones(len(targets), dtype=bool), u2)
report("u2 施于偶半", half == 0, u2)
report("u2 施于奇半", half == 1, u2)
print("方向一致性: cos(u_A,u_B)=%.3f cos(u_A,u2)=%.3f" % (
    float(np.dot(u_A, u_B) / (np.linalg.norm(u_A) * np.linalg.norm(u_B))),
    float(np.dot(u_A, u2) / (np.linalg.norm(u_A) * np.linalg.norm(u2)))), flush=True)
