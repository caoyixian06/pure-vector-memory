# -*- coding: utf-8 -*-
"""check_farmerge.py — 兄弟优势 vs 会话距离: 隔多远的碎片还能靠'彼此相似'找到
adv = ev_ev − q·(较远那片)  ; adv>0 = 从近的那片出发,兄弟检索优于问题检索
分桶: 同session / 隔1-3场 / 隔4-9场 / 隔10场以上
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
qs = [json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()]
multi = [q for q in qs if len(q.get("evidence_messages") or []) >= 2]
print("multi-ev questions:", len(multi), flush=True)

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
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
qtexts = [q["question"] for q in multi]
bQ = []
for s in range(0, len(qtexts), 64):
    bQ.append(np.asarray(bge.encode(qtexts[s:s + 64])["dense_vecs"], dtype=np.float32))
bQ = l2n(np.concatenate(bQ))

buckets = {"同session": [], "隔1-3场": [], "隔4-9场": [], "隔10场+": []}
for qi, q in enumerate(multi):
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    pairs = []
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                pairs.append((e.get("session_index") or 0, j))
                break
    if len(pairs) < 2:
        continue
    qv = bQ[qi]
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            s1, r1 = pairs[a]
            s2, r2 = pairs[b]
            gap = abs(s1 - s2)
            ev_ev = float(D[r1] @ D[r2])
            q1, q2 = float(qv @ D[r1]), float(qv @ D[r2])
            q_far = min(q1, q2)
            adv = ev_ev - q_far
            if gap == 0:
                buckets["同session"].append((ev_ev, q_far, adv))
            elif gap <= 3:
                buckets["隔1-3场"].append((ev_ev, q_far, adv))
            elif gap <= 9:
                buckets["隔4-9场"].append((ev_ev, q_far, adv))
            else:
                buckets["隔10场+"].append((ev_ev, q_far, adv))

print("%-10s %5s | ev-ev均值 | 问题够远片均值 | 兄弟优势 | 优势>0占比" % ("桶", "对数"), flush=True)
for k, v in buckets.items():
    if not v:
        print(k, "无数据", flush=True)
        continue
    arr = np.array(v)
    print("%-10s %5d |  %.3f   |   %.3f       |  %+.3f  |  %.0f%%" % (
        k, len(v), arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(), 100 * (arr[:, 2] > 0).mean()), flush=True)
print("CHECK_DONE", flush=True)
