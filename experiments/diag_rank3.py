# -*- coding: utf-8 -*-
"""diag_rank3.py — 补测: ①同会话含人名竞争池大小 ②320道rank>25错题的证据名次分布
(开到k=50/100/200各能捞回多少)"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
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
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)

def ev_text(e):
    for f in ("text", "message", "message_text"):
        if isinstance(e.get(f), str) and e.get(f).strip():
            return e[f]
    return ""
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

rows = json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
qs_miss = [r for r in rows if r.get("llm_score") != 1]
Q = []
qlist = [r["question"] for r in qs_miss]
for s in range(0, len(qlist), 64):
    Q.append(np.asarray(model.encode(qlist[s:s + 64])["dense_vecs"], dtype=np.float32))
Q = np.concatenate(Q)
Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
S = Q @ D.T

ranks = []
pools = []
dbg = 0
for j, r in enumerate(qs_miss):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    ckey = "loco-" + sid
    cand = CONV.get(ckey, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(ev_text(e))
        if len(t) < 15:
            continue
        for i in cand:
            if KIND[i] == "raw" and t[:40] in normsub(TEXTS[i]):
                hits.append(i)
                break
    if dbg < 3:
        e0 = (r.get("evidence_messages") or [{}])[0]
        print("DBG sid=%s ckey=%s cand=%d ev_text=%r hits=%d nCONV=%d firstkey=%s" % (
            sid, ckey, len(cand), ev_text(e0)[:60], len(hits), len(CONV), next(iter(CONV), "-")))
        dbg += 1
    if not hits:
        continue
    order = np.argsort(-S[j])
    pos = {i: p for p, i in enumerate(order)}
    rk = min(pos[i] for i in hits) + 1
    if rk <= 25:
        continue
    ranks.append(rk)
    names = {str(r.get(f)).lower() for f in ("speaker_a", "speaker_b") if r.get(f)}
    pools.append(sum(1 for i in cand if any(n in TEXTS[i].lower() for n in names)))

ranks = np.array(ranks)
print("rank>25题数:", len(ranks))
print("证据名次: p25=%d p50=%d p75=%d p90=%d max=%d" % tuple(np.percentile(ranks, [25, 50, 75, 90, 100])))
for k in (50, 100, 200, 400):
    print("  窗口开到k=%d 能捞回: %d题 (%.0f%%)" % (k, (ranks <= k).sum(), 100 * (ranks <= k).mean()))
print("竞争池(同会话含人名记录数): p50=%.0f p90=%.0f, 会话记录总数中位=%.0f" % (
    np.median(pools), np.percentile(pools, 90), np.median([len(v) for v in CONV.values()])))
