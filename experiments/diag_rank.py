# -*- coding: utf-8 -*-
"""diag_rank.py — 全量533错题(对照849对题)金证据在库内排名解剖:
证据排第几 / 被谁挤掉 / 余弦差多少 → 回答"为什么搜不到"
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
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("bge loaded", flush=True)

HERE = "C:/locomo_refined/memsys"
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)

MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"])
    KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
print("lib", N, D.shape, flush=True)

REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    REC[r.get("memory_id")] = r

def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(r, ensure_ascii=False)

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

# 预归一化库文本
LIBTXT = [None] * N
_cache = {}
def libtxt(i):
    if LIBTXT[i] is None:
        LIBTXT[i] = norm(rec_text(REC.get(MID[i], {})))
    return LIBTXT[i]

def ev_text(e):
    for f in ("message", "text", "message_text", "content"):
        v = e.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return ""

rows = json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
print("rows", len(rows), flush=True)

# 证据字段名确认
_e = None
for r in rows:
    if r.get("evidence_messages"):
        _e = r["evidence_messages"][0]
        break
print("EV_KEYS", list(_e.keys()) if _e else None, flush=True)

def find_ev_indices(row):
    sid = row.get("sample_id") or ("conv-" + str(row.get("conversation_idx")))
    prefix = "loco-" + sid + "_"
    cand = [i for i in range(N) if MID[i].startswith(prefix)]  # 本会话记录
    hits = []
    for e in row.get("evidence_messages") or []:
        t = norm(ev_text(e))
        if len(t) < 15:
            continue
        needle = t[:40]
        for i in cand:
            if KIND[i] == "raw" and needle in libtxt(i):
                hits.append(i)
                break
    return hits

# 问题批量编码
qs_miss = [r for r in rows if r.get("llm_score") != 1]
qs_ok = [r for r in rows if r.get("llm_score") == 1]
allq = [r["question"] for r in qs_miss] + [r["question"] for r in qs_ok]
Q = []
for s in range(0, len(allq), 64):
    enc = model.encode(allq[s:s + 64])
    Q.append(np.asarray(enc["dense_vecs"], dtype=np.float32))
Q = np.concatenate(Q)
Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
S = Q @ D.T  # (nq, N)
print("sims done", S.shape, flush=True)

def rank_of(sim_row, idx_set):
    if not idx_set:
        return None, None  # rank, cos
    order = np.argsort(-sim_row)
    pos = {i: p for p, i in enumerate(order)}
    best = min(idx_set, key=lambda i: pos[i])
    return pos[best] + 1, float(sim_row[best])

def analyze(group, tag, offset):
    buckets = {"no_ev_in_lib": 0, "rank_gt25": 0, "rank_6_25": 0, "rank_le5": 0}
    gaps = []
    for j, r in enumerate(group):
        hits = find_ev_indices(r)
        rk, ec = rank_of(S[j + offset], hits)
        top1 = float(np.max(S[j + offset]))
        if rk is None:
            buckets["no_ev_in_lib"] += 1
            continue
        if rk > 25:
            buckets["rank_gt25"] += 1
            gaps.append((top1 - ec, j, r, ec, top1))
        elif rk > 5:
            buckets["rank_6_25"] += 1
        else:
            buckets["rank_le5"] += 1
    print("===", tag, len(group), buckets, flush=True)
    if gaps:
        g = np.array([x[0] for x in gaps])
        print("rank>25 余弦差距 top1-evidence: mean %.3f median %.3f" % (g.mean(), np.median(g)), flush=True)
        gaps.sort(key=lambda x: -x[0])
        for gap, j, r, ec, top1 in gaps[:3]:
            srow = S[j + offset]
            o = np.argsort(-srow)
            print("--- 例: Q:", r["question"][:90], flush=True)
            print("    gold:", str(r.get("answer"))[:80], "| 证据cos=%.3f 在top25外(top1=%.3f, 差%.3f)" % (ec, top1, gap), flush=True)
            for rank1, k in enumerate(o[:2], start=1):
                rec = REC.get(MID[k], {})
                print("    第%d名[%.3f]:" % (rank1, srow[k]), rec_text(rec)[:110], flush=True)
    return buckets

analyze(qs_miss, "MISS(错题533)", 0)
analyze(qs_ok, "CORRECT(对题849)", len(qs_miss))

# 分cat的错题桶
from collections import defaultdict
catb = defaultdict(lambda: {"no_ev_in_lib": 0, "rank_gt25": 0, "rank_6_25": 0, "rank_le5": 0})
for j, r in enumerate(qs_miss):
    hits = find_ev_indices(r)
    rk, ec = rank_of(S[j], hits)
    c = str(r.get("category"))
    if rk is None:
        catb[c]["no_ev_in_lib"] += 1
    elif rk > 25:
        catb[c]["rank_gt25"] += 1
    elif rk > 5:
        catb[c]["rank_6_25"] += 1
    else:
        catb[c]["rank_le5"] += 1
print("=== MISS by category ===", flush=True)
for c in sorted(catb):
    print("cat", c, dict(catb[c]), flush=True)
