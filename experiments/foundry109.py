# -*- coding: utf-8 -*-
"""foundry109.py — 盲调组合搜索(零训练零标签, 总分对错选优):
聚合{max,top2mean,top3mean,mean} × 通道{1024,256,sum,max2} = 16组合"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry109_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
P("loaded %.0fs" % (time.time() - t0))

AGGS = {"max": lambda v: float(np.max(v)),
        "top2mean": lambda v: float(np.sort(v)[-2:].mean()),
        "top3mean": lambda v: float(np.sort(v)[-3:].mean()),
        "mean": lambda v: float(np.mean(v))}
CHS = {"1024": None, "256": None, "sum": None, "max2": None}
res = {(a, c): [0, 0] for a in AGGS for c in CHS}
n = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    if not hay_keys:
        continue
    n += 1
    # 每会话的记录cos集合(排除hdr)
    sess_c1024, sess_c256 = {}, {}
    for s2 in hay_keys:
        c1, c2 = [], []
        for i in SID2ROWS[s2]:
            if RAWS[i].startswith("[hdr"):
                continue
            c1.append(float(D[i] @ X[qi]))
            c2.append(float(V256[i] @ Q256[qi]))
        sess_c1024[s2] = c1
        sess_c256[s2] = c2
    sess_list = sorted(hay_keys)
    for a_name, agg in AGGS.items():
        for c_name in CHS:
            if c_name == "1024":
                sc = {s2: agg(sess_c1024[s2]) for s2 in sess_list if sess_c1024[s2]}
            elif c_name == "256":
                sc = {s2: agg(sess_c256[s2]) for s2 in sess_list if sess_c256[s2]}
            elif c_name == "sum":
                sc = {s2: agg(sess_c1024[s2]) + agg(sess_c256[s2]) for s2 in sess_list if sess_c1024[s2] and sess_c256[s2]}
            else:
                sc = {s2: max(agg(sess_c1024[s2]), agg(sess_c256[s2])) for s2 in sess_list if sess_c1024[s2] and sess_c256[s2]}
            order = sorted(sc, key=lambda s2: -sc[s2])
            for k2 in (5, 15):
                if gold_h <= set(order[:k2]):
                    res[(a_name, c_name)][0 if k2 == 5 else 1] += 1
    if qi % 100 == 0:
        P("  %d %.0fs" % (qi, time.time() - t0))

P("\n===== 组合搜索(n=%d) =====" % n)
rows = sorted(res.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
P("聚合×通道      @5     @15")
for (a, c), (h5, h15) in rows:
    P("%-6s×%-5s %5.1f%%  %5.1f%%" % (a, c, 100.0 * h5 / n, 100.0 * h15 / n))
P("F109_DONE %.0fs" % (time.time() - t0))
