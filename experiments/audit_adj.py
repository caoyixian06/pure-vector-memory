# -*- coding: utf-8 -*-
"""audit_adj.py — 143例的邻接分析: 证据的前1/2条(框架问句?)的名次与进窗情况"""
import io, json, os, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

MID, KIND = [], []
for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
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
RAWFULL = [rec_raw(m) for m in MID]
NORM = [norm(x) for x in RAWFULL]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
N = len(MID)

z = np.load(HERE + "/r40bf_ckpt.npz")
FINAL = z["FINAL"]
IDS = [str(x) for x in z["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}

dump = [json.loads(l) for l in io.open(HERE + "/why135.jsonl", encoding="utf-8")]
QSMap = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    QSMap[q["qa_id"]] = q

def ev_keys(q):
    return [norm(em.get("text") or "")[:60] for em in (q.get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]

res = {"n": 0, "prev_is_q": 0, "prev_in_win": 0, "prev_rank_p50": None,
       "prev2_is_q": 0, "frame_in_win_any": 0, "frame_top5": 0, "adj_would_fix": 0}
prev_ranks = []
for r in dump:
    qa = r["qa_id"]
    q = QSMap[qa]
    keys = ev_keys(q)
    if not keys:
        continue
    i = IDX[qa]
    order = list(map(int, np.argsort(-FINAL[i])))
    rank = {j: rr + 1 for rr, j in enumerate(order)}
    # 定位证据记录(与why135同一匹配)
    ev_recs = []
    for j in range(N):
        if not NORM[j]:
            continue
        for k in keys:
            if k in NORM[j]:
                ev_recs.append(j)
                break
    if not ev_recs:
        continue
    j0 = ev_recs[0]
    res["n"] += 1
    # 前1条/前2条(同会话)
    frame_ranks = []
    for off in (1, 2):
        jj = j0 - off
        if jj < 0 or jj >= N or CONVKEY[jj] != CONVKEY[j0]:
            continue
        is_q = RAWFULL[jj].rstrip().endswith("?")
        rr = rank.get(jj)
        if off == 1 and is_q:
            res["prev_is_q"] += 1
        if off == 2 and is_q:
            res["prev2_is_q"] += 1
        if is_q and rr:
            frame_ranks.append(rr)
    if frame_ranks:
        best = min(frame_ranks)
        prev_ranks.append(best)
        if best <= 15:
            res["frame_in_win_any"] += 1
        if best <= 5:
            res["frame_top5"] += 1
        # 框架问句在窗内且其 Evidence 在其后2条内 → 邻域扩展可救
        if best <= 9 and any((j0 - ev_recs[0]) <= 2 or True for _ in [0]):
            if any(abs(j0 - jj2) <= 2 and CONVKEY[jj2] == CONVKEY[j0] for jj2 in ev_recs):
                res["adj_would_fix"] += 1
prev_ranks.sort()
if prev_ranks:
    res["prev_rank_p50"] = prev_ranks[len(prev_ranks) // 2]
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
print("ADJ_AUDIT_DONE", flush=True)
