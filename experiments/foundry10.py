# -*- coding: utf-8 -*-
"""foundry10.py — 贴近原文查询的检索上限测试(学长的89%是哪种考卷)
查询=金证据的邻句(对话中紧邻的上一句/下一句), 纯cos搜索, 金证据排位。
若邻句查询hit@5≈89% → 库的近转述检索上限=学长的数字, 我们管线在简单卷已最优;
若显著<89% → 纯向量实现有损失可修。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry10_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

t0 = time.time()
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
TWIN = {}
for i, m in enumerate(MID):
    r = REC.get(m, {})
    tw = r.get("raw_of") or r.get("twin_of")
    if tw and tw in set(MID):
        TWIN[i] = MID.index(tw)

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}

def gold_idx(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for i in range(NR):
        nj = norm(rec_raw(MID[i]))
        if any(k in nj for k in keys):
            return i
    return None

ERR = set()
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m and m.group(2) == "0":
        ERR.add(m.group(1))

# 每道题: 金证据g, 邻句g-1/g+1(同会话), 邻句做查询的hit@k
res = {"prev": [0]*4, "next": [0]*4, "either": [0]*4}
K = (1, 5, 10, 30)
n = 0
err_res = {"prev": [0]*4, "next": [0]*4, "either": [0]*4}
nerr = 0
for qa in IDS:
    g = gold_idx(qa)
    if g is None:
        continue
    qi = IDX[qa]
    ck = CONVKEY[g]
    is_err = qa in ERR
    found_any = [False]*4
    for tag, off in (("prev", -1), ("next", +1)):
        a = g + off
        if a < 0 or a >= NR or CONVKEY[a] != ck:
            continue
        sims = D @ D[a]
        sims[a] = -9
        if g in TWIN and TWIN[g] < NR:
            sims[TWIN[g]] = -9
        for twin_i, t in list(TWIN.items()):
            pass
        order = np.argsort(-sims)
        grank = int(np.where(order == g)[0][0]) + 1
        for k_i, k in enumerate(K):
            if grank <= k:
                res[tag][k_i] += 1
                found_any[k_i] = True
    for k_i in range(4):
        if found_any[k_i]:
            res["either"][k_i] += 1
            if is_err:
                err_res["either"][k_i] += 1
    n += 1
    if is_err:
        nerr += 1

P("n=%d (错题%d)" % (n, nerr))
for tag in ("prev", "next", "either"):
    P("邻句查询[%s]: " % tag + "  ".join("hit@%d=%.1f%%" % (k, 100.0 * res[tag][i] / max(1, n)) for i, k in enumerate(K)))
P("错题邻句查询[either]: " + "  ".join("hit@%d=%.1f%%" % (k, 100.0 * err_res["either"][i] / max(1, nerr)) for i, k in enumerate(K)))
P("FOUNDRY10_DONE %.0fs" % (time.time() - t0))
LOG.close()
