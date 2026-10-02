# -*- coding: utf-8 -*-
"""foundry8.py — 多跳关系行走验证(用户机制: 词宿主归属+逐跳存在性检查)
跳1: 问题词→词宿主→R1; 拔桥: R1文本中高IDF且不在问题的词; 查存: 桥词有宿主才续;
跳2: 桥词→R2。靶: cat2多跳题, 金证据hit@k vs 现系统FINAL。
"""
import io, json, os, re, sys, time, hashlib
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry8_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

t0 = time.time()
def nkey(w):
    return re.sub(r"[^a-z]", "", w.lower())

# ===== 词宿主表(规范化键) =====
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
HOSTS = defaultdict(set)
for w, hosts in WH.items():
    k = nkey(w)
    if len(k) >= 4:
        HOSTS[k] |= set(hosts)
IDF = {k: np.log(1 + 11753 / max(1, len(h))) for k, h in HOSTS.items()}
P("词宿主键=%d %.0fs" % (len(HOSTS), time.time() - t0))

# ===== 库文本 =====
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
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
RAWN = {}
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
for m in MID:
    RAWN[m] = norm(rec_raw(m))
MSET = set(MID)
P("库=%d %.0fs" % (len(MID), time.time() - t0))

# ===== 金证据 =====
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
def gold_rec(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for m in MID:
        if any(k in RAWN[m] for k in keys):
            return m
    return None

# ===== 错题集 =====
ERR = set()
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m and m.group(2) == "0":
        ERR.add(m.group(1))

z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]

STOP = set("what where when who which how why the a an is are was were did do does of in on at to for and or with about from that this".split())

def walk(qa, max_hop2=200):
    q = Q[qa]
    qtok = [nkey(w) for w in re.findall(r"[a-zA-Z]{4,}", q.get("question") or "")]
    qtok = [w for w in qtok if w not in STOP]
    seeds = [w for w in dict.fromkeys(qtok) if w in HOSTS][:6]
    R1 = defaultdict(float)
    for w in seeds:
        for m in HOSTS[w]:
            if m in MSET:
                R1[m] += IDF.get(w, 5.0)
    # 拔桥: R1文本中的高IDF词, 不在问题里
    bridges = defaultdict(float)
    r1_top = sorted(R1, key=R1.get, reverse=True)[:max_hop2]
    for m in r1_top:
        for w in set(nkey(x) for x in re.findall(r"[a-zA-Z]{4,}", rec_raw(m))):
            if w in qtok or w not in HOSTS or w in seeds:
                continue
            bridges[w] += IDF.get(w, 5.0)
    br_top = sorted(bridges, key=bridges.get, reverse=True)[:12]
    R2 = defaultdict(float)
    for w in br_top:
        for m in HOSTS[w]:
            if m in MSET:
                R2[m] += bridges[w] * 0.5
    return seeds, R1, R2, br_top

cats = {}
for qa, q in Q.items():
    cats.setdefault(q.get("category"), []).append(qa)
P("cat分布: " + str({k: len(v) for k, v in sorted(cats.items())}))

for CAT in ("2",):
    qlist = [qa for qa in cats.get(CAT, []) if qa in IDX]
    P("\n===== cat%s 多跳题 n=%d =====" % (CAT, len(qlist)))
    res = {"walk": [0, 0, 0], "final": [0, 0, 0], "hybrid": [0, 0, 0]}
    nerr = 0
    for qa in qlist:
        g = gold_rec(qa)
        if g is None:
            continue
        seeds, R1, R2, br_top = walk(qa)
        # walk打分: R1原分 + R2(经由桥的2跳分)
        walk_sc = defaultdict(float)
        for m, s in R1.items():
            walk_sc[m] += s
        for m, s in R2.items():
            walk_sc[m] += s
        worder = sorted(walk_sc, key=walk_sc.get, reverse=True)
        for k_i, k in enumerate((5, 10, 30)):
            if g in worder[:k]:
                res["walk"][k_i] += 1
        forder = [MID[i] for i in np.argsort(-FINAL[IDX[qa]])[:30]]
        for k_i, k in enumerate((5, 10, 30)):
            if g in forder[:k]:
                res["final"][k_i] += 1
        forder_full = [MID[i] for i in np.argsort(-FINAL[IDX[qa]])]
        bonus = {m: 1.0 for m in worder[:15]}
        hscore = []
        frank = {m: r for r, m in enumerate(forder_full)}
        for m in set(forder_full[:30]) | set(worder[:15]):
            hscore.append((frank.get(m, 60) - 15.0 * bonus.get(m, 0), m))
        hscore.sort()
        for k_i, k in enumerate((5, 10, 30)):
            if g in [m for _, m in hscore[:k]]:
                res["hybrid"][k_i] += 1
        if qa in ERR:
            nerr += 1
    n = len([qa for qa in qlist if gold_rec(qa) is not None])
    P("可定位=%d 其中错题=%d" % (n, nerr))
    for nm in ("walk", "final", "hybrid"):
        a, b, c = res[nm]
        P("  %-7s hit@5=%.1f%% hit@10=%.1f%% hit@30=%.1f%%" % (
            nm, 100.0 * a / max(1, n), 100.0 * b / max(1, n), 100.0 * c / max(1, n)))
P("\nFOUNDRY8_DONE %.0fs" % (time.time() - t0))
LOG.close()
