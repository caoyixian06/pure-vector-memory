# -*- coding: utf-8 -*-
"""foundry9.py — ①256维里有没有"时间维"(逐维Δ+拆半+判别AUC) ②规则先行检索:时间题按"含时间表达"筛池
零API。对照: 1024空间同测。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry9_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

t0 = time.time()
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
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
RAWN = [norm(t) for t in RAW]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
P("库=%d 双空间就绪 %.0fs" % (NR, time.time() - t0))

# ===== 时间规则标签: 文本含月份/星期/年份数字 =====
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
def has_time(text):
    tl = str(text).lower()
    if any(m in tl for m in MONTHS):
        return True
    if any(w in tl for w in WEEK):
        return True
    if re.search(r"\b(19|20)\d\d\b", tl):
        return True
    return False

TLAB = np.array([has_time(t) for t in RAW])
P("含时间表达记录: %d/%d = %.1f%%" % (TLAB.sum(), NR, 100.0 * TLAB.mean()))

# ===== ①逐维时间维检测 =====
def dim_probe(name, MAT):
    pos, neg = MAT[TLAB], MAT[~TLAB]
    if len(pos) < 50 or len(neg) < 50:
        return
    Delta = pos.mean(0) - neg.mean(0)
    top = np.argsort(-np.abs(Delta))[:10]
    P("\n[%s] 逐维时间Δ:" % name)
    P("  |Δ|最大10维: " + ", ".join("d%d=%.4f" % (j, Delta[j]) for j in top))
    P("  |Δ|>0.01维数: %d / %d" % (int((np.abs(Delta) > 0.01).sum()), MAT.shape[1]))
    # 拆半稳定性
    ph = np.array([i % 2 == 0 for i in range(len(pos))])
    nh = np.array([i % 2 == 0 for i in range(len(neg))])
    d1 = pos[ph].mean(0) - neg[nh].mean(0)
    d2 = pos[~ph].mean(0) - neg[~nh].mean(0)
    st = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))
    P("  拆半稳定cos=%.3f" % st)
    # 判别: Δ投影 + 最强单维
    dv = Delta / (np.linalg.norm(Delta) + 1e-9)
    pj = MAT @ dv
    from sklearn.metrics import roc_auc_score
    y = TLAB.astype(int)
    auc_p = roc_auc_score(y, pj)
    bj = int(np.argmax(np.abs(Delta)))
    auc_1 = roc_auc_score(y, MAT[:, bj])
    P("  Δ投影判别AUC=%.3f | 最强单维(d%d)AUC=%.3f" % (auc_p, bj, auc_1))

P("\n===== ①时间维逐维检测 =====")
dim_probe("1024空间", D)
dim_probe("256空间", QW)

# ===== ②规则先行检索(时间题) =====
P("\n===== ②规则先行: 时间题按槽规律筛池 =====")
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
def gold_rec(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for i, m in enumerate(MID):
        if any(k in RAWN[i] for k in keys):
            return i
    return None
def is_time_q(qa):
    ans = " ".join(str(x) for x in (Q[qa].get("answer") or [])).lower()
    return any(m in ans for m in MONTHS) or re.search(r"\b(19|20)\d\d\b", ans) is not None

tq = [qa for qa in IDS if IDX[qa] is not None and is_time_q(qa)]
X = None
# 题目向量: 用xu2缓存? 直接重嵌太重 -> 用FINAL第1名做话题代理不行; 用C0? 无X. 简化: 池内按FINAL排序
P("时间题=%d" % len(tq))
res = {"rule": [0, 0, 0], "final": [0, 0, 0]}
rule_recall = []
for qa in tq:
    g = gold_rec(qa)
    if g is None:
        continue
    # 规则池: 全库含时间表达记录, 池内按FINAL名次排
    rule_pool = [i for i in np.argsort(-FINAL[IDX[qa]]) if TLAB[i]][:50]
    rule_recall.append(100.0 * sum(1 for i in np.argsort(-FINAL[IDX[qa]])[:50] if TLAB[i]) / 50.0)
    for k_i, k in enumerate((5, 10, 30)):
        if g in rule_pool[:k]:
            res["rule"][k_i] += 1
    forder = [i for i in np.argsort(-FINAL[IDX[qa]])[:30]]
    for k_i, k in enumerate((5, 10, 30)):
        if g in forder[:k]:
            res["final"][k_i] += 1
n = len([qa for qa in tq if gold_rec(qa) is not None])
P("FINAL前50里含时间表达比例(规则池可行性): 均值=%.1f%%" % (np.mean(rule_recall) if rule_recall else -1))
P("时间题n=%d" % n)
P("规则先行(池=含时间表达, 池内按FINAL): hit@5=%.1f%% hit@10=%.1f%% hit@30=%.1f%%" % (
    tuple(100.0 * res["rule"][i] / max(1, n) for i in range(3))))
P("FINAL无规则:          hit@5=%.1f%% hit@10=%.1f%% hit@30=%.1f%%" % (
    tuple(100.0 * res["final"][i] / max(1, n) for i in range(3))))
P("FOUNDRY9_DONE %.0fs" % (time.time() - t0))
LOG.close()
