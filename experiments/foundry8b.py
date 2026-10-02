# -*- coding: utf-8 -*-
"""foundry8b.py — 用户机制实测: 向量槽匹配的链式多跳
跳1: 题目向量→R1(余弦top); 跳2查询=R1头部记录向量质心(不转文本,向量直接链);
槽条件: 命中槽轴(place/time/who)的记录才进/加权R2。
对照: FINAL / 纯链式 / 槽条件链式。靶: cat2多跳题金证据hit@k。
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry8b_results.txt"
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
MSET = set(MID)
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
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

_c = np.load(HERE + "/xz_cache.npz")
X, Z = _c["X"], _c["Z"]
_cw = np.load(HERE + "/cue_word_cache.npz", allow_pickle=True)
WV = _cw["V"]
WV = WV / np.maximum(np.linalg.norm(WV, axis=1, keepdims=True), 1e-9)
WORDS = [str(w).lower() for w in _cw["WORDS"]]
W2I = {w: i for i, w in enumerate(WORDS)}
def cent(ws):
    vs = [WV[W2I[w]] for w in ws if w in W2I]
    M = np.stack(vs)
    return M.mean(0)
def axis(pos, neg):
    a = cent(pos) - cent(neg)
    return a / (np.linalg.norm(a) + 1e-9)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NAMES = ["john", "mary", "james", "sarah", "mike", "dave", "caroline", "melanie",
         "nate", "joanna", "deborah", "evan", "sam", "maria", "gina", "calvin",
         "andrew", "audrey", "jolene", "tim"]
PLACES = ["california", "texas", "america", "canada", "germany", "france", "japan",
          "china", "london", "paris", "brazil", "york", "europe", "spain", "italy"]
CONTROL = ["table", "chair", "idea", "water", "music", "book", "phone", "tree",
           "bread", "window", "car", "shirt"]
AX_P = axis(PLACES, CONTROL)
AX_T = axis(MONTHS, CONTROL)
AX_W = axis(NAMES, CONTROL)
slotP = QW @ AX_P
slotT = QW @ AX_T
slotW = QW @ AX_W
P("axes ready %.0fs" % (time.time() - t0))

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
def normtext(s):
    return norm(s)
def gold_rec(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for i, m in enumerate(MID):
        if any(k in normtext(rec_raw(m)) for k in keys):
            return i
    return None

ERR = set()
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m and m.group(2) == "0":
        ERR.add(m.group(1))

cats = {}
for qa, q in Q.items():
    cats.setdefault(q.get("category"), []).append(qa)

K = (5, 10, 30)
def run_cat(CAT):
    qlist = [qa for qa in cats.get(CAT, []) if qa in IDX]
    res = {"final": [0]*3, "chain": [0]*3, "slot_chain": [0]*3, "hybrid": [0]*3}
    n = 0
    nerr = {"final": 0, "chain": 0, "slot_chain": 0, "hybrid": 0}
    for qa in qlist:
        g = gold_rec(qa)
        if g is None:
            continue
        qi = IDX[qa]
        n += 1
        is_err = qa in ERR
        order = list(np.argsort(-C0[qi]))
        top3 = order[:3]
        rstar = l2n(D[top3].mean(0, keepdims=True))[0]
        chain_sc = D @ rstar
        slot_sc = slotP + slotT + slotW
        forder = [i for i in np.argsort(-FINAL[qi])[:30]]
        for k_i, k in enumerate(K):
            if g in forder[:k]:
                res["final"][k_i] += 1
                if is_err:
                    nerr["final"] += 1
        chain_order = [i for i in np.argsort(-chain_sc) if i not in set(forder)]
        chain_cand = forder + chain_order[:30]
        for k_i, k in enumerate(K):
            if g in chain_cand[:k]:
                res["chain"][k_i] += 1
                if is_err:
                    nerr["chain"] += 1
        # 槽条件链式: R2里按槽分加权重排
        qtok = set(norm(w) for w in re.findall(r"[a-zA-Z]{4,}", Q[qa].get("question") or ""))
        best_slot = max((("P", slotP[g] if False else 0), ), key=lambda x: x[1])
        # 题目级槽型: 用题目token的槽轴最大投影
        st = sum(slotT[[MID.index(m) for m in []]]) if False else None
        # 简化: 题目向量对槽轴的投影选槽
        rstar256 = l2n(QW[top3].mean(0, keepdims=True))[0]
        proj = {"P": float(rstar256 @ AX_P), "T": float(rstar256 @ AX_T), "W": float(rstar256 @ AX_W)}
        slotname = max(proj, key=proj.get)
        slotvec = {"P": AX_P, "T": AX_T, "W": AX_W}[slotname]
        r2 = [i for i in np.argsort(-chain_sc) if i not in set(forder)][:200]
        r2_sc = [(float(chain_sc[i]) + 0.5 * float(QW[i] @ slotvec), i) for i in r2]
        r2_sc.sort(reverse=True)
        slot_cand = forder + [i for _, i in r2_sc[:30]]
        for k_i, k in enumerate(K):
            if g in slot_cand[:k]:
                res["slot_chain"][k_i] += 1
                if is_err:
                    nerr["slot_chain"] += 1
        # hybrid: FINAL主序 + 链式新命中最高的插队
        new_hits = [i for i in chain_order[:30] if i not in set(forder)]
        horder = list(forder)
        for pos_i, i in enumerate(new_hits[:10]):
            horder.append(i)
        for k_i, k in enumerate(K):
            if g in horder[:k] and g not in forder[:k]:
                res["hybrid"][k_i] += 1
                if is_err:
                    nerr["hybrid"] += 1
    P("cat%s n=%d" % (CAT, n))
    for nm in ("final", "chain", "slot_chain", "hybrid"):
        a, b, c = res[nm]
        P("  %-11s hit@5=%.1f%% hit@10=%.1f%% hit@30=%.1f%% (其中原错题救回=%d)" % (
            nm, 100.0*a/max(1,n), 100.0*b/max(1,n), 100.0*c/max(1,n), nerr[nm]))

for CAT in ("2", "4", "1", "3"):
    run_cat(CAT)
P("FOUNDRY8B_DONE %.0fs" % (time.time() - t0))
LOG.close()
