# -*- coding: utf-8 -*-
"""foundry121.py — 形态信号驱动的检索策略(盲调总分):
臂a固定记录cos | 臂b锐度路由(低锐度→会话max锁会话) | 臂c时序最新规则(高时间投影→最新优先)"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry121_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
TIME_WORDS = re.compile("yesterday|tomorrow|today|tonight|last week|next week|last month|next month|last year|next year|this weekend|weeks ago|days ago|months ago|January|February|March|April|June|July|August|September|October|November|December", re.I)
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
ORDER = []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
    ORDER.append(r.get("order") or 0)
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
ORDER_ARR = np.array(ORDER)
is_dated = np.array([bool(TIME_WORDS.search(r)) for r in RAWS])
delta_t = D[is_dated].mean(0) - D[~is_dated].mean(0)
delta_t = delta_t / max(np.linalg.norm(delta_t), 1e-9)
P("loaded %.0fs" % (time.time() - t0))

# 预计算每题: 域内/锐度/时间投影(零标签)
PRE = {}
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    dd = D[domain] @ X[qi]
    sharp = float(np.sort(dd)[-1] - np.sort(dd)[-50:].mean())
    tproj = float(X[qi] @ delta_t)
    PRE[qi] = (domain, dd, sharp, tproj)
P("precompute %.0fs" % (time.time() - t0))

# 全局锐度/时间的分位(零标签自适应阈值)
sharp_all = np.array([PRE[qi][2] for qi in PRE])
tproj_all = np.array([PRE[qi][3] for qi in PRE])
SH_T = float(np.quantile(sharp_all, 0.5))   # 锐度中位: 上半=锐 下半=钝
TP_T = float(np.quantile(tproj_all, 0.75))  # 时间投影top25%=时序题

res = {k: {1: 0, 3: 0, 5: 0} for k in ("a", "b", "c")}
nt = 0
for qi in sorted(PRE):
    ans_raw = d[qi].get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    domain, dd, sharp, tproj = PRE[qi]
    nt += 1
    def hitk(order_list, k2):
        return any(len(ans_stems & stems_of(RAWS[i])) / len(ans_stems) >= 0.5 for i in order_list[:k2])
    # a: 记录cos
    oa = domain[np.argsort(-dd)].tolist()
    # b: 锐度路由 — 钝题(锐度<中位): 会话max锁前3会话, 会话内按cos
    if sharp < SH_T:
        dsids = np.array([SIDS[i] for i in domain])
        umap = {s: j for j, s in enumerate(sorted(set(dsids.tolist())))}
        uidx = np.array([umap[s] for s in dsids.tolist()])
        smax = np.full(len(umap), -9.0)
        np.maximum.at(smax, uidx, dd)
        uarr = np.array(sorted(umap))
        top3s = set(uarr[np.argsort(-smax)[:3]].tolist())
        in3 = np.array([SIDS[i] in top3s for i in domain])
        sub = domain[in3]
        ob = sub[np.argsort(-(D[sub] @ X[qi]))].tolist() + domain[~in3][np.argsort(-dd[~in3])].tolist()
    else:
        ob = oa
    # c: 时序最新规则 — 高时间投影题: top15按order(记录时间序)重排,最新优先
    if tproj > TP_T:
        top15 = domain[np.argsort(-dd)[:15]]
        oc = top15[np.argsort(-ORDER_ARR[top15])].tolist() + domain[np.argsort(-dd)][15:].tolist()
    else:
        oc = oa
    for k2 in (1, 3, 5):
        if hitk(oa, k2):
            res["a"][k2] += 1
        if hitk(ob, k2):
            res["b"][k2] += 1
        if hitk(oc, k2):
            res["c"][k2] += 1

P("\n===== 形态策略三分(n=%d, 锐度中位路由/时间top25%%规则) =====" % nt)
for k, nm in (("a", "记录cos基线"), ("b", "锐度路由(钝→会话锁)"), ("c", "时序最新规则")):
    P("%s: top1=%.1f%%  top3=%.1f%%  top5=%.1f%%" % (
        nm, 100.0 * res[k][1] / nt, 100.0 * res[k][3] / nt, 100.0 * res[k][5] / nt))
P("F121_DONE %.0fs" % (time.time() - t0))
