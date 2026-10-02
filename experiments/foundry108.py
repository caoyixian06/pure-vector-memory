# -*- coding: utf-8 -*-
"""foundry108.py — 零训练显式融合(用户令: 只用记忆库和问题, 禁微调):
四个已验证强信号的会话级融合(LME) + 记录级融合(LoCoMo)
融合=无参数等权z(零训练零调参): G1max(0.884)+capmatch(0.856)+长度(0.756)+双cos"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry108_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

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

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)

res = {"cos5": 0, "cos15": 0, "fuse5": 0, "fuse15": 0, "g1only5": 0, "g1only15": 0}
n = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = sorted(set(i for i, s2 in enumerate(SIDS) if s2 in hay_keys))
    if len(domain) < 100:
        continue
    cq = D[domain] @ X[qi] + V256[domain] @ Q256[qi]
    qtext = q["question"]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qcaps = set(re.findall(r"[A-Z][a-z]+", qtext))
    qtok = toks(qtext)
    # 候选会话 = 域内会话(haystack规模适中, 全评)
    sess_list = sorted(hay_keys)
    s_g1, s_cap, s_len, s_cos = [], [], [], []
    for s2 in sess_list:
        rows = SID2ROWS[s2]
        best_g1 = 0.0
        best_cap = 0.0
        best_len = 0.0
        best_cos = -9.0
        # 会话max聚合(已验证形态): 逐记录扫
        cnt = 0
        for i in rows:
            r = RAWS[i]
            if r.startswith("[hdr"):
                continue
            cnt += 1
            rsts = set(stem(w) for w in re.findall(r"[a-z']+", r.lower()))
            g1 = len(qstems & rsts) / max(1, len(qstems))
            if g1 > best_g1:
                best_g1 = g1
            rcaps = set(re.findall(r"[A-Z][a-z]+", r))
            cm = len(qcaps & rcaps) / max(1, len(qcaps)) if qcaps else 0.0
            if cm > best_cap:
                best_cap = cm
            ln = len(r.split())
            if ln > best_len:
                best_len = ln
            c = float(D[i] @ X[qi] + V256[i] @ Q256[qi])
            if c > best_cos:
                best_cos = c
        s_g1.append(best_g1)
        s_cap.append(best_cap)
        s_len.append(best_len)
        s_cos.append(best_cos)
    s_g1, s_cap, s_len, s_cos = map(np.asarray, (s_g1, s_cap, s_len, s_cos))
    fuse = zscore(s_g1) + zscore(s_cap) + zscore(s_len) + zscore(s_cos)
    order_f = [sess_list[i] for i in np.argsort(-fuse)]
    order_c = [sess_list[i] for i in np.argsort(-s_cos)]
    order_g = [sess_list[i] for i in np.argsort(-s_g1)]
    n += 1
    for tag, order in (("cos", order_c), ("fuse", order_f), ("g1only", order_g)):
        for k2 in (5, 15):
            seen = set(order[:k2])
            if gold_h <= seen:
                res[tag + str(k2)] += 1
    if qi % 100 == 0:
        P("  %d %.0fs" % (qi, time.time() - t0))

P("\n===== LME零训练显式融合 (会话级, n=%d) =====" % n)
P("裸cos(会话max):  @5=%.1f%%  @15=%.1f%%" % (100.0 * res["cos5"] / n, 100.0 * res["cos15"] / n))
P("G1max单信号:     @5=%.1f%%  @15=%.1f%%" % (100.0 * res["g1only5"] / n, 100.0 * res["g1only15"] / n))
P("四信号等权z融合: @5=%.1f%%  @15=%.1f%%" % (100.0 * res["fuse5"] / n, 100.0 * res["fuse15"] / n))
P("对照: 序数ranker直迁 79.2/92.8(有训练) | 裸cos记录级 71.0/78.8")
P("F108_DONE %.0fs" % (time.time() - t0))
