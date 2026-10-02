# -*- coding: utf-8 -*-
"""lme_probe3.py — 头部条件验尸(用户直觉): cos排序头部陪衬 vs 金 的方向
随机陪衬0.85是边际统计; LoCoMo引力反向是头部现象——LME头部条件重测"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

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
sess, sid2h = {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    sid2h[sid] = h
    if h not in sess:
        sess[h] = turns
hashes = list(sess.keys())
RAWS, SIDS = [], []
for si, h in enumerate(hashes):
    RAWS.append("[hdr]")
    SIDS.append("lme-s" + h[:12])
    for t in sess[h]:
        role = t.get("role") or "user"
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
        SIDS.append("lme-s" + h[:12])
NR = len(RAWS)
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
V256 = l2n(np.load(OUT + "/rec_vec256.npy").astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
Q256 = l2n(np.load(OUT + "/q_qwen256.npy").astype(np.float32))
P("loaded NR=%d %.0fs" % (NR, time.time() - t0))

def auc(pos, neg):
    allv = sorted(list(pos) + list(neg))
    import bisect as bs
    rp = 0.0
    for v in pos:
        rp += bs.bisect_left(allv, v) + 1
    n1, n2 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2)

rng = np.random.RandomState(0)
# 头部条件: 域内cos排序, 头部陪衬=前30非金; 分层: 全头30 / 紧头10
res = {k: ([], []) for k in ("dqc_h30", "vqc_h30", "dqc_h10", "vqc_h10",
                              "g1_h30", "len_h30", "echo_h30", "dqc_rand")}
for qi, q in enumerate(d):
    gold_sids = set(q.get("answer_session_ids") or [])
    gold_rows = set()
    for s in gold_sids:
        if s in sid2h:
            gold_rows |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    if not gold_rows:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows = np.array(sorted(set(r for k2 in hay_keys for r in SID2ROWS.get(k2, []))))
    if len(hay_rows) < 100:
        continue
    cq = D[hay_rows] @ X[qi] + V256[hay_rows] @ Q256[qi]
    order = np.argsort(-cq)
    head = hay_rows[order[:40]]
    head_noise = [i for i in head if i not in gold_rows]
    gsel = rng.choice(sorted(gold_rows), size=min(6, len(gold_rows)), replace=False).tolist()
    if len(head_noise) < 5:
        continue
    qtext = q["question"]
    qtok = toks(qtext)
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qlen = max(1, len(qtok))
    for i in gsel:
        rt = toks(RAWS[i])
        res["dqc_h30"][0].append(float(D[i] @ X[qi]))
        res["vqc_h30"][0].append(float(V256[i] @ Q256[qi]))
        res["g1_h30"][0].append(len(qstems & set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()))) / max(1, len(qstems)))
        res["len_h30"][0].append(len(RAWS[i].split()) / 40.0)
        res["echo_h30"][0].append(len(qtok & rt) / max(1, len(rt)))
    h10 = head_noise[:10]
    for i in head_noise:
        rt = toks(RAWS[i])
        res["dqc_h30"][1].append(float(D[i] @ X[qi]))
        res["vqc_h30"][1].append(float(V256[i] @ Q256[qi]))
        res["g1_h30"][1].append(len(qstems & set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()))) / max(1, len(qstems)))
        res["len_h30"][1].append(len(RAWS[i].split()) / 40.0)
        res["echo_h30"][1].append(len(qtok & rt) / max(1, len(rt)))
    for i in gsel:
        res["dqc_h10"][0].append(float(D[i] @ X[qi]))
        res["vqc_h10"][0].append(float(V256[i] @ Q256[qi]))
    for i in h10:
        res["dqc_h10"][1].append(float(D[i] @ X[qi]))
        res["vqc_h10"][1].append(float(V256[i] @ Q256[qi]))
    # 随机陪衬对照(复算)
    rnd = rng.choice(hay_rows, size=min(30, len(hay_rows)), replace=False)
    rnd = [i for i in rnd if i not in gold_rows][:20]
    for i in rnd:
        res["dqc_rand"][1].append(float(D[i] @ X[qi]))
    for i in gsel:
        res["dqc_rand"][0].append(float(D[i] @ X[qi]))

P("\n===== 头部条件验尸 =====")
P("随机陪衬(对照):   dqc AUC=%.3f (金=%.3f 噪=%.3f)" % (
    auc(res["dqc_rand"][0], res["dqc_rand"][1]),
    np.mean(res["dqc_rand"][0]), np.mean(res["dqc_rand"][1])))
P("头部30陪衬:       dqc AUC=%.3f (金=%.3f 噪=%.3f)" % (
    auc(res["dqc_h30"][0], res["dqc_h30"][1]),
    np.mean(res["dqc_h30"][0]), np.mean(res["dqc_h30"][1])))
P("紧头10陪衬:       dqc AUC=%.3f (金=%.3f 噪=%.3f)" % (
    auc(res["dqc_h10"][0], res["dqc_h10"][1]),
    np.mean(res["dqc_h10"][0]), np.mean(res["dqc_h10"][1])))
P("头部30: vqc AUC=%.3f  g1 AUC=%.3f  len AUC=%.3f  echo AUC=%.3f" % (
    auc(res["vqc_h30"][0], res["vqc_h30"][1]),
    auc(res["g1_h30"][0], res["g1_h30"][1]),
    auc(res["len_h30"][0], res["len_h30"][1]),
    auc(res["echo_h30"][0], res["echo_h30"][1])))
P("done %.0fs" % (time.time() - t0))
