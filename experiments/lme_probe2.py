# -*- coding: utf-8 -*-
"""lme_probe2.py — 规律层LME全面验尸: 引力反向/G1/三团/长度/问号/256vs1024判别力"""
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
pos = {k: [] for k in ("dqc", "vqc", "g1", "jac", "qcov", "len", "qmark", "echo")}
neg = {k: [] for k in pos}
gg_j, gn_j, nn_j = [], [], []   # 三团: 256cos (金会话内对/金噪对/噪噪对)
for qi, q in enumerate(d):
    gold_sids = set(q.get("answer_session_ids") or [])
    gold_rows = set()
    for s in gold_sids:
        if s in sid2h:
            gold_rows |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    if not gold_rows:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    hay_rows = np.array(sorted(r for k2 in hay_keys for r in SID2ROWS.get(k2, [])))
    if len(hay_rows) < 50:
        continue
    qtext = q["question"]
    qtok = toks(qtext)
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", qtext.lower()) if w not in QSTOP and len(w) > 2)
    qlen = max(1, len(qtok))
    gold_s = rng.choice(sorted(gold_rows), size=min(8, len(gold_rows)), replace=False).tolist()
    noise_s = rng.choice(hay_rows, size=min(40, len(hay_rows)), replace=False).tolist()
    noise_s = [i for i in noise_s if i not in gold_rows][:30]
    for i in gold_s:
        rt = toks(RAWS[i])
        inter = qtok & rt
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()))
        pos["dqc"].append(float(D[i] @ X[qi]))
        pos["vqc"].append(float(V256[i] @ Q256[qi]))
        pos["g1"].append(len(qstems & rstems) / max(1, len(qstems)))
        pos["jac"].append(len(inter) / max(1, len(qtok | rt)))
        pos["qcov"].append(len(inter) / qlen)
        pos["len"].append(len(RAWS[i].split()) / 40.0)
        pos["qmark"].append(1.0 if RAWS[i].rstrip().endswith("?") else 0.0)
        pos["echo"].append(len(inter) / max(1, len(rt)))
    for i in noise_s:
        rt = toks(RAWS[i])
        inter = qtok & rt
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[i].lower()))
        neg["dqc"].append(float(D[i] @ X[qi]))
        neg["vqc"].append(float(V256[i] @ Q256[qi]))
        neg["g1"].append(len(qstems & rstems) / max(1, len(qstems)))
        neg["jac"].append(len(inter) / max(1, len(qtok | rt)))
        neg["qcov"].append(len(inter) / qlen)
        neg["len"].append(len(RAWS[i].split()) / 40.0)
        neg["qmark"].append(1.0 if RAWS[i].rstrip().endswith("?") else 0.0)
        neg["echo"].append(len(inter) / max(1, len(rt)))
    # 三团(256cos): 金会话内对/金噪对/噪噪对
    if len(gold_s) >= 2:
        for a2 in range(len(gold_s)):
            for b2 in range(a2 + 1, len(gold_s)):
                gg_j.append(float(V256[gold_s[a2]] @ V256[gold_s[b2]]))
    for gi in gold_s[:4]:
        for ni in noise_s[:6]:
            gn_j.append(float(V256[gi] @ V256[ni]))
    for a2 in range(min(8, len(noise_s))):
        for b2 in range(a2 + 1, min(8, len(noise_s))):
            nn_j.append(float(V256[noise_s[a2]] @ V256[noise_s[b2]]))

P("\n===== 规律层LME验尸 (金=%d 噪=%d采样) =====" % (len(pos["dqc"]), len(neg["dqc"])))
P("①引力反向检查:")
P("  1024cos: 金=%.3f 噪=%.3f AUC=%.3f  %s" % (
    np.mean(pos["dqc"]), np.mean(neg["dqc"]), auc(pos["dqc"], neg["dqc"]),
    "(金<噪=反向成立)" if np.mean(pos["dqc"]) < np.mean(neg["dqc"]) else "(正向!)"))
P("  256cos:  金=%.3f 噪=%.3f AUC=%.3f  %s" % (
    np.mean(pos["vqc"]), np.mean(neg["vqc"]), auc(pos["vqc"], neg["vqc"]),
    "(金<噪=反向成立)" if np.mean(pos["vqc"]) < np.mean(neg["vqc"]) else "(正向!)"))
P("②G1词干呼应: 金=%.3f 噪=%.3f AUC=%.3f (LoCoMo=0.781)" % (
    np.mean(pos["g1"]), np.mean(neg["g1"]), auc(pos["g1"], neg["g1"])))
P("③Jaccard: 金=%.3f 噪=%.3f AUC=%.3f" % (
    np.mean(pos["jac"]), np.mean(neg["jac"]), auc(pos["jac"], neg["jac"])))
P("④长度判别器: 金=%.3f 噪=%.3f AUC=%.3f (LoCoMo黑盒: 金更长)" % (
    np.mean(pos["len"]), np.mean(neg["len"]), auc(pos["len"], neg["len"])))
P("⑤问号噪声: 金=%.3f 噪=%.3f (LoCoMo: 金9.6%%噪34.2%%)" % (
    np.mean(pos["qmark"]), np.mean(neg["qmark"])))
P("⑥回声(echo覆盖率): 金=%.3f 噪=%.3f AUC=%.3f" % (
    np.mean(pos["echo"]), np.mean(neg["echo"]), auc(pos["echo"], neg["echo"])))
P("⑦三团(256cos): 金金=%.3f 金噪=%.3f 噪噪=%.3f  (LoCoMo Jaccard版: 0.563/0.077/0.128)" % (
    np.mean(gg_j), np.mean(gn_j), np.mean(nn_j)))
P("⑧256vs1024判别力: AUC(256)=%.3f vs AUC(1024)=%.3f  (LoCoMo黑盒: 256强9倍)" % (
    auc(pos["vqc"], neg["vqc"]), auc(pos["dqc"], neg["dqc"])))
P("done %.0fs" % (time.time() - t0))
