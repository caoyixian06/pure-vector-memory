# -*- coding: utf-8 -*-
"""dispersion.py — 小坏蛋"答案分散度"的指纹测试
D1 计算每题dispersion = 证据跨越的session数 / 答案元素数(或证据记录数)
D2 dispersion预测错误: 对题vs错题的dispersion分布, AUC
D3 dispersion高题: 兄弟扩展检索 vs 普通检索的证据入窗率对比(追捕结局)
D4 按dispersion四分位: 各档的错率(剂量-反应曲线)
"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows_all = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))]
rows = [r for r in rows_all if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}

targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits:
        targets[i] = hits
keys = sorted(targets.keys())
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

# D1: dispersion = 证据session跨度 / 证据条数(或答案元素数)
disp = {}
ev_counts = {}
for i in keys:
    q = qmap.get(rows[i]["qa_id"])
    if not q:
        continue
    sess_set = set()
    for e in q.get("evidence_messages") or []:
        s2 = e.get("session_index")
        if s2 is not None:
            sess_set.add(s2)
    n_ev = len(targets[i])
    if n_ev == 0:
        continue
    disp[i] = len(sess_set) / n_ev
    ev_counts[i] = n_ev

# D2: dispersion预测错误
ok = np.array([1 if rows[i].get("llm_score") == 1 else 0 for i in keys])
dp = np.array([disp.get(i, 0) for i in keys])
def auc(x, y):
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    n1 = y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
print("D2 dispersion预测错误AUC=%.3f (对题均值%.2f vs 错题均值%.2f)" % (
    auc(dp, 1 - ok), dp[ok == 1].mean(), dp[ok == 0].mean()), flush=True)

# D4: 剂量-反应
print("D4 dispersion四分位错率:", flush=True)
q1, q2, q3 = np.percentile(dp, [25, 50, 75])
for lo, hi, nm in ((0, q1, "Q1最低"), (q1, q2, "Q2"), (q2, q3, "Q3"), (q3, 9, "Q4最高")):
    m = (dp >= lo) & (dp <= hi if hi != 9 else True) if hi != 9 else (dp >= lo)
    print("  %s(%.2f~%s): 错率%.0f%%" % (nm, lo, "9" if hi == 9 else "%.2f" % hi, 100 * (1 - ok[m].mean())), flush=True)

# D3: 高dispersion题的兄弟扩展 vs 普通(证据入窗率)
print("D3 高dispersion(>Q3)题的检索对比:", flush=True)
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
hi_keys = [i for i in keys if disp.get(i, 0) > q3]
win_bro = win_norm = 0
n_hi = 0
for i in hi_keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    allh = hs | tws
    top = TOP50[i][np.argsort(-RER[i])]
    # 普通: top25
    in_norm = len(allh & set(top[:25].tolist())) >= len(allh)
    # 兄弟扩展: top6种子→邻句+cos0.7簇, 预算40
    used, star = set(), []
    for s0 in top[:6]:
        if s0 in used:
            continue
        grp = [s0]
        if s0 + 1 < N and MID[s0 + 1].rsplit("_", 1)[0] == MID[s0].rsplit("_", 1)[0]:
            grp.append(s0 + 1)
        cl = [j for j in TOP50[i] if j not in used and j not in grp and float(D[s0] @ D[j]) >= 0.7][:8]
        grp += cl
        used |= set(grp)
        star += grp
    star = star[:40]
    in_bro = len(allh & set(star)) >= len(allh)
    n_hi += 1
    win_bro += in_bro
    win_norm += in_norm
print("  高dispersion题%d道: 全集入窗率 兄弟扩展=%.0f%% vs 普通top25=%.0f%%" % (
    n_hi, 100 * win_bro / max(1, n_hi), 100 * win_norm / max(1, n_hi)), flush=True)
print("DISPERSION_DONE", flush=True)
