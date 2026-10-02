# -*- coding: utf-8 -*-
"""v3_batch1.py — 第一批四件的离线代理验证(零GLM)
F1 raw先验: 召回段(全池)给raw记录+0.05 → 融合top50重建 → 精排 → 证据进窗率
F2 首提+会话头: 融合分 +0.08*is_first +0.05*(1-sesspos) → 同管线
F3 占座者惩罚: 精排后, 对"问句形 或 寒暄词表命中 且 词面重叠高"的top10记录 -1.5
F4 自信度: 子代理的8特征logit复现, 报错率四分位
对照组: r37配置(融合+精排, 无新件)
指标: 错题证据进25(含孪生)/对题证据进5/均名次 + ev@1
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
# 对齐: 子代理按SB行=有证据题匹配, 这里直接用 fusion_cache 行序=out_dense_all 过滤有q/a/ev的行序
rows = [r for r in rows_all if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
n = len(rows)

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
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
RAWFLAG = np.array([1.0 if KIND[i] == "raw" else 0.0 for i in range(N)], dtype=np.float32)
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
# 首提标记: 会话内专名首次出现
FIRST = np.zeros(N, dtype=np.float32)
SESSPOS = np.zeros(N, dtype=np.float32)
seen_names = {}
for sess, ids in CONV.items():
    ids_sorted = sorted(ids, key=lambda j: MID[j])
    for pos, j in enumerate(ids_sorted):
        if KIND[j] != "raw":
            continue
        t = TEXTS[j]
        sp = t.split(":")[0] if ":" in t[:30] else ""
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", t)
        newf = 0
        for nm in names:
            if nm not in seen_names.get(sess, set()):
                seen_names.setdefault(sess, set()).add(nm)
                newf = 1
        FIRST[j] = newf
        SESSPOS[j] = pos / max(1, len(ids_sorted))
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

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
print("matched:", len(targets), flush=True)

zc = np.load(HERE + "/fusion_cache.npz")
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
SQc, VEXc = zc["SQ"], zc["VEX"]
bQ = emb([r["question"] for r in rows])
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def np_entropy(x):
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-6, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def fused_row(i, w_raw=0.0, w_first=0.0, w_pos=0.0):
    row = zs(SBS[i]) + zs(SQc[i]) + 0.5 * zs(VEXc[i])
    if w_raw:
        row = row + w_raw * RAWFLAG
    if w_first:
        row = row + w_first * FIRST + w_pos * (1.0 - SESSPOS)
    return row

def ev_metrics(order_fn, tag):
    m25 = o5 = cnt = 0
    ev1 = 0
    rsum = 0
    for i in keys:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        allh = list(hs) + list(tws)
        if not all(h2 in pos for h2 in allh):
            continue
        rk = min(pos[h2] for h2 in allh) + 1
        rsum += rk; cnt += 1
        if order[0] in hs or order[0] in tws:
            ev1 += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-28s 错题进25 %3d  对题进5 %3d  ev@1 %.3f  均名次%.0f" % (
        tag, m25, o5, ev1 / max(1, cnt), rsum / cnt), flush=True)

def pipeline(i, row_fused):
    top = np.argsort(-row_fused)[:50]
    # RER只对原TOP50有缓存; 对新top50近似: 用缓存的RER映射, 不在缓存的记录给RER分位[-1]底分
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_fused.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return np.argsort(-newf), top, sc

print("== 基线 r37配置(无新件) ==", flush=True)
ev_metrics(lambda i: pipeline(i, fused_row(i))[0], "融合+精排(=r37排序)")

print("== F1 raw先验(召回段+0.05) ==", flush=True)
ev_metrics(lambda i: pipeline(i, fused_row(i, w_raw=0.05))[0], "+raw先验0.05")
ev_metrics(lambda i: pipeline(i, fused_row(i, w_raw=0.10))[0], "+raw先验0.10")

print("== F2 首提+会话头 ==", flush=True)
ev_metrics(lambda i: pipeline(i, fused_row(i, w_first=0.08, w_pos=0.05))[0], "+首提0.08+会话头0.05")

print("== F1+F2 合并 ==", flush=True)
ev_metrics(lambda i: pipeline(i, fused_row(i, w_raw=0.05, w_first=0.08, w_pos=0.05))[0], "+raw0.05+首提+会话头")

print("== F3 占座者惩罚(精排后处理) ==", flush=True)
def occupy_pen(i):
    order, top, sc = pipeline(i, fused_row(i))
    qtoks = toks(rows[i]["question"])
    row = np.zeros(N, dtype=np.float32)
    row[order[:25]] = np.linspace(25, 1, 25)
    for r0, j in enumerate(order[:10]):
        t = TEXTS[j]
        pen = 0.0
        if t.rstrip().endswith("?") or COUR.search(t):
            pen = 1.5
        row[j] -= pen
    return np.argsort(-row)
ev_metrics(occupy_pen, "精排+占座者惩罚")

print("== F4 自信度特征(题级风险) ==", flush=True)
feats = []
labs = []
for i in keys:
    top = TOP50[i]
    sc = RER[i]
    inside = np.argsort(-sc)
    s = sc[inside]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    f = [s[0], s[0] - s[1], -float(np_entropy(sc)), SBS[i][top[inside[0]]]]
    feats.append(f)
    labs.append(0 if rows[i].get("llm_score") == 1 else 1)
feats = np.array(feats); labs = np.array(labs)
def np_entropy(x):
    p = np.clip(x, 1e-6, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())
# 简易logit(直接线性判别, 拆半)
half = len(feats) // 2
mu1, mu0 = feats[:half][labs[:half] == 1].mean(0), feats[:half][labs[:half] == 0].mean(0)
sw = feats[:half][labs[:half] == 1].var(0) + feats[:half][labs[:half] == 0].var(0) + 1e-9
w = (mu1 - mu0) / sw
sB = feats[half:] @ w
lB = labs[half:]
r = np.argsort(sB)
ranks = np.empty(len(sB), dtype=np.float64); ranks[r] = np.arange(1, len(sB) + 1)
np_ = lB.sum()
a = (ranks[lB].sum() - np_ * (np_ + 1) / 2) / (np_ * (~lB).sum())
qs_ = np.percentile(sB, [25, 50, 75])
err_q = [lB[sB <= qs_[0]].mean(), lB[(sB > qs_[0]) & (sB <= qs_[1])].mean(),
         lB[(sB > qs_[1]) & (sB <= qs_[2])].mean(), lB[sB > qs_[2]].mean()]
print("  自信度AUC(预测答错)=%.3f" % a, flush=True)
print("  低→高错率: %.0f%% → %.0f%% → %.0f%% → %.0f%%" % tuple(100 * x for x in err_q), flush=True)
print("BATCH1_DONE", flush=True)
