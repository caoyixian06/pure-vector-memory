# -*- coding: utf-8 -*-
"""diag_topsep.py — top列表里 证据 vs 噪声 的向量可分性
标签: EVID(top50中的金证据) / TWIN_EV(证据的孪生摘要) / NOISE(其余)
特征(全零LLM可计算):
 f1 精排分  f2 BGE-u2 cos  f3 qwen cos  f4 词票  f5 与top1相似
 f6 top10簇支持度  f7 草稿答案cos(answer-conditioned, 臂B预测)  f8 问句形/寒暄/日期/长度
输出: 各特征两类均值+d'(分离度)+AUC, 无拟合
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
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
qid2i = {r["qa_id"]: i for i, r in enumerate(rows)}
preds = {}
for l in open(HERE + "/submission_r33b.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        preds[d["qa_id"]] = d.get("predicted_answer") or ""
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

bQ = emb(Q)
ans_texts = [preds.get(r["qa_id"], "") or "unknown" for r in rows]
bA = emb(ans_texts)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
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
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        targets[i] = hits
print("matched:", len(targets), flush=True)

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQu = l2n(bQ + u2)

COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
DATEP = re.compile(r"\b(20\d\d|January|February|March|April|May|June|July|August|September|October|November|December|yesterday|tomorrow|last week|next week|month)\b", re.I)

feats = {k: [] for k in ("f1精排", "f2_bge_u2", "f3_qwen", "f4词票", "f5近top1", "f6簇支持", "f7草稿cos", "f8问句形", "f8寒暄", "f8含日期")}
labels = []
for i in range(n):
    if i not in targets:
        continue
    hits = set(targets[i])
    tws = {TWIN.get(h) for h in hits} - {None}
    top = TOP50[i]
    order_in = np.argsort(-RER[i])  # 精排序
    SBSrow = D @ bQu[i]
    top10 = top[order_in[:10]]
    for slot, j in enumerate(top):
        lab = "EVID" if j in hits else ("TWIN_EV" if j in tws else "NOISE")
        t = TEXTS[j]
        feats["f1精排"].append(float(RER[i][list(top).index(j)]))
        feats["f2_bge_u2"].append(float(SBSrow[j]))
        feats["f3_qwen"].append(float(SQc[i][j]))
        feats["f4词票"].append(float(VEXc[i][j]))
        feats["f5近top1"].append(float(D[j] @ D[top[order_in[0]]]))
        feats["f6簇支持"].append(float(np.mean(D[top10] @ D[j])))
        feats["f7草稿cos"].append(float(D[j] @ bA[i]))
        feats["f8问句形"].append(1.0 if t.rstrip().endswith("?") else 0.0)
        feats["f8寒暄"].append(1.0 if COUR.search(t) else 0.0)
        feats["f8含日期"].append(1.0 if DATEP.search(t) else 0.0)
        labels.append(lab)

labels = np.array(labels)
ev = labels == "EVID"
no = labels == "NOISE"
tw = labels == "TWIN_EV"
print("top50内记录: EVID=%d TWIN_EV=%d NOISE=%d" % (ev.sum(), tw.sum(), no.sum()), flush=True)

def auc(x, pos, neg):
    x = np.asarray(x, dtype=np.float64)
    r = np.argsort(np.concatenate([x[pos], x[neg]]))
    ranks = np.empty_like(r, dtype=np.float64)
    ranks[r] = np.arange(1, len(r) + 1)
    rp = ranks[:pos.sum()]
    return (rp.sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())

print("%-12s | EVID均值 | NOISE均值 | d'   | AUC" % "特征")
for k, v in feats.items():
    v = np.asarray(v, dtype=np.float64)
    mu_e, mu_n = v[ev].mean(), v[no].mean()
    sd = np.sqrt((v[ev].var() + v[no].var()) / 2) + 1e-9
    d = (mu_e - mu_n) / sd
    a = auc(v, ev, no)
    print("%-12s | %8.3f | %8.3f | %+.2f | %.3f" % (k, mu_e, mu_n, d, a), flush=True)
