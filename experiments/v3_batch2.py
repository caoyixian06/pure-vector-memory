# -*- coding: utf-8 -*-
"""v3_batch2.py — F3剂量修正 + 自信度路由离线验证
G1 占座惩罚剂量扫描(1.5/0.8/0.5/0.3): 找对题进5不降、均名次不炸的最大剂量
G2 路由模拟: 低自信题(25%)扩池top50→100(伪扩展: RER缓存只到50, 用融合序补位)重排,
   看低自信子集的证据进窗率提升
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
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
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
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
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
def fused_row(i):
    return zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])

def metrics(order_fn, tag="", sel=None):
    m25 = o5 = cnt = 0
    rsum = 0
    ev1 = 0
    sel_keys = keys if sel is None else list(sel)
    for i in sel_keys:
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
    print("  %-26s 错题进25 %3d  对题进5 %3d  ev@1 %.3f  均名次%.0f (n=%d)" % (
        tag, m25, o5, ev1 / max(1, cnt), rsum / max(1, cnt), cnt), flush=True)

print("== G1 占座惩罚剂量扫描 ==", flush=True)
def occupy(i, pen):
    row_f = fused_row(i)
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    order = np.argsort(-newf)
    row = np.zeros(N, dtype=np.float32)
    row[order[:25]] = np.linspace(25, 1, 25)
    qtoks = toks(rows[i]["question"])
    for r0, j in enumerate(order[:10]):
        t = TEXTS[j]
        if t.rstrip().endswith("?") or COUR.search(t):
            row[j] -= pen
    return np.argsort(-row)
metrics(lambda i: occupy(i, 0), "无惩罚(对照)")
for pen in (0.3, 0.5, 0.8):
    metrics(lambda i, p=pen: occupy(i, p), "占座惩罚%.1f" % pen)

print("== G2 自信度路由(低自信扩池) ==", flush=True)
# 自信度: 用G1前的4特征(重算)
def np_entropy(x):
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-6, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())
feats, labs, keylist = [], [], []
for i in keys:
    top = TOP50[i]
    sc = RER[i]
    inside = np.argsort(-sc)
    s = sc[inside]
    f = [s[0], s[0] - s[1], -np_entropy(sc)]
    feats.append(f)
    labs.append(0 if rows[i].get("llm_score") == 1 else 1)
    keylist.append(i)
feats = np.array(feats); labs = np.array(labs)
half = len(feats) // 2
mu1 = feats[:half][labs[:half] == 1].mean(0)
mu0 = feats[:half][labs[:half] == 0].mean(0)
sw = feats[:half][labs[:half] == 1].var(0) + feats[:half][labs[:half] == 0].var(0) + 1e-9
w = (mu1 - mu0) / sw
scores = feats @ w
qs_ = np.percentile(scores, 75)
low_conf = [keylist[k] for k in range(len(keylist)) if scores[k] >= qs_[0] and scores[k] <= qs_[2]]  # 中段
# 低自信 = 错率最高的四分位(高分)
low_conf = [keylist[k] for k in range(len(keylist)) if scores[k] > qs_[2]]
low_set = set(low_conf)
print("低自信题数:", len(low_set), flush=True)
# 路由: 低自信扩池(top50→融合序前100), 其余用标准
def routed(i):
    row_f = fused_row(i)
    if i in low_set:
        top = np.argsort(-row_f)[:100]
        rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
        sc = np.array([rer_map.get(j, -7.0) for j in top])
        inside = np.argsort(-sc)
        newf = row_f.copy()
        newf[top[inside[:50]]] = np.linspace(50, 1, 50)
        newf[top[inside[50:]]] = np.linspace(0.5, 0.1, len(inside) - 50)
        return np.argsort(-newf)[:60]
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return np.argsort(-newf)
metrics(routed, "路由(低自信扩池100)")
metrics(routed, sel=low_set, tag="  仅低自信子集")
metrics(lambda i: occupy(i, 0.5), sel=low_set, tag="  低自信+占座0.5")
