# -*- coding: utf-8 -*-
"""raw_nums.py — 直接看数字: 答案/证据/噪声三组数组的逐位规律 + top排名榜的数字规律
R1 逐位对比: 答案句/证据句/噪声句 三组在每维上的原始数值(不是差, 是数值本身)
R2 数字的分布形状: 每组向量的正维个数/负维个数/最大值/最小值/均值/中位
R3 top排名榜规律: 第1~50名记录的数值特征随名次的单调性(找出'名次的数字签名')
R4 证据和噪声最像的维 vs 最不像的维: 在最像的维上两类的数值是否同步波动(共模)
R5 答案↔证据 vs 噪声↔证据: 答案句到底和证据句差在哪几维
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

# 收集三组(答案句=金答案文本的嵌入, 证据句=库内证据记录, 噪声=同池非证据)
ans_rows = [r for r in rows if r.get("question") and r.get("answer")]
A_vec = emb([str(r["answer"][0]) for r in rows])
ev_ids, no_ids = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev_ids += [j for j in top if j in hs or j in tws]
    no_ids += [j for j in top if j not in hs and j not in tws][:6]
ev_ids, no_ids = np.array(ev_ids), np.array(no_ids)
E = D[ev_ids]
NO = D[no_ids]
print("组大小: 答案句%d 证据%d 噪声%d" % (len(A_vec), len(E), len(NO)), flush=True)

print("== R2 数字的分布形状(原始数值,未差分) ==", flush=True)
for name, M in (("答案句", A_vec), ("证据句", E), ("噪声句", NO)):
    posr = (M > 0).mean(0)
    print("  %s: 全库均值%.4f 均值中位%.4f | 正维占比中位%.1f%% | 单点L2范数(归一前)≈1" % (
        name, M.mean(), float(np.median(M.mean(0))), 100 * np.median(posr)), flush=True)
# 每维均值分布的三组对比(哪一段维度的数值系统性地高/低)
me, mn, ma = E.mean(0), NO.mean(0), A_vec.mean(0)
top_d = np.argsort(-np.abs(me - mn))[:15]
print("  证据vs噪声差最大15维:", top_d.tolist(), flush=True)
print("  该15维上 证据均值:", np.round(me[top_d], 4).tolist(), flush=True)
print("           噪声均值:", np.round(mn[top_d], 4).tolist(), flush=True)
print("           答案均值:", np.round(ma[top_d], 4).tolist(), flush=True)
d2 = np.argsort(-np.abs(ma - me))[:15]
print("  答案vs证据差最大15维:", d2.tolist(), flush=True)
print("           答案均值:", np.round(ma[d2], 4).tolist(), flush=True)
print("           证据均值:", np.round(me[d2], 4).tolist(), flush=True)

print("== R3 top排名榜的数字签名 ==", flush=True)
# 每个名次(1..50)上记录的统计特征均值: 与问题cos(用r36融合分近似=名次), 与top1相似, RER分, 证据率
from collections import defaultdict
stat = defaultdict(lambda: defaultdict(list))
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    srow = z1["RER"][i]
    order = top[np.argsort(-srow)]
    t1 = order[0]
    for rank, j in enumerate(order, start=1):
        isev = 1 if (j in hs or j in tws) else 0
        stat[rank]["ev"].append(isev)
        stat[rank]["t1"].append(float(D[j] @ D[t1]))
        stat[rank]["dim"].append(float(np.mean(D[j])))
print("rank | 证据率  | 与第1名cos | 数值均值")
for rk in (1, 2, 3, 5, 8, 12, 18, 25, 35, 50):
    if rk in stat:
        s = stat[rk]
        print("%4d | %.3f  | %.4f    | %.5f" % (
            rk, np.mean(s["ev"]), np.mean(s["t1"]), np.mean(s["dim"])), flush=True)

print("== R4 共模检验: 证据与噪声最像的维上, 数值是否同步 ==", flush=True)
diff_dim = me - mn
corr_all = np.corrcoef(me, mn)[0, 1]
print("  证据均值与噪声均值的跨维相关=%.3f (≈1=完全共模)" % corr_all, flush=True)
# 三组两两
print("  证据vs答案 跨维相关=%.3f | 噪声vs答案=%.3f" % (
    np.corrcoef(me, ma)[0, 1], np.corrcoef(mn, ma)[0, 1]), flush=True)
# 去掉公共成分(减噪声均值)后, 证据剩余的'独有数值'集中在哪
resid = me - mn
print("  剩余独有数值: 正值维数%d 负值维数%d | 最大正维%d(值%.4f) 最大负维%d(值%.4f)" % (
    (resid > 0).sum(), (resid < 0).sum(), int(np.argmax(resid)), resid.max(),
    int(np.argmin(resid)), resid.min()), flush=True)

print("== R5 答案句相对证据句的数值偏移 ==", flush=True)
off = ma - me
print("  偏移均值%.4f 标准差%.4f | 正偏维%d 负偏维%d" % (
    off.mean(), off.std(), (off > 0).sum(), (off < 0).sum()), flush=True)
big = np.argsort(-np.abs(off))[:10]
print("  最大偏移维:", big.tolist(), flush=True)
print("  答案值:", np.round(ma[big], 4).tolist(), flush=True)
print("  证据值:", np.round(me[big], 4).tolist(), flush=True)
# 答案是否就是'证据-寒暄成分+问题成分'的线性组合(解回归)
X = np.stack([me, mn, qc]).T  # (1024, 3)
yv = ma
coef, res_, *_ = np.linalg.lstsq(X, yv, rcond=None)
pred = X @ coef
r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
print("  回归: 答案 ≈ %.2f×证据 + %.2f×噪声 + %.2f×问题质心  (R²=%.3f)" % (
    coef[0], coef[1], coef[2], r2), flush=True)
print("RAWNUMS_DONE", flush=True)
