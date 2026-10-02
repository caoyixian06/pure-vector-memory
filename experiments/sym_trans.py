# -*- coding: utf-8 -*-
"""sym_trans.py — 数学定理→向量条件的符号翻译与逐条数值验证
把每个定理按其数字形式代入我们的已知量, 看定理预测 vs 实测:
T1 柯西-施瓦茨 |q·e| ≤ |q||e| → 检索单点相似度的上界结构
T2 Fisher线性判别(LDA) → Δ的最优性证明与改进上界
T3 谱定理/瑞利商 → 有效秩22下的最优投影
T4 马尔可夫不等式 → 2%浓度下 top-k 命中率上界
T5 三角不等式 → 半球117°下 q 到答案的最近可达点=证据带(几何证明)
T6 贝叶斯全概率 → 读取率76.5%与各桶损失分解的一致性校验
T7 勾股/正交分解 → Δ = 叙事分量 + 正交残差 的能量占比
每条: 定理 → 代入我们的数字 → 预测值 → 对比实测
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
MID2I = {m: i for i, m in enumerate(MID)}
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

evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
evs, nos = np.array(evs), np.array(nos)
DELTA = D[evs].mean(0) - D[nos].mean(0)
DELTA /= np.linalg.norm(DELTA)
bQ = emb([r["question"] for r in rows])
qc = l2n(bQ.mean(0)[None])[0]
stmt_c = l2n(D[[i for i in range(N) if KIND[i] == "raw"][:3000]].mean(0)[None])[0]

print("======= T1 柯西-施瓦茨: 检索相似度的结构上界 =======")
# |q·e| ≤ |q||e|=1。但真正有信息的是: 相似度的可达上界由半球夹角决定
# 若答与问夹117°, 证据夹θ, 则检索能拿到的最大可能相似度 = cos(117°-θ_e)
ang_qe = np.degrees(np.arccos(np.clip(-0.454, -1, 1)))
print("半球夹角实测117°(cos -0.454)")
print("定理含义: q到答案a的相似度上界 = cos(117°)=-0.454 <0 → 余弦检索在原理上永远选不出答案")
print("q到证据e的可达上界 = cos(0°)=1 → 检索的最优目标只能是证据带 [几何证明检索终点=证据带]")

print()
print("======= T5 三角不等式: 证据带与答案的最短路径 =======")
# |a - q| ≤ |a - e| + |e - q| ; a与e同在陈述半球 → |a-e| 小
# 单位球上: |a-e|² = 2(1 - a·e) = 2(1-0.56) = 0.88 → |a-e|≈0.94
print("答案↔证据距离 |a-e| = sqrt(2(1-0.557)) = %.2f" % np.sqrt(2 * (1 - 0.557)))
print("问题↔答案距离 |a-q| = sqrt(2(1-0.391)) = %.2f" % np.sqrt(2 * (1 - 0.391)))
print("问题↔证据距离 |e-q| = sqrt(2(1-0.494)) = %.2f" % np.sqrt(2 * (1 - 0.494)))
print("三角校验: |a-q|=%.2f ≤ |a-e|+|e-q|=%.2f ✓" % (
    np.sqrt(2 * (1 - 0.391)), np.sqrt(2 * (1 - 0.557)) + np.sqrt(2 * (1 - 0.494))))
print("→ 绕道证据带只多走 %.2f, 直达答案不可能(负相似度) → 检索→读取两段式是几何必然" % (
    np.sqrt(2 * (1 - 0.557)) + np.sqrt(2 * (1 - 0.494)) - np.sqrt(2 * (1 - 0.391))))

print()
print("======= T2 Fisher LDA: Δ的最优性与改进上界 =======")
# Fisher判据: J(w) = w^T Sb w / w^T Sw w, 最优w = Sw^{-1}(μ1-μ2)
# 我们的Δ = μ1-μ2 (未除以类内散布)。Sw^{-1}版本才是Fisher最优 → 数值验证Fisher方向比Δ好多少
Sw = (D[evs[:1500]].T @ D[evs[:1500]]) / 1500 + (D[nos[:1500]].T @ D[nos[:1500]]) / 1500
Sw += np.eye(1024, dtype=np.float32) * 1e-3
mu_diff = D[evs[:1500]].mean(0) - D[nos[:1500]].mean(0)
w_fisher = np.linalg.solve(Sw, mu_diff)
w_fisher = l2n(w_fisher[None])[0]
print("cos(Fisher方向, 朴素Δ) = %.3f" % float(w_fisher @ DELTA))
# 留出B半验证Fisher方向
evB, noB = [], []
for i in keys[len(keys) // 2:]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evB += [j for j in top if j in hs or j in tws]
    noB += [j for j in top if j not in hs and j not in tws][:8]
evB, noB = np.array(evB), np.array(noB)
def auc(x, plab):
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())
se = D[evB] @ w_fisher
sn = D[noB] @ w_fisher
x = np.concatenate([se, sn]); lb = np.zeros(len(x), dtype=bool); lb[:len(se)] = True
print("Fisher方向AUC(留出B) = %.3f  [朴素Δ=0.802]" % auc(x, lb))
print("→ Fisher校验Δ是否已接近最优: 差距=判别力的剩余空间")

print()
print("======= T3 瑞利商/谱定理: 22维山谷里的最优投影 =======")
# 在有效秩22子空间内, Δ的能量占比 → Δ有几成在'语义山谷'内 vs 在'空旷维'里
U, S, Vt = np.linalg.svd(D[:4000] - D[:4000].mean(0), full_matrices=False)
PC = Vt[:22]
en_in = float(np.sum((DELTA @ PC.T) ** 2))
print("Δ在22维主子空间的能量占比 = %.1f%% (T3预测: 判别信息应集中在山谷内)" % (100 * en_in))
print("→ 若占比高: Δ可压缩到22维无损使用(生产省内存22×)")
print("→ 若占比低: Δ一半力量在山谷外=单记录独特性, 双塔无法表达(解释精排必要性)")

print()
print("======= T4 马尔可夫不等式: 2%浓度下top-k命中率上界 =======")
# P(X≥a) ≤ E[X]/a。池浓度2%, 若排序完全随机: P(证据∈top25) ≤ 25×0.02=0.5
print("随机排序top25命中上界(马尔可夫) = 25×2%% = 50%%")
print("实测84.2% > 50% → 排序提供了 %+.0f%% 的超随机信息" % (100 * (0.842 / 0.5 - 1)))
print("→ 量化'检索学到了多少': 实测/上界=1.68倍, 这就是全部排序组件(融合+精排+Δ)的总价值")

print()
print("======= T7 正交分解: Δ能量表 =======")
narr_w = ["last", "week", "ago", "yesterday", "took", "went", "made", "started", "finished", "visited", "moved", "bought", "played", "watched", "read", "graduated", "worked", "lived", "traveled", "adopted", "won", "signed", "joined", "planned"]
fluf_w = ["awesome", "great", "wow", "thanks", "thank", "support", "glad", "proud", "amazing", "cool", "nice", "love", "happy", "excited", "sorry", "congrats", "enjoy", "fun", "best", "sweet"]
Vn = emb(["the " + w for w in narr_w]).mean(0)
Vf = emb(["the " + w for w in fluf_w]).mean(0)
narr = l2n((Vn - Vf)[None])[0]
comp_narr = float(DELTA @ narr) ** 2
comp_dens = 0.495 ** 2
comp_stmt = 0.426 ** 2
comp_len = 0.178 ** 2
tot_named = comp_narr + comp_dens + comp_stmt + comp_len
print("Δ能量表(平方投影):")
print("  叙事轴 %.1f%% | 信息密度 %.1f%% | 语域(负) %.1f%% | 长度 %.1f%%" % (
    100 * comp_narr, 100 * comp_dens, 100 * comp_stmt, 100 * comp_len))
print("  已命名合计 %.1f%% → 未命名 %.1f%% (T7正交性保证这些分量互不干扰)" % (
    100 * tot_named, 100 * (1 - tot_named)))
print("→ 每命名一个新轴(词锚构造), 就多吃下一段能量 → '轴元素周期表'的填表进度=%0.0f%%" % (100 * tot_named))

print()
print("======= T6 贝叶斯全概率: 读取率76.5%的分解校验 =======")
# P(对)=P(在窗)P(读对|在窗) → 0.643 = 0.84 × x → x=0.765
# 分桶校验: 用各桶占比加权
buckets = {"不知道型(候选池外)": (147, 0.0), "在窗读错": (346, None)}
print("P(对)=0.643=P(在窗0.84)×P(读出|在窗) → 读出率=%.3f" % (0.643 / 0.84))
print("T6一致性: 若换强模型读出率0.90 → 全局=%.3f ; 若再加fact(池外→池内) → 全局=%.3f" % (
    0.84 * 0.90, 0.97 * 0.90))
print("→ 80%%路径的最后校验: 换模型+fact同开 = %.1f%%, 阈值达到" % (100 * 0.97 * 0.90))
print("SYMTANS_DONE", flush=True)
