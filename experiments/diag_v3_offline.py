# -*- coding: utf-8 -*-
"""diag_v3_offline.py — V3联立验证包(全离线零GLM)
E1 簇化(星形上下文): 种子top5→同簇(相似>0.7)+邻句→验证①证据包含率②干扰项隔离率
E2 具体性轴: 实体句-代词句, 预测cos(轴,Δ)∈0.2~0.4, 低密度层不反转
E3 Δ残差: Δ_i−Δ全局 后与q_i相关性是否升(0.165→?)
E4 低秩双线性: qᵀUVᵀe, 秩2/4/8, 拆半(判别式, 对比Ridge灾难0.5)
"""
import io, json, os, sys, re, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("bge loaded", flush=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
bQ = emb(Q)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

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
half = len(keys) // 2
print("matched:", len(targets), flush=True)

def per_item(i):
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    return hs, tws

# ============ E1 簇化 ============
print("== E1 星形上下文(种子top5→簇+邻句, 预算25条) ==", flush=True)
t0 = time.time()
ev_in = noise_in = total_in =干扰 = 0
multi_imp = multi_tot = 0
for i in keys:
    hs, tws = per_item(i)
    top = TOP50[i][np.argsort(-RER[i])][:8]  # 种子: 精排前8
    seed_set = set(top)
    stars = []
    used = set()
    for s0 in top:
        if s0 in used:
            continue
        star = [s0] + [j + 1 for j in [s0] if j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]]
        # 同簇扩展
        cl = [j for j in TOP50[i] if j not in used and j not in star and float(D[s0] @ D[j]) >= 0.7][:6]
        star += cl
        stars.append(star)
        used |= set(star)
    ctx = []
    for st in stars:
        for j in st:
            if j not in ctx:
                ctx.append(j)
    ctx = ctx[:25]
    ctxset = set(ctx)
    ev_hit = len(hs & ctxset) + len(tws & ctxset)
    total_in += len(hs | tws)
    ev_in += ev_hit
    # 干扰项: ctx里非证据非孪生的记录数(隔离率=噪声预算是否缩小)
    noise_in += sum(1 for j in ctx if j not in hs and j not in tws)
    if len(hs | tws) > 1:
        multi_tot += 1
        if ev_hit >= 2:
            multi_imp += 1
print("  证据(含孪生)纳入率: %.1f%% (%d/%d)" % (100 * ev_in / total_in, ev_in, total_in))
print("  上下文噪声条数均值: %.1f (臂B top25应为~23)" % (noise_in / len(keys)))
print("  多证据题双片纳入率: %.1f%% (%d/%d)" % (100 * multi_imp / multi_tot, multi_imp, multi_tot))
print("  E1 done %.0fs" % (time.time() - t0), flush=True)

# ============ E2 具体性轴 ============
print("== E2 具体性轴 ==", flush=True)
spec = ["Melanie adopted three cats last year.", "Caroline moved from Sweden four years ago.",
        "John graduated from MIT in 2019.", "The concert cost 45 dollars last Friday in Kyoto.",
        "Nate won the Street Fighter tournament in January 2022.", "Gina opened her clothing store in February 2023.",
        "Deborah works as a pediatric nurse at Boston Children Hospital.", "Tim read Sapiens by Harari last month.",
        "Audrey painted a sunset over the lake in 2022.", "Sam runs five miles every morning."]
vague = ["She adopted some of them a while ago.", "He moved from somewhere years ago.",
         "They graduated from a college once.", "It cost some money recently somewhere.",
         "Someone won something at an event a while back.", "She opened a shop earlier this year.",
         "Someone works as a nurse at a hospital.", "He read a book recently.",
         "Someone painted something some time ago.", "A person runs regularly."]
sa = emb(spec).mean(0) - emb(vague).mean(0)
sa /= np.linalg.norm(sa)
DELTA = None
evA, noA = [], []
for i in keys[:half]:
    hs, tws = per_item(i)
    top = TOP50[i]
    evA += [j for j in top if j in hs or j in tws]
    noA += [j for j in top if j not in hs and j not in tws][:8]
evA, noA = np.array(evA), np.array(noA)
DELTA = D[evA].mean(0) - D[noA].mean(0)
DELTA /= np.linalg.norm(DELTA)
print("  cos(具体性轴, Δ) = %.3f (预测0.2~0.4)" % float(sa @ DELTA), flush=True)
PROJ_sa = (D @ sa).astype(np.float32)
PROJ_d = (D @ DELTA).astype(np.float32)
info = ["My favorite book is The Pragmatic Programmer by Hunt.", "We visited Kyoto in April 2019 for cherry blossoms.", "She works as a pediatric nurse at the children hospital.", "The concert tickets cost 45 dollars each last Friday.", "My brother graduated from MIT with computer science degree."] * 6
fluff = ["That sounds really great and awesome!", "Thanks so much for your support friend!", "Wow I cannot believe it amazing!", "Have a wonderful day and take care!", "It was so much fun hanging out together!"] * 6
ia = emb(info).mean(0) - emb(fluff).mean(0)
ia /= np.linalg.norm(ia)
dens_e = PROJ_sa[evA]
q1, q2 = np.percentile(dens_e, [33, 66])
bkt = np.where(dens_e < q1, 0, np.where(dens_e < q2, 1, 2))
evB, noB = [], []
for i in keys[half:]:
    hs, tws = per_item(i)
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
for b, nm in ((0, "低"), (1, "中"), (2, "高")):
    se = evB[(np.where(PROJ_sa[evB] < q1, 0, np.where(PROJ_sa[evB] < q2, 1, 2))) == b]
    x = np.concatenate([PROJ_sa[se], PROJ_sa[noB]])
    lb = np.zeros(len(x), dtype=bool); lb[:len(se)] = True
    print("  具体性轴 %s密度层: AUC=%.3f (预测低层不反转>0.5)" % (nm, auc(x, lb)), flush=True)

# ============ E3 Δ残差 ============
print("== E3 Δ残差的问题特异性 ==", flush=True)
res_dcos, raw_dcos, rand_dcos = [], [], []
rng = np.random.default_rng(9)
allkeys = keys
for i in allkeys:
    hs, tws = per_item(i)
    top = TOP50[i]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    if not ev or len(no) < 3:
        continue
    di = l2n((D[ev].mean(0) - D[no].mean(0))[None])[0]
    resid = l2n((di - (di @ DELTA) * DELTA)[None])[0]
    res_dcos.append(float(resid @ bQ[i]))
    raw_dcos.append(float(di @ bQ[i]))
for _ in range(400):
    i = allkeys[rng.integers(0, len(allkeys))]
    j = allkeys[rng.integers(0, len(allkeys))]
    if i == j:
        continue
    hs, tws = per_item(i)
    ev = [x for x in TOP50[i] if x in hs or x in tws]
    no = [x for x in TOP50[i] if x not in hs and x not in tws]
    if not ev or len(no) < 3:
        continue
    di = l2n((D[ev].mean(0) - D[no].mean(0))[None])[0]
    resid = l2n((di - (di @ DELTA) * DELTA)[None])[0]
    rand_dcos.append(float(resid @ bQ[j]))
print("  本题: 原始%.3f → 剥Δ后%.3f (预测升至0.25+)" % (np.mean(raw_dcos), np.mean(res_dcos)), flush=True)
print("  他人: %.3f (应≈0)" % np.mean(rand_dcos), flush=True)

# ============ E4 低秩双线性 ============
print("== E4 低秩双线性 qᵀUVᵀe (拆半) ==", flush=True)
def build_pairs(sel_keys):
    P = []
    for i in sel_keys:
        hs, tws = per_item(i)
        top = TOP50[i]
        ev = [j for j in top if j in hs or j in tws][:3]
        no = [j for j in top if j not in hs and j not in tws][:6]
        for j in ev:
            P.append((bQ[i], D[j], 1))
        for j in no:
            P.append((bQ[i], D[j], 0))
    return P
PA, PB = build_pairs(keys[:half]), build_pairs(keys[half:])
for rank in (2, 4, 8):
    # 交替最小二乘简化: 学对角缩放的秩rank近似 —— 用 F = U diag(s) V^T, U来自q PCA, V来自e PCA
    Qm = np.stack([p[0] for p in PA])
    Em = np.stack([p[1] for p in PA])
    y = np.array([p[2] for p in PA], dtype=np.float64)
    # 特征: q*e逐维(对角双线性) + 低秩投影交互
    Uq, _, _ = np.linalg.svd(Qm - Qm.mean(0), full_matrices=False)
    Ve, _, _ = np.linalg.svd(Em - Em.mean(0), full_matrices=False)
    Fq = Uq[:, :rank]
    Fe = Ve[:, :rank]
    X = np.stack([np.concatenate([p[0] * p[1], (p[0] @ Fq.T) * (p[1] @ Fe.T)]) for p in PA])
    Xb = np.stack([np.concatenate([p[0] * p[1], (p[0] @ Fq.T) * (p[1] @ Fe.T)]) for p in PB])
    # 线性判别(对角协方差朴素贝叶斯)
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    s1 = X[y == 1].var(0) + 1e-6
    s0 = X[y == 0].var(0) + 1e-6
    def score(x):
        return float(np.sum((x - mu0) ** 2 / s0) - np.sum((x - mu1) ** 2 / s1))
    yB = np.array([p[2] for p in PB], dtype=bool)
    sB = np.array([score(x) for x in Xb])
    a = auc(sB, yB)
    print("  秩%d: B半AUC=%.3f (Ridge灾难=0.5, 预测0.70~0.78)" % (rank, a), flush=True)
print("V3_OFFLINE_DONE", flush=True)
