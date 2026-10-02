# -*- coding: utf-8 -*-
"""foundry132.py — 200题×256维: 金证据 vs 噪声 差异解剖 (用户指令 2026-09-19)
用200个问题与全库匹配(256维), 对比金证据记录与噪声记录:
  层1 标量: 全库cos金/噪 gap/AUC, 最佳金vs最佳噪margin, 硬噪声(top50)AUC, 名次
  层2 维度: 逐维金噪质心差(库内z, 池化200题), vs全噪 & vs硬噪(top100)两版
           product空间(q*r逐维z)金噪差
  层3 结构: 金内聚(金金cos) vs 金-硬噪cos, 原始范数金vs噪
  参考: 同题1024维cos对照
"""
import io, json, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry132_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s); LOG.write(s + "\n"); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def auc_pn(pos, neg):
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * max(1, len(neg)))

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        d = json.loads(l)
        REC[d.get("memory_id")] = d
    except Exception:
        pass

def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""

RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
QWRAW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QWRAW[i] = v
QW = l2n(QWRAW)
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
Q256 = l2n(np.load(HERE + "/q256_cache.npz")["Q"].astype(np.float32))
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32)

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g

P("loaded NR=%d nQ=%d %.0fs" % (NR, len(IDS), time.time() - t0))
cand = [k for k, qa in enumerate(IDS) if gold_set(qa)]
rng = np.random.RandomState(20260919)
pick = rng.choice(len(cand), min(200, len(cand)), replace=False)
SEL = [cand[i] for i in pick]
GSEL = [gold_set(IDS[k]) for k in SEL]
P("selected %d questions (of %d gold-bearing)" % (len(SEL), len(cand)))

mu = QW.mean(axis=0)
sd = QW.std(axis=0) + 1e-9
Zc = (QW - mu) / sd
Zsum = Zc.sum(axis=0)

sc = dict(margin=[], gap=[], gap_hard=[], auc=[], auc_hard=[], rank=[], gg=[], gh=[],
          m1024=[], gap1024=[], auc1024=[], auc1024h=[])
DD_all, DD_hard, PD_all, PD_hard = [], [], [], []
gold_pool, noise_pool = [], []
gn_pool, nn_pool = [], []
pos = np.empty(NR, dtype=np.int32)

for t, k_i in enumerate(SEL):
    G = GSEL[t]
    gidx = np.array(sorted(G), dtype=np.int64)
    ng = len(gidx)
    qv = Q256[k_i]
    cq = QW @ qv
    gmask = np.zeros(NR, dtype=bool)
    gmask[gidx] = True
    gv = cq[gidx]
    nv = cq[~gmask]
    gold_pool.append(gv)
    noise_pool.append(nv[rng.choice(len(nv), min(2000, len(nv)), replace=False)])
    sc["margin"].append(float(gv.max() - nv.max()))
    sc["gap"].append(float(gv.mean() - nv.mean()))
    sc["auc"].append(auc_pn(gv, nv))
    nn_idx = np.where(~gmask)[0]
    hard100 = nn_idx[np.argsort(-cq[nn_idx])[:100]]
    hard50 = nn_idx[np.argsort(-cq[nn_idx])[:50]]
    sc["gap_hard"].append(float(gv.mean() - cq[hard50].mean()))
    sc["auc_hard"].append(auc_pn(gv, cq[hard50]))
    desc = np.argsort(-cq)
    pos[desc] = np.arange(NR)
    bg = int(gidx[np.argmax(cq[gidx])])
    sc["rank"].append(int(pos[bg]) + 1)
    if ng >= 2:
        GM = QW[gidx] @ QW[gidx].T
        iu = np.triu_indices(ng, 1)
        sc["gg"].append(float(GM[iu].mean()))
        sc["gh"].append(float((QW[gidx] @ QW[hard50].T).mean()))
    gn = np.linalg.norm(QWRAW[gidx], axis=1)
    gn_pool.append(gn[gn > 0])
    idx_n = nn_idx[::40]
    nz = np.linalg.norm(QWRAW[idx_n], axis=1)
    nn_pool.append(nz[nz > 0])
    z_g = Zc[gidx].mean(axis=0)
    z_n_all = (Zsum - Zc[gidx].sum(axis=0)) / max(1, NR - ng)
    z_n_hard = Zc[hard100].mean(axis=0)
    DD_all.append(z_g - z_n_all)
    DD_hard.append(z_g - z_n_hard)
    PR = QW * qv[None, :]
    pm = PR.mean(axis=0)
    ps = PR.std(axis=0) + 1e-9
    pz_g = ((PR[gidx] - pm) / ps).mean(axis=0)
    pz_n_all = ((PR.sum(axis=0) - PR[gidx].sum(axis=0)) / max(1, NR - ng) - pm) / ps
    pz_n_hard = ((PR[hard100] - pm) / ps).mean(axis=0)
    PD_all.append(pz_g - pz_n_all)
    PD_hard.append(pz_g - pz_n_hard)
    qv2 = X[k_i] / (np.linalg.norm(X[k_i]) + 1e-9)
    cq2 = D @ qv2
    gv2 = cq2[gidx]
    nv2 = cq2[~gmask]
    sc["m1024"].append(float(gv2.max() - nv2.max()))
    sc["gap1024"].append(float(gv2.mean() - nv2.mean()))
    sc["auc1024"].append(auc_pn(gv2, nv2))
    hard50b = nn_idx[np.argsort(-cq2[nn_idx])[:50]]
    sc["auc1024h"].append(auc_pn(gv2, cq2[hard50b]))
    if (t + 1) % 50 == 0:
        P("  %d/%d %.0fs" % (t + 1, len(SEL), time.time() - t0))

A = lambda k: np.array(sc[k], dtype=np.float64)
GP = np.concatenate(gold_pool)
NP = np.concatenate(noise_pool)
mar = A("margin")
rk = np.array(sc["rank"])
P("")
P("===== 层1 标量: 256维cos 金 vs 噪 (n=%d题) =====" % len(SEL))
P("金cos均值 %.4f | 噪(全库抽样)cos均值 %.4f | 池化gap %.4f" % (GP.mean(), NP.mean(), GP.mean() - NP.mean()))
P("逐题gap中位 %.4f | gap>0.02题占 %.1f%%" % (np.median(A("gap")), 100.0 * (A("gap") > 0.02).mean()))
P("逐题AUC(金vs全噪) 均值 %.4f 中位 %.4f" % (A("auc").mean(), np.median(A("auc"))))
P("硬噪AUC(金vs top50噪) 均值 %.4f 中位 %.4f" % (A("auc_hard").mean(), np.median(A("auc_hard"))))
P("局部gap(金均值-top50噪均值) 中位 %.4f" % np.median(A("gap_hard")))
P("margin(最佳金-最佳噪) 中位 %.4f | >0占 %.1f%% | >0.02占 %.1f%% | <-0.02占 %.1f%%" % (
    np.median(mar), 100.0 * (mar > 0).mean(), 100.0 * (mar > 0.02).mean(), 100.0 * (mar < -0.02).mean()))
P("最佳金名次: 中位 %d | @1 %.1f%% @5 %.1f%% @15 %.1f%%" % (
    int(np.median(rk)), 100.0 * (rk <= 1).mean(), 100.0 * (rk <= 5).mean(), 100.0 * (rk <= 15).mean()))
P("[1024对照] gap中位 %.4f | AUC %.4f | 硬噪AUC %.4f | margin中位 %.4f" % (
    np.median(A("gap1024")), A("auc1024").mean(), A("auc1024h").mean(), np.median(A("m1024"))))
P("")
P("===== 层3 结构 =====")
if sc["gg"]:
    P("金金内聚cos %.4f vs 金-硬噪cos %.4f (差 %.4f)" % (
        np.mean(sc["gg"]), np.mean(sc["gh"]), np.mean(sc["gg"]) - np.mean(sc["gh"])))
P("原始256范数: 金 %.4f vs 噪 %.4f" % (np.concatenate(gn_pool).mean(), np.concatenate(nn_pool).mean()))
P("")
P("===== 层2 维度: 金噪逐维差异(库内z, 池化%d题) =====" % len(SEL))
for nm, DD in (("vs全噪", np.array(DD_all)), ("vs硬噪top100", np.array(DD_hard))):
    eff = DD.mean(axis=0)
    tt = eff / (DD.std(axis=0) / np.sqrt(len(SEL)) + 1e-12)
    sig = int((np.abs(tt) > 4.5).sum())
    top = np.argsort(-np.abs(eff))[:10]
    P("[%s] 显著维(|t|>4.5): %d/256 | top10维: %s" % (
        nm, sig, ", ".join("d%d(%+.3f)" % (d, eff[d]) for d in top)))
for nm, PD in (("product vs全噪", np.array(PD_all)), ("product vs硬噪", np.array(PD_hard))):
    eff = PD.mean(axis=0)
    tt = eff / (PD.std(axis=0) / np.sqrt(len(SEL)) + 1e-12)
    sig = int((np.abs(tt) > 4.5).sum())
    top = np.argsort(-np.abs(eff))[:10]
    P("[%s] 显著维: %d/256 | top10维: %s" % (
        nm, sig, ", ".join("d%d(%+.3f)" % (d, eff[d]) for d in top)))
P("")
P("F132_DONE %.0fs" % (time.time() - t0))
LOG.close()
