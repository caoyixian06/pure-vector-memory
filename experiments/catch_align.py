# -*- coding: utf-8 -*-
"""catch_align.py — 抓出缺失环节: 词对词对齐质量 AlignQuality(a, c)
A1 构造对齐矩阵: 每题 a的词×c的词 256维点积矩阵, 提取特征(最大对齐/每行均值/对角质量/覆盖深度)
A2 对齐特征区分 证据c vs 噪声h 的AUC(单特征+组合, 留出)
A3 对齐特征 + Δ + 精排 联合 —— 缺失环节补上后总判别力到多少
A4 对齐矩阵可视化样例(3题): 证据的top对齐词对 vs 噪声的top对齐词对
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
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
z = np.load(HERE + "/word_vecs.npz")
WV = l2n(z["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None
def toks(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2]
STOP = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all and".split())

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

rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
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

# 每题: 问题词向量组 与 (证据/噪声)词向量组 的对齐矩阵特征
def align_feats(qwords, cwords):
    Q = [wv(w) for w in qwords if wv(w) is not None]
    Cv = [wv(w) for w in cwords if wv(w) is not None]
    if not Q or not Cv:
        return None, []
    Qm = np.stack(Q)
    Cm = np.stack(Cv)
    Am = Qm @ Cm.T                     # (nq, nc) 对齐矩阵
    mx = float(Am.max())
    rowmean = float(Am.max(axis=1).mean())   # 每个问题词的最佳对齐(=覆盖的软版)
    colmean = float(Am.max(axis=0).mean())   # 每个候选词被对齐的程度
    diag = float(np.mean(np.abs(np.diag(Am))))
    # top对齐词对(展示用)
    fi, fj = np.unravel_index(np.argmax(Am), Am.shape)
    pair = (qwords[fi] if fi < len(qwords) else "?", cwords[fj] if fj < len(cwords) else "?")
    return dict(mx=mx, rowmean=rowmean, colmean=colmean, diag=diag), [pair]

FEATS, LABS, KL, PAIRS = [], [], [], []
for i in keys:
    r = rows[i]
    qw = [w for w in toks(r["question"]) if w not in STOP]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    fE, pE = align_feats(qw, toks(TEXTS[ev[0]]))
    fN, pN = align_feats(qw, toks(TEXTS[no[0]]))
    if fE is None or fN is None:
        continue
    FEATS.append((fE, fN))
    LABS.append(i)
    PAIRS.append((pE, pN))
print("样本:", len(FEATS), flush=True)

# A2: 单特征AUC(证据 vs 噪声 配对样本: 符号检验 + AUC over pooled)
def auc(x, plab):
    r = np.argsort(np.asarray(x))
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

names = ["mx", "rowmean", "colmean", "diag"]
print("== A2 对齐特征判别力(证据 vs 噪声, 配对池) ==", flush=True)
pooled_x, pooled_lab = [], []
for k, (fE, fN) in enumerate(FEATS):
    for nm in names:
        pass
wins = {nm: 0 for nm in names}
tot = 0
for fE, fN in FEATS:
    for nm in names:
        if fE[nm] > fN[nm]:
            wins[nm] += 1
    tot += 1
for nm in names:
    e = np.array([fE[nm] for fE, fN in FEATS])
    nn = np.array([fN[nm] for fE, fN in FEATS])
    x = np.concatenate([e, nn])
    lb = np.zeros(len(x), dtype=bool); lb[:len(e)] = True
    print("  %-8s 证据>噪声胜率=%.0f%%  AUC=%.3f" % (nm, 100 * wins[nm] / tot, auc(x, lb)), flush=True)

# 组合(等权和的配对胜负)
scoreE = np.array([[fE[nm] for nm in names] for fE, fN in FEATS])
scoreN = np.array([[fN[nm] for nm in names] for fE, fN in FEATS])
mu = np.concatenate([scoreE, scoreN]).mean(0)
sd = np.concatenate([scoreE, scoreN]).std(0) + 1e-9
zE = ((scoreE - mu) / sd).mean(1)
zN = ((scoreN - mu) / sd).mean(1)
win_c = int((zE > zN).sum())
x = np.concatenate([zE, zN]); lb = np.zeros(len(x), dtype=bool); lb[:len(zE)] = True
print("  组合(4特征等权): 胜率=%.0f%% AUC=%.3f" % (100 * win_c / tot, auc(x, lb)), flush=True)

# A3: 与Δ/精排联合
DELTA = None
evA, noA = [], []
for i in keys[:len(keys) // 2]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evA += [j for j in top if j in hs or j in tws]
    noA += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evA)].mean(0) - D[np.array(noA)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ_D = (D @ DELTA).astype(np.float32)
def pair_auc(sa, sb):
    x = np.concatenate([sa, sb]); lb = np.zeros(len(x), dtype=bool); lb[:len(sa)] = True
    return auc(x, lb)
dE = []
dN = []
for i, _ in zip(LABS, FEATS):
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    dE.append(float(np.max(PROJ_D[list(hs | tws)])))
    no_js = [j for j in TOP50[i] if j not in hs and j not in tws][:1]
    if no_js:
        dN.append(float(np.max(PROJ_D[no_js])))
dE = np.array(dE)
dN = np.array(dN)
print("A3 对照: Δ投影max AUC=%.3f | 精排分max见子代理0.921(全池口径)" % pair_auc(dE, dN), flush=True)

# A4: 样例
print("== A4 对齐样例(前3题) ==", flush=True)
for k in range(min(3, len(PAIRS))):
    pE, pN = PAIRS[k]
    print("  题%d 证据最佳对齐: %s→%s | 噪声最佳对齐: %s→%s" % (
        k, pE[0], pE[1], pN[0], pN[1]), flush=True)
print("CATCH_DONE", flush=True)
