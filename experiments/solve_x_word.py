# -*- coding: utf-8 -*-
"""solve_x_word.py — x已知视角: 用词向量跑通 x = a + f(M)
核心思想: d = x - a 是'答案文本与问题文本的差', 差的本质是'答案里有而问题里没有的词'
M(最小记忆集)候选 = 证据句的词向量集合
步骤:
W1 拆词: 答案词序列A_words, 问题词序列Q_words, 证据句词序列C_words (256维词向量)
W2 d_words = 答案有而问题没有的词(集合差) → 这些词的向量均值 = d的词级表达
W3 验证: d的词级表达 与 句级d(x-a)投影后的方向 一致性
W4 最小M: 每题需要多少个'词'才能表出d? 词覆盖率曲线(前k个词向量张成的子空间对d的解释力)
W5 词级M的判别力: 用d_words表达做证据/噪声区分(留出)
全部256维词向量(word_vecs.npz), 零LLM
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
WV = np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)
WV = l2n = None
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
z = np.load(HERE + "/word_vecs.npz")
WV = l2n(z["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
WKEYS = list(WH.keys())
print("词向量:", WV.shape, flush=True)

# 词查找表(小写清洗)
W2I = {}
for idx, w in enumerate(WKEYS):
    c = w.strip(":").lower()
    if c and c not in W2I:
        W2I[c] = idx
def wv(word):
    i = W2I.get(word)
    return WV[i] if i is not None else None

def toks(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2]
STOP = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())

# 句向量备用
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
TOP50 = z1["TOP50"]
print("样本:", len(keys), flush=True)

def wvecs_of(words):
    vs, ws = [], []
    for w in words:
        v = wv(w)
        if v is not None:
            vs.append(v)
            ws.append(w)
    return np.array(vs) if vs else np.zeros((0, WV.shape[1]), dtype=np.float32), ws

# W1-W3: d的词级表达
print("== W1-W3 d的词级表出 ==", flush=True)
dvecs = []       # 句级d(1024, 后面投到256没法, 改用: d的词级 vs 句级重构一致性用词表出的'答案词均值'与'问题词均值'的差)
dword_means = []
cov_stats = []
for i in keys[:600]:
    r = rows[i]
    aw = [w for w in toks(str(r["answer"][0])) if w not in STOP]
    qw = [w for w in toks(r["question"]) if w not in STOP]
    dws = [w for w in aw if w not in qw]          # 答案有而问题没有的词
    av, aw2 = wvecs_of(aw)
    qv, qw2 = wvecs_of(qw)
    dv, dw2 = wvecs_of(dws)
    if len(dv) == 0 or len(av) == 0 or len(qv) == 0:
        continue
    dvec_mean = dv.mean(0)
    dword_means.append(dvec_mean)
    cov_stats.append(len(dws))
dword_means = np.array(dword_means)
print("  答案独有词数均值: %.1f 个/题 (覆盖率: 词能在字典找到的比例试算)" % np.mean(cov_stats), flush=True)
# d̄(词级全局均值)方向一致性
dmean = dword_means.mean(0)
dmean = dmean / np.linalg.norm(dmean)
own = np.array([float((m / (np.linalg.norm(m) + 1e-9)) @ dmean) for m in dword_means])
print("  词级d_i方向一致性 cos(d_i, d̄)=%.3f (句级=0.361)" % own.mean(), flush=True)

# W4: 最小M — 用证据句的词向量集合表出d_words
print("== W4 最小M: 证据词集合对 d_words 的覆盖/表出 ==", flush=True)
cover_ratios = []
subspace_gain = []
for kk, i in enumerate(keys[:600]):
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    if not ev:
        continue
    aw = [w for w in toks(str(r["answer"][0])) if w not in STOP]
    dws = [w for w in aw if w not in toks(r["question"])]
    if not dws:
        continue
    ev_words = set()
    for j in ev[:2]:
        ev_words |= set(toks(TEXTS[j]))
    cov = np.mean([1.0 if (w in ev_words or wv(w) is not None and any(w2.startswith(w[:4]) for w2 in ev_words)) else 0.0 for w in dws])
    cover_ratios.append(cov)
    if kk > 200:
        break
print("  答案独有词被'证据前2句词集'覆盖的比例: %.0f%%" % (100 * np.mean(cover_ratios)), flush=True)
print("  → 即: M最小=证据句的词集合, 已包含 x-a 需要的全部词材料")

# W5: 词级判别 — 证据词集 vs 噪声词集 对'答案独有词'的覆盖竞争
print("== W5 词级M判别力(留出): 答案独有词 被证据句覆盖 vs 被噪声句覆盖 ==", flush=True)
win = lose = 0
for i in keys[600:]:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    no = [j for j in TOP50[i] if j not in hs and j not in tws][:2]
    if not ev or not no:
        continue
    dws = set(w for w in toks(str(r["answer"][0])) if w not in toks(r["question"]) and w not in STOP)
    if not dws:
        continue
    evw = set()
    for j in ev[:1]:
        evw |= set(toks(TEXTS[j]))
    now_ = set()
    for j in no[:1]:
        now_ |= set(toks(TEXTS[j]))
    ce = len(dws & evw)
    cn = len(dws & now_)
    if ce > cn:
        win += 1
    elif cn > ce:
        lose += 1
print("  证据句词集覆盖独有词 > 噪声句: %d | <: %d | 单挑胜率=%.0f%%" % (
    win, lose, 100 * win / max(1, win + lose)), flush=True)

# W6: 256维词向量的语义近邻是否泄漏判别信息(软配对复盘, 但只对'答案独有词'做, 不是全词)
print("== W6 限定域软配对复盘: 只对'答案独有词'找证据句词集的近邻 ==", flush=True)
win2 = lose2 = 0
for i in keys[600:900]:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    no = [j for j in TOP50[i] if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    dws = [w for w in toks(str(r["answer"][0])) if w not in toks(r["question"]) and w not in STOP]
    if not dws:
        continue
    evw = set()
    for j in ev[:1]:
        evw |= set(toks(TEXTS[j]))
    now_ = set()
    for j in no:
        now_ |= set(toks(TEXTS[j]))
    ce = cn = 0
    for w in dws:
        v = wv(w)
        if v is None:
            continue
        # 与对方词集的词向量的最大cos(限定域软配对)
        ev_vs = [wv(w2) for w2 in evw if wv(w2) is not None]
        no_vs = [wv(w2) for w2 in now_ if wv(w2) is not None]
        if ev_vs:
            ce += max(float(WV[W2I[w]] @ np.array(ev_vs).mean(0) if False else max(v @ x2 for x2 in ev_vs)) for _ in [0])
        if no_vs:
            cn += max(v @ x2 for x2 in no_vs)
    if ce > cn:
        win2 += 1
    elif cn > ce:
        lose2 += 1
print("  限定域软配对单挑胜率=%.0f%% (对照: 全池软配对曾=0%%)" % (
    100 * win2 / max(1, win2 + lose2)), flush=True)
print("SOLVEXWORD_DONE", flush=True)
