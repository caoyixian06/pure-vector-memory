# -*- coding: utf-8 -*-
"""solve_x_word2.py — 词向量方程第二包
X1 独有词的词性/语义结构: 3.1个独有词都是什么(专名词/数字/动词?) — 决定伪fact的模板
X2 覆盖判别的抗噪性: 词覆盖在'问题改写'(同义换词)下稳不稳 — 用hyde假说词替代真答案词测判别衰减
X3 词向量空间的半球: 256词空间里 问题词云 vs 答案词云 是否分离(cos质心/夹角)
X4 覆盖排序的端到端代理: 用独有词(hyde代理)的覆盖率重排top50, 代理指标能涨多少(错题进25/对题进5)
X5 最优独有词数: 用前k个Δ高分句的独有词(而非hyde)做代理, k=1/2/3的判别力
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
    c = w.strip(":").lower()
    if c and c not in W2I:
        W2I[c] = idx
def wv(word):
    i = W2I.get(word)
    return WV[i] if i is not None else None
def toks(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2]
STOP = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())

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
zc = np.load(HERE + "/fusion_cache.npz")

# X1: 独有词结构
print("== X1 独有词结构(3.1词/题都是什么) ==", flush=True)
from collections import Counter
cat = Counter()
DIG = re.compile(r"\d")
CAPS = re.compile(r"^[A-Z][a-z]+$")
samples = []
for i in keys[:800]:
    r = rows[i]
    aw = [w for w in toks(str(r["answer"][0])) if w not in STOP]
    dws = [w for w in aw if w not in toks(r["question"])]
    for w in dws:
        if DIG.search(w):
            cat["数字/日期"] += 1
        elif CAPS.match(w):
            cat["专名(大写开头)"] += 1
        else:
            cat["普通词"] += 1
    if len(samples) < 12 and dws:
        samples.append((r["question"][:40], dws[:4]))
for k, v in cat.most_common():
    print("  %s: %d" % (k, v), flush=True)
print("  样例:", flush=True)
for q, dw in samples:
    print("   Q:%s → 独有词%s" % (q, dw), flush=True)

# X2: 改写鲁棒性 — 用hyde假说词替代真答案独有词, 判别力衰减
print("== X2 改写鲁棒性(hyde词代理) ==", flush=True)
_hc = {}
if os.path.exists(HERE + "/hyde_cache.jsonl"):
    for l in open(HERE + "/hyde_cache.jsonl", encoding="utf-8"):
        if l.strip():
            dd = json.loads(l)
            _hc[dd["qh"]] = dd.get("hyp") or ""
import hashlib
def cached_hyp(q):
    return _hc.get(hashlib.md5(q.encode()).hexdigest()[:16], "")
win_true = win_hyp = 0
tot = 0
for i in keys[600:1100]:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    no = [j for j in TOP50[i] if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    dws_true = set(w for w in toks(str(r["answer"][0])) if w not in toks(r["question"]) and w not in STOP)
    hyp = cached_hyp(r["question"])
    dws_hyp = set(w for w in toks(hyp) if w not in toks(r["question"]) and w not in STOP)
    evw = set()
    for j in ev[:1]:
        evw |= set(toks(TEXTS[j]))
    now_ = set()
    for j in no:
        now_ |= set(toks(TEXTS[j]))
    if dws_true:
        if len(dws_true & evw) > len(dws_true & now_):
            win_true += 1
        tot += 1
    if dws_hyp:
        if len(dws_hyp & evw) > len(dws_hyp & now_):
            win_hyp += 1
print("  真答案独有词: 胜率%.0f%% (n=%d)" % (100 * win_true / max(1, tot), tot), flush=True)
print("  hyde独有词:   胜率%.0f%% (同池, n=%d可用)" % (100 * win_hyp / max(1, tot), tot), flush=True)

# X3: 词空间半球
print("== X3 词空间的问题词云 vs 答案词云 ==", flush=True)
qcs, acs = [], []
for i in keys[:500]:
    r = rows[i]
    qws = [wv(w) for w in toks(r["question"]) if w not in STOP and wv(w) is not None]
    aws = [wv(w) for w in toks(str(r["answer"][0])) if w not in STOP and wv(w) is not None]
    if qws and aws:
        qcs.append(np.mean(qws, axis=0))
        acs.append(np.mean(aws, axis=0))
qcs, acs = np.array(qcs), np.array(acs)
qcm = l2n(qcs.mean(0)[None])[0]
acm = l2n(acs.mean(0)[None])[0]
print("  词云质心夹角cos=%.3f (句空间a-b=0.392)" % float(qcm @ acm), flush=True)
# 词级: 同一句的词云 vs 该句256句向量
E256 = None
print("  (词云质心 vs 句向量的一致性待X4间接验证)", flush=True)

# X4: 覆盖排序端到端代理 — hyde独有词覆盖率重排top50
print("== X4 覆盖重排端到端代理 ==", flush=True)
def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)
m25 = o5 = cnt = 0
base_m25 = base_o5 = 0
for i in keys:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    inside = np.argsort(-RER[i])
    order = top[inside]
    hyp = cached_hyp(r["question"])
    dws = set(w for w in toks(hyp) if w not in toks(r["question"]) and w not in STOP)
    row = np.linspace(50, 1, 50)
    if dws:
        cov = np.array([len(dws & set(toks(TEXTS[j]))) for j in order], dtype=np.float32)
        row = row + 8.0 * zs(cov)
    order2 = order[np.argsort(-row)]
    all_ids = list(hs) + list(tws)
    if not all(h2 in set(order2.tolist()) for h2 in all_ids):
        continue  # 重排后证据滑出50的题跳过(罕见)
    pos = {j: p for p, j in enumerate(order2)}
    rk = min(pos[h2] for h2 in all_ids) + 1
    cnt += 1
    if r.get("llm_score") == 1:
        o5 += rk <= 5
    else:
        m25 += rk <= 25
print("  臂B底+覆盖重排: 错题进25 %d 对题进5 %d (对照臂B: 405/720)" % (m25, o5), flush=True)

# X5: 最优独有词源 — Δ高分句的独有词(零LLM代理)
print("== X5 Δ高分句独有词代理 ==", flush=True)
raw_idx = [i2 for i2 in range(N) if KIND[i2] == "raw"]
stmt = l2n(D[raw_idx[:3000]].mean(0)[None])[0]
bQ = l2n(np.stack([emb_bge([r["question"]])[0] for r in rows]))
bQu = l2n(bQ + (stmt - l2n(bQ.mean(0)[None])[0]))
PROJ = (D @ l2n((bQu[0] * 0 + (stmt - l2n(bQ.mean(0)[None])[0]))[None])[0])
win3 = lose3 = 0
for i in keys[600:1100]:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    ev = [j for j in TOP50[i] if j in hs or j in tws]
    no = [j for j in TOP50[i] if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    # Δ高分句(本题池内Δ投影最高的非证据句)的独有词当代理 — 用high-Δ句作为'预期答案词'来源
    # 简化: 用融合分第2名句(非证据)的独有词 — 它常是同话题的另一段
    fused_top = TOP50[i][np.argsort(-RER[i])]
    donor = fused_top[1] if fused_top[1] not in hs and fused_top[1] not in tws else fused_top[2]
    dws = set(w for w in toks(TEXTS[donor]) if w not in toks(r["question"]) and w not in STOP)
    evw = set()
    for j in ev[:1]:
        evw |= set(toks(TEXTS[j]))
    now_ = set()
    for j in no:
        now_ |= set(toks(TEXTS[j]))
    if not dws:
        continue
    if len(dws & evw) > len(dws & now_):
        win3 += 1
    elif len(dws & now_) > len(dws & evw):
        lose3 += 1
print("  池内次名句独有词代理: 胜率%.0f%%" % (100 * win3 / max(1, win3 + lose3)), flush=True)
print("WORD2_DONE", flush=True)
