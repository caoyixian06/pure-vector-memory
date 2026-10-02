# -*- coding: utf-8 -*-
"""diag_fusion.py — 四通道融合检索实验(用户设计: 问题词/文本向量 × 库256/1024向量加权组合)
通道:
  T_bge : 问题1024文本 × 库1024文本(现行主链)
  T_qwen: 问题256文本 × 库256文本(记录自带vector字段)
  W_ex  : 问题词→词字典精确命中→宿主计票(软权重)
  W_soft: 问题词256向量 × 词字典26827×256 软配对(cos≥0.75 top3)→宿主计票
组合: score = z(T_bge) + a*z(T_qwen) + b*z(W) , 外科手术u2开关
目标: 找让金证据排名最高的组合(miss进25/对题进5/均名次)
PART A: 无关句 vs 证据句的本底对比(证据在库内余弦分布的百分位)
"""
import io, json, os, sys, re, time, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def emb_qwen(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        for att in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r2:
                    out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
    return l2n(np.concatenate(out))

# ---- 库 ----
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
        rr = json.loads(l)
        REC[rr.get("memory_id")] = rr
    except Exception:
        pass
QW = np.zeros((N, 256), dtype=np.float32)
miss_vec = 0
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
    else:
        miss_vec += 1
QW = l2n(QW)
print("lib", N, "qwen256缺向量记录:", miss_vec, flush=True)

bQ = emb_bge(Q)
print("bge q done", flush=True)
wQ = emb_qwen(Q)
print("qwen q done", flush=True)

# ---- 证据定位 ----
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
targets = []
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
        targets.append((i, hits))
print("matched:", len(targets), flush=True)
tidx = {i: h for i, h in targets}
ok = np.array([rows[i].get("llm_score") == 1 for i, _ in targets])

# ---- 通道打分(逐题算, 存排名用) ----
STOP = set("a an the is are was were be been being do does did have has had i you he she it we they me him her us them my your his its our their what when where who whom why how which that this these those there to of in on at for with about from by as and or but if so not no did does do s t re ve ll d m".split())
WV = np.load(HERE + "/word_vecs.npz")["V"].astype(np.float32)
WV = l2n(WV)
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
# 清洗词字典键(去jieba粘连的标点)
CLEAN = {}
for w in WH.keys():
    c = w.strip(":").lower()
    if c and c not in CLEAN:
        CLEAN[c] = w
print("词字典清洗后可查键:", len(CLEAN), flush=True)

def qwords(q):
    return [w for w in re.findall(r"[a-z']+", q.lower()) if w not in STOP and len(w) > 1]

# 问题词的256向量(软配对用)
allw = sorted({w for q in Q for w in qwords(q)})
print("unique q words:", len(allw), flush=True)
WV_q = emb_qwen(allw)
WQ_VEC = dict(zip(allw, WV_q))

def word_vote_soft(q, tau=0.75, topk=3):
    vote = np.zeros(N, dtype=np.float32)
    for w in qwords(q):
        v = WQ_VEC.get(w)
        if v is None:
            continue
        cs = WV @ v
        top = np.argsort(-cs)[:topk]
        for ti in top:
            if cs[ti] < tau:
                continue
            hosts = WH.get(WKEYS[ti])
            if not hosts:
                continue
            wt = float(cs[ti])
            for h in hosts:
                j = MID2I.get(h)
                if j is not None:
                    vote[j] += wt
    return vote

WKEYS = list(WH.keys())

def word_vote_exact(q):
    vote = np.zeros(N, dtype=np.float32)
    for w in qwords(q):
        key = CLEAN.get(w)
        if not key:
            continue
        for h in WH[key]:
            j = MID2I.get(h)
            if j is not None:
                vote[j] += 1.0
    return vote

# 词命中率统计
hit = sum(1 for q in Q for w in qwords(q) if CLEAN.get(w))
tot = sum(len(qwords(q)) for q in Q)
print("问题词精确命中率: %.1f%%" % (100 * hit / max(tot, 1)), flush=True)

def zs(x):
    mu, sd = x.mean(), x.std() + 1e-9
    return (x - mu) / sd

def eval_scores(SC, tag):
    """SC: (n, N) 逐题得分 → 排名指标"""
    m25 = o5 = 0
    rsum = 0
    cnt = 0
    for i, hits in targets:
        s = SC[i]
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        rsum += rk
        cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-26s 错题进25 %3d/%d  对题进5 %3d  均名次%.0f" % (tag, m25, int((~ok).sum()), o5, rsum / cnt), flush=True)
    return m25, o5, rsum / cnt

print("== PART A 本底: 证据句在库内余弦分布的位置 ==", flush=True)
pct = []
beaten = []
for i, hits in targets:
    s = D @ bQ[i]
    ev = max(s[h] for h in hits)
    pct.append(100.0 * (s > ev).mean())
    beaten.append(int((s > ev).sum()))
pct = np.array(pct)
print("金证据的余弦百分位: 中位%.0f%% 均值%.0f%% ; 平均被%d条记录压过" % (
    np.median(pct), pct.mean(), np.mean(beaten)), flush=True)

print("== PART B 通道单独表现 ==", flush=True)
SB = (bQ @ D.T).astype(np.float32)
eval_scores(SB, "T_bge 纯1024(基线)")
SQ = (wQ @ QW.T).astype(np.float32)
eval_scores(SQ, "T_qwen 纯256文本")

# 词通道逐题算(慢部分)
VEX = np.zeros((n, N), dtype=np.float32)
VSO = np.zeros((n, N), dtype=np.float32)
t0 = time.time()
for i in range(n):
    VEX[i] = word_vote_exact(Q[i])
    VSO[i] = word_vote_soft(Q[i])
print("word votes done", round(time.time() - t0, 1), "s", flush=True)
eval_scores(VEX, "W_ex 精确词宿主票")
eval_scores(VSO, "W_soft 软配对宿主票")

# 手术方向u2
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
bQ_s = l2n(bQ + u2)
SBS = (bQ_s @ D.T).astype(np.float32)

print("== PART C 组合搜索 ==", flush=True)
ZB, ZQ_, ZVX, ZVS = [], [], [], []
for i in range(n):
    ZB.append(zs(SB[i])); ZQ_.append(zs(SQ[i])); ZVX.append(zs(VEX[i])); ZVS.append(zs(VSO[i]))
best = []
for tag_surg, SBuse in (("原查询", ZB), ("手术u2", [zs(x) for x in SBS])):
    for a in (0.0, 0.5, 1.0):
        for b in (0.0, 0.5, 1.0):
            for wtag, ZV in (("ex", ZVX), ("soft", ZVS)):
                if a == 0 and b == 0:
                    continue
                SC = [SBuse[i] + a * ZQ_[i] + b * ZV[i] for i in range(n)]
                m25, o5, mr = eval_scores(SC, "%s a=%.1f b=%.1f %s" % (tag_surg, a, b, wtag))
                best.append((m25 + o5, m25, o5, mr, tag_surg, a, b, wtag))
best.sort(reverse=True)
print("TOP3:", best[:3], flush=True)
np.savez_compressed(HERE + "/fusion_cache.npz", SB=SB, SQ=SQ, VEX=VEX, VSO=VSO)
print("SAVED fusion_cache.npz", flush=True)
