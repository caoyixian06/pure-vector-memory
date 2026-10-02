# -*- coding: utf-8 -*-
"""diag_dims.py — 维度级解剖: 问题/答案/证据向量的活跃维度分工与规律
PART1 BGE全量+qwen400样本嵌入
PART2 规律: 活跃维度重叠 / 问题指纹vs答案指纹 / delta方向一致性 / 槽位指纹(日期/人名)
PART3 纯规则手术测试: 问题向量向答案质心方向平移 α, 看320道rank>25的证据能捞回几道
"""
import io, json, os, sys, time, random, re, urllib.request
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
A = [str(r["answer"][0]) for r in rows]
EV = [r["evidence_messages"][0]["text"] for r in rows]
n = len(rows)
rng = random.Random(7)
perm = list(range(n)); rng.shuffle(perm)
EVOTH = [EV[perm[i]] for i in range(n)]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def embed_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

def embed_qwen(texts):
    out = []
    for s in range(0, len(texts), 32):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 32], "dimensions": 256}).encode()
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

print("PART1 embed", flush=True)
bQ, bA, bEV, bEO = embed_bge(Q), embed_bge(A), embed_bge(EV), embed_bge(EVOTH)
sub = 400
wQ, wA, wEV = embed_qwen(Q[:sub]), embed_qwen(A[:sub]), embed_qwen(EV[:sub])

def topk(v, k=30):
    return set(np.argsort(-np.abs(v))[:k].tolist())

def analyze(ZQ, ZA, ZEV, ZEO, tag, k=30):
    # Z 为同一套统计量 z-score 后的矩阵
    o_qa = [len(topk(ZQ[i], k) & topk(ZA[i], k)) for i in range(len(ZQ))]
    o_qe = [len(topk(ZQ[i], k) & topk(ZEV[i], k)) for i in range(len(ZQ))]
    o_qo = [len(topk(ZQ[i], k) & topk(ZEO[i], k)) for i in range(len(ZQ))]
    print("[%s] 活跃top%d维重叠: 问题∩答案=%.1f 问题∩证据=%.1f 问题∩随机=%.1f (随机期望=%.1f)" % (
        tag, k, np.mean(o_qa), np.mean(o_qe), np.mean(o_qo), k * k / ZQ.shape[1]))
    # 指纹: 各类质心的活跃维
    cq, ca = np.abs(ZQ.mean(0)), np.abs(ZA.mean(0))
    fq, fa = topk(cq, 50), topk(ca, 50)
    print("[%s] 问题指纹50维 ∩ 答案指纹50维 = %d 个; 质心余弦=%.3f" % (
        tag, len(fq & fa), float(np.dot(l2n(ZQ.mean(0)[None])[0], l2n(ZA.mean(0)[None])[0]))))
    return fq, fa

print("PART2a BGE", flush=True)
poolb = np.concatenate([bQ, bA, bEV])
mu, sd = poolb.mean(0), poolb.std(0) + 1e-9
zb = lambda M: (M - mu) / sd
fq_b, fa_b = analyze(zb(bQ), zb(bA), zb(bEV), zb(bEO), "BGE1024")

print("PART2b qwen", flush=True)
poolw = np.concatenate([wQ, wA, wEV])
muw, sdw = poolw.mean(0), poolw.std(0) + 1e-9
zw = lambda M: (M - muw) / sdw
analyze(zw(wQ), zw(wA), zw(wEV), zw(wEV[:sub]), "qwen256")

# 槽位指纹(BGE): 日期答案 vs 人名
names = sorted({str(r.get("speaker_a")) for r in rows if r.get("speaker_a")} |
               {str(r.get("speaker_b")) for r in rows if r.get("speaker_b")})
nameV = embed_bge(names)
zn = (nameV - mu) / sd
name_prof = set(np.argsort(-np.abs(zn).mean(0))[:50].tolist())
when_idx = [i for i in range(n) if re.match(r"(?i)^when\b", Q[i])]
other_idx = [i for i in range(n) if not re.match(r"(?i)^when\b", Q[i])][:len(when_idx)]
date_prof = set(np.argsort(-np.abs(zb(bA)[when_idx]).mean(0))[:50].tolist())
zA = zb(bA); zQ = zb(bQ); zE = zb(bEV)
def oprof(idxs, prof, src):
    return np.mean([len(topk(src[i], 30) & prof) / 30 for i in idxs])
print("槽位指纹: 日期型答案50维∩问题top30: When题=%.2f 其他题=%.2f" % (
    oprof(when_idx, date_prof, zA), oprof(other_idx, date_prof, zA)))
print("          人名50维∩top30: 问题=%.2f 证据=%.2f 答案=%.2f" % (
    oprof(range(n), name_prof, zQ), oprof(range(n), name_prof, zE), oprof(range(n), name_prof, zA)))

# PART3 手术: q' = normalize(q + a*(ac-qc))
print("PART3 surgery", flush=True)
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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

qc = bQ.mean(0); qc /= np.linalg.norm(qc)
ac = bA.mean(0); ac /= np.linalg.norm(ac)
u = ac - qc

miss = [(i, r) for i, r in enumerate(rows) if r.get("llm_score") != 1]
targets = []
for i, r in miss:
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
    if not hits:
        continue
    s0 = D @ bQ[i]
    order = np.argsort(-s0)
    pos = {x: p for p, x in enumerate(order)}
    rk0 = min(pos[h] for h in hits) + 1
    if rk0 > 25:
        targets.append((i, hits))

print("手术对象(证据rank>25错题):", len(targets), flush=True)
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    got25 = got100 = 0
    ranksum = 0
    for i, hits in targets:
        qp = bQ[i] + alpha * u
        qp /= np.linalg.norm(qp)
        s = D @ qp
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        got25 += rk <= 25
        got100 += rk <= 100
        ranksum += rk
    print("  α=%.2f: 进top25 %d题 进top100 %d题 平均名次%.0f" % (
        alpha, got25, got100, ranksum / len(targets)), flush=True)
