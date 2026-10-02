# -*- coding: utf-8 -*-
"""diag_enum.py — 枚举聚合题的向量规律解剖
A. 枚举题在查询时可否识别(问题向量质心分类器 vs 词面规则)
B. 金答案的各项是不是"同类兄弟"(项内余弦 vs 跨题随机项余弦)
C. 多条证据记录彼此的相似度 vs 它们与问题的相似度(兄弟扩展是否优于问题检索)
D. 枚举题 vs 单答案题的 q↔a 余弦几何对比
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

def split_items(a):
    a = str(a)
    parts = re.split(r",|\band\b", a)
    return [p.strip(" .\"'") for p in parts if len(p.strip(" .\"'")) >= 2]

def is_enum(r):
    a = str(r["answer"][0])
    return ("," in a) or (" and " in a.lower())

enum_rows = [r for r in rows if is_enum(r)]
single_rows = [r for r in rows if not is_enum(r)]
print("enum题 %d / single题 %d" % (len(enum_rows), len(single_rows)), flush=True)

# ---- A. 查询时识别 ----
QE, QS = emb([r["question"] for r in enum_rows]), emb([r["question"] for r in single_rows])
ce, cs = QE.mean(0), QS.mean(0)
ce /= np.linalg.norm(ce); cs /= np.linalg.norm(cs)
print("A1. 枚举题质心 vs 单答案题质心 cos=%.3f" % float(ce @ cs), flush=True)

def acc_of(QE_, QS_):
    de = QE_ @ ce - QE_ @ cs
    ds = QS_ @ ce - QS_ @ cs
    return (de > 0).mean(), (ds < 0).mean()
ae, asg = acc_of(QE[:len(QE)//2], QS[:len(QS)//2])
ae2, asg2 = acc_of(QE[len(QE)//2:], QS[len(QS)//2:])
print("A2. 质心分类器(后半区验证): 枚举识别率%.1f%% 单答案识别率%.1f%% (前半区: %.1f%%/%.1f%%)" % (
    100*ae2, 100*asg2, 100*ae, 100*asg), flush=True)

KW = re.compile(r"\b(ways|all|both|every|things|times|places|activities|events|books|games|shows|hobbies|foods)\b", re.I)
def kw_hit(r):
    return bool(KW.search(r["question"]))
kh_e = np.mean([kw_hit(r) for r in enum_rows])
kh_s = np.mean([kw_hit(r) for r in single_rows])
print("A3. 词面规则(复数/列举词): 枚举题命中%.1f%% 单答案题误中%.1f%%" % (100*kh_e, 100*kh_s), flush=True)
# 词面+质心联合
def combo(r, qv):
    return kw_hit(r) or (qv @ ce - qv @ cs > 0)
ce_hit = [(kw_hit(r) or (QE[i] @ ce - QE[i] @ cs > 0)) for i, r in enumerate(enum_rows)]
cs_hit = [(kw_hit(r) or (QS[i] @ ce - QS[i] @ cs > 0)) for i, r in enumerate(single_rows)]
print("A4. 联合(或): 枚举%.1f%% 单答案误中%.1f%%" % (100*np.mean(ce_hit), 100*np.mean(cs_hit)), flush=True)

# ---- B. 答案项是同类兄弟? ----
within, across = [], []
all_items = []
for r in enum_rows:
    its = split_items(r["answer"][0])[:6]
    if len(its) >= 2:
        all_items.append(its)
flat = [x for its in all_items for x in its]
IV = emb(flat)
pos = 0
vecs_per_q = []
for its in all_items:
    vs = IV[pos:pos + len(its)]
    pos += len(its)
    vecs_per_q.append(vs)
    for a in range(len(vs)):
        for b in range(a + 1, len(vs)):
            within.append(float(vs[a] @ vs[b]))
rng = np.random.default_rng(3)
n_across = min(2000, len(within) * 2)
for _ in range(n_across):
    q1, q2 = rng.integers(0, len(vecs_per_q)), rng.integers(0, len(vecs_per_q))
    if q1 == q2:
        continue
    v1 = vecs_per_q[q1][rng.integers(0, len(vecs_per_q[q1]))]
    v2 = vecs_per_q[q2][rng.integers(0, len(vecs_per_q[q2]))]
    across.append(float(v1 @ v2))
print("B. 同题答案项间cos: 均值%.3f中位%.3f | 跨题随机项cos: 均值%.3f中位%.3f" % (
    np.mean(within), np.median(within), np.mean(across), np.median(across)), flush=True)

# ---- C. 多证据: 兄弟相似 vs 问题相似 ----
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
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

def find_ev(r):
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
    return hits

QE_all = emb([r["question"] for r in rows])
ev_ev, q_ev2 = [], []
n_multi = 0
for i, r in enumerate(rows):
    if not is_enum(r):
        continue
    hits = find_ev(r)
    if len(hits) >= 2:
        n_multi += 1
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                ev_ev.append(float(D[hits[a]] @ D[hits[b]]))
        for h in hits[1:]:
            q_ev2.append(float(QE_all[i] @ D[h]))
print("C. 枚举题多证据组%d: 证据-证据cos均值%.3f | 问题-第二片证据cos均值%.3f" % (
    n_multi, np.mean(ev_ev) if ev_ev else -1, np.mean(q_ev2) if q_ev2 else -1), flush=True)
print("   (证据-证据 > 问题-证据 → 第二片离第一片更近, 兄弟扩展可行)", flush=True)

# ---- D. q↔a 几何对比 ----
AE = emb([str(r["answer"][0]) for r in enum_rows])
AS = emb([str(r["answer"][0]) for r in single_rows])
qa_e = np.einsum("ij,ij->i", QE, AE)
qa_s = np.einsum("ij,ij->i", QS, AS)
print("D. q↔答案cos: 枚举题均值%.3f | 单答案题均值%.3f" % (qa_e.mean(), qa_s.mean()), flush=True)
# 枚举题答案 vs 单答案答案的字长
le = np.mean([len(str(r["answer"][0])) for r in enum_rows])
ls = np.mean([len(str(r["answer"][0])) for r in single_rows])
print("   答案均长: 枚举%.0f字 vs 单答案%.0f字" % (le, ls), flush=True)
