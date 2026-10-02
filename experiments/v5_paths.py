# -*- coding: utf-8 -*-
"""v5_paths.py — 三条翻案路 + catch主统计读取(全离线零GLM)
P1 伪fact切原文(不切摘要): 原文原子事实 → 对桶一147道的召回
P2 题目宾语词锚定: 题目宾语词的名词近邻在候选句中找同类别词 → 枚举题判别
P3 时间题25道逐题诊断: 错因四分类
P0 catch主统计读取(catch_align.log已有部分则直接读)
"""
import io, json, os, sys, re
import time
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
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

# P0: catch主统计
print("== P0 catch对齐判别(已算部分) ==", flush=True)
if os.path.exists(HERE + "/catch_align.log"):
    for l in open(HERE + "/catch_align.log", encoding="utf-8", errors="ignore"):
        if re.search(r"AUC|胜率", l):
            print("  " + l.strip()[:110], flush=True)

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
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
R37 = {r["qa_id"]: r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8"))}
def is_dk(p):
    if not p:
        return True
    p = str(p).strip().lower()
    for kw in ["not sure", "don't know", "dont know", "unknown", "no information",
               "not mentioned", "not specified", "cannot", "无法", "不知道", "未提及"]:
        if kw in p:
            return True
    return len(p) <= 2

# ===== P1: 伪fact切原文 =====
print("== P1 伪fact切原文(英文原文原子句) → 桶一147道召回 ==", flush=True)
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
atoms = []
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") != "raw":
        continue
    s = (r.get("raw") or "").strip()
    sess = r.get("session_id") or ""
    if s.startswith("[Session"):
        continue
    # 英文原文按句号/叹号切, 保留含动词+≥4词的句子
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if not re.search(r"\b(is|was|are|were|have|has|had|do|does|did|will|would|can|could|went|go|moved|moved|make|made|take|took|get|got|play|played|read|reads|started|finished|work|works|worked|live|lives|lived|like|likes|loves|adopted|won|signed|joined|planning|plan|bought|sold|graduated|studying|study)\b", p, re.I):
            continue
        atoms.append((p, sess))
print("  原文原子句:", len(atoms), flush=True)
t0 = time.time()
AVEC = emb([a[0] for a in atoms])
print("  embedded %.0fs" % (time.time() - t0), flush=True)
sess_atoms = {}
for idx, (txt, sess) in enumerate(atoms):
    sess_atoms.setdefault(sess, []).append(idx)

bucket1 = [r for r in R37.values() if r.get("llm_score") != 1 and is_dk(r.get("predicted_answer"))]
hit25 = hit10 = hit5 = 0
for r in bucket1:
    q = qmap.get(r["qa_id"])
    sid = "loco-" + str(q.get("sample_id")) if q else ""
    idxs = sess_atoms.get(sid, [])
    if not idxs:
        continue
    qv = emb([r["question"]])[0]
    sims = AVEC[idxs] @ qv
    order = np.argsort(-sims)[:25]
    def etoks(s2):
        return toks(s2)
    found = False
    for rk, oi in enumerate(order):
        at_t = toks(atoms[idxs[oi]][0])
        for e in q.get("evidence_messages") or []:
            et = toks(e.get("text") or "")
            if not et:
                continue
            ov = len(at_t & et) / max(1, min(len(at_t), len(et)))
            if ov >= 0.5:
                found = True
                if rk < 5:
                    hit5 += 1
                if rk < 10:
                    hit10 += 1
                break
        if found:
            break
    if found:
        hit25 += 1
print("  原文原子句替身覆盖证据: %d/%d (%.0f%%) | top5:%d top10:%d" % (
    hit25, len(bucket1), 100 * hit25 / max(1, len(bucket1)), hit5, hit10), flush=True)

# ===== P2: 题目宾语词锚定(枚举题) =====
print("== P2 题目宾语词锚定 → 枚举题证据vs噪声 ==", flush=True)
WIN = LOSE = 0
for i, r in enumerate(rows):
    if r.get("llm_score") is None:
        continue
    ans = str(r["answer"][0])
    if ("," not in ans) and (" and " not in ans.lower()):
        continue
    q = r["question"]
    # 题目宾语词 = 问题中非首词的名词性内容词(排除助词/疑问词/人名)
    qwords = [w for w in toks(q) if w not in STOP2] if False else [w for w in re.findall(r"[a-z']+", q.lower()) if len(w) > 3]
    # 候选句
    hs = targets.get(i)
    if not hs:
        continue
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    z1 = np.load(HERE + "/rerank_stage1.npz") if False else None
# 重写: 用全局缓存
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
for i in targets:
    r = rows[i]
    ans = str(r["answer"][0])
    if ("," not in ans) and (" and " not in ans.lower()):
        continue
    q = r["question"]
    qw = [w for w in re.findall(r"[a-z']+", q.lower()) if len(w) > 3 and w not in ("what", "which", "where", "when", "does", "did", "caroline", "melanie")]
    if not qw:
        continue
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    # 题目宾语词在句中的覆盖(词干近似: 前4字符)
    def cov(sentence):
        st = set(t2[:4] for t2 in re.findall(r"[a-z']+", sentence.lower()))
        return sum(1 for w in qw if w[:4] in st)
    ce, cn = cov(TEXTS[ev[0]]), cov(TEXTS[no[0]])
    if ce > cn:
        WIN += 1
    elif cn > ce:
        LOSE += 1
print("  题目宾语词覆盖竞争: 证据胜%d 噪声胜%d 胜率=%.0f%%" % (
    WIN, LOSE, 100 * WIN / max(1, WIN + LOSE)), flush=True)

# ===== P3: 时间题25道逐题诊断 =====
print("== P3 时间题错题逐题诊断 ==", flush=True)
import collections
diag = collections.Counter()
for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8")):
    if r.get("llm_score") == 1 or str(r.get("category")) != "3":
        continue
    pred = str(r.get("predicted_answer") or "")
    gold = " ".join(str(x) for x in (r.get("answer") or []))
    if is_dk(pred):
        diag["检索失败(不知道)"] += 1
    elif re.search(r"20\d\d", pred) and re.search(r"20\d\d", gold):
        # 都有日期但不同 → 锚定/算术
        diag["日期错(锚定或算术)"] += 1
    elif re.search(r"(week|month|before|after|ago)", gold, re.I) and not re.search(r"(week|month|before|after|ago)", pred, re.I):
        diag["区间口径(pred单日)"] += 1
    else:
        diag["其他"] += 1
for k, v in diag.most_common():
    print("  %s: %d" % (k, v), flush=True)
print("V5PATHS_DONE", flush=True)
