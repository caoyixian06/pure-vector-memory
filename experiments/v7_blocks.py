# -*- coding: utf-8 -*-
"""v7_blocks.py — ①②③验证包(全离线零GLM)
B1 邻域块单元: 证据句±1邻域块的词集合覆盖 vs 单句覆盖(50错题)
B2 时序锚: 同session过滤对证据入窗的影响
B3 实体桥: 共现边扩散一轮, 对证据入窗的影响
对照: v2原子句现状(cos∪cov并集=82%)
指标: 50错题证据入窗率(top25) + 新增噪声条数
"""
import io, json, os, sys, re, random
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
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_list(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if len(w) > 2]
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
COUR = re.compile(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank|cool|nice)\b", re.I)

# 原子句库(v2切分+邻域索引)
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
    if s.startswith("[Session"):
        continue
    sess = r.get("session_id") or ""
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if len(words) <= 5 and COUR.search(p):
            continue
        atoms.append((p, sess, r.get("memory_id")))
A2VEC = emb([a[0] for a in atoms])
A2TOK = [toks_set(a[0]) for a in atoms]
A2SESS = [a[1] for a in atoms]
A2HOST = [a[2] for a in atoms]
sess2idx = {}
for k2, a in enumerate(atoms):
    sess2idx.setdefault(a[1], []).append(k2)
# 邻域索引: 同memory_id的相邻原子句(同turn切出来的就是邻居) + 同session相邻turn
NEIGH = [set() for _ in atoms]
for k2, a in enumerate(atoms):
    NEIGH[k2].add(k2)
print("原子句:", len(atoms), flush=True)

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
bucket1 = [r for r in R37.values() if r.get("llm_score") != 1 and is_dk(r.get("predicted_answer"))]
print("桶一:", len(bucket1), flush=True)

# 块词集合: 原子句 + 同host的兄弟原子句(同turn切出的其他句) + 邻接host的词
host2atoms = {}
for k2, a in enumerate(atoms):
    host2atoms.setdefault(a[2], []).append(k2)
mid_keys = list(host2atoms.keys())
# 邻接host(同session相邻turn): 按mid序
mid_sorted = sorted(host2atoms.keys())
adj_host = {}
for a, b in zip(mid_sorted, mid_sorted[1:]):
    adj_host.setdefault(a, []).append(b)
    adj_host.setdefault(b, []).append(a)

def block_toks(k2, use_block=True):
    if not use_block:
        return A2TOK[k2]
    out = set(A2TOK[k2])
    host = atoms[k2][2]
    for sib in host2atoms.get(host, []):
        if sib != k2:
            out |= A2TOK[sib]
    for nh in adj_host.get(host, [])[:1]:
        for sib in host2atoms.get(nh, []):
            out |= A2TOK[sib]
    return out

qmap_keys = qmap
def recall_eval(use_block, use_time, use_bridge, tag):
    hit = 0
    noise_cnt = 0
    for r in bucket1:
        q = qmap_keys.get(r["qa_id"])
        sid = "loco-" + str(q.get("sample_id")) if q else ""
        idxs = sess2idx.get(sid, [])
        if not idxs:
            continue
        qv = A2VEC[idxs] @ emb([r["question"]])[0]
        qtok = toks_set(r["question"])
        et_set = set()
        for e in q.get("evidence_messages") or []:
            et_set |= toks_set(e.get("text") or "")
        # 综合: cos + 覆盖
        cov = np.array([len(A2TOK[idxs[k]] & qtok) for k in range(len(idxs))], dtype=np.float32)
        score = qv * 0.5 + cov * 0.5
        order = np.argsort(-score)[:25]
        inwin_tokens = set()
        sel_hosts = set()
        for oi in order:
            k = idxs[oi]
            inwin_tokens |= (block_toks(k, use_block) if use_block else A2TOK[k])
            sel_hosts.add(atoms[k][2])
        if use_bridge:
            # 实体桥: 选中句的实体词(大写词)在其他原子句中的共现扩散一轮
            ents = set(re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(TEXTS[MID2I.get(h2, "")] or "" for h2 in sel_hosts if h2 in MID2I)))
            for k2, a in enumerate(atoms):
                if a[1] != sid or k2 in [idxs[x] for x in order]:
                    continue
                t = a[0]
                if any(e2 in t for e2 in ents):
                    inwin_tokens |= A2TOK[k2]
        found = False
        for e in q.get("evidence_messages") or []:
            et = toks_set(e.get("text") or "")
            if et and len(inwin_tokens & et) / max(1, len(et)) >= 0.5:
                found = True
                break
        if found:
            hit += 1
        noise_cnt += max(0, 25 - 0)
    print("  %-30s 召回 %d/%d (%.0f%%)" % (tag, hit, len(bucket1), 100 * hit / len(bucket1)), flush=True)
    return hit

print("== ①②③验证(桶一147道) ==", flush=True)
recall_eval(False, False, False, "对照: v2单句(cos+覆盖)")
recall_eval(True, False, False, "B1: +邻域块")
recall_eval(True, True, False, "B1+B2: +邻域块+时序(同session已隐含)")
recall_eval(True, True, True, "B1+B2+B3: 全开(含实体桥)")
print("V7BLOCKS_DONE", flush=True)
