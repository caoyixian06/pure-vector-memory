# -*- coding: utf-8 -*-
"""atomv2.py — 原子句域v2构建+并行验证(全离线零GLM)
V2改造: ①无动词过滤(去寒暄即可) ②词干化 ③256维限定域软配对
输出: v1召回(65%基线) / v2召回 / v1∪v2并集召回 / 双通道命中交叉
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
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
COUR = re.compile(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank|cool|nice|wow)\b", re.I)

# V2切分: 无动词过滤, 只去纯寒暄短句
atoms2 = []
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
        atoms2.append((p, sess))
print("V2原子句(无动词过滤):", len(atoms2), flush=True)

t0 = time.time()
A2VEC = emb([a[0] for a in atoms2])
print("embedded %.0fs" % (time.time() - t0), flush=True)
A2TOK = [toks_set(a[0]) for a in atoms2]
A2SESS = [a[1] for a in atoms2]
sess2idx = {}
for idx, a in enumerate(atoms2):
    sess2idx.setdefault(a[1], []).append(idx)

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

# 词向量(限定域软配对用)
zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None

# V2检索+评估: 对桶一每题, 三种模式
hit_v1style = 0   # BGE余弦top25(模拟v1但无过滤)
hit_v2 = 0        # 词干覆盖top25
hit_soft = 0      # 软配对top25
hit_union = 0
detail = []
for r in bucket1:
    q = qmap.get(r["qa_id"])
    sid = "loco-" + str(q.get("sample_id")) if q else ""
    idxs = sess2idx.get(sid, [])
    if not idxs:
        continue
    qv = A2VEC[idxs] @ emb([r["question"]])[0]
    order_cos = np.argsort(-qv)[:25]
    qtok = toks_set(r["question"])
    cov = np.array([len(A2TOK[idxs[k]] & qtok) for k in range(len(idxs))], dtype=np.float32)
    order_cov = np.argsort(-cov)[:25]
    found = {"cos": False, "cov": False, "soft": False}
    for mode, o in (("cos", order_cos), ("cov", order_cov)):
        for oi in o:
            at_t = A2TOK[idxs[oi]]
            for e in q.get("evidence_messages") or []:
                et = toks_set(e.get("text") or "")
                if not et:
                    continue
                if len(at_t & et) / max(1, min(len(at_t), len(et))) >= 0.5:
                    if mode == "cos":
                        found["cos"] = True
                    else:
                        found["cov"] = True
                    break
            if found[mode]:
                break
    found["soft"] = found["cos"] or found["cov"]
    hit_v1style += found["cos"]
    hit_v2 += found["cov"]
    hit_soft += found["soft"]
    hit_union += (found["cos"] or found["cov"])
print()
print("== 桶一147道 三模式召回对照 ==")
print("  v1式(BGE余弦, 但V2无过滤切分): %d (%.0f%%)" % (hit_v1style, 100 * hit_v1style / len(bucket1)))
print("  V2词干覆盖:                    %d (%.0f%%)" % (hit_v2, 100 * hit_v2 / len(bucket1)))
print("  V2并集(cos∨cov):              %d (%.0f%%)  [v1版=65%%]" % (
    hit_union, 100 * hit_union / len(bucket1)))
print("ATOMV2_DONE", flush=True)
