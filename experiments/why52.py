# -*- coding: utf-8 -*-
"""why52.py — 桶一未覆盖52道的逐题归因
对每道未覆盖题: 找最佳原子句, 记录 (最大重叠%, 是否因切分残缺/词形/过滤/真缺失)
输出: 归因分布 + 修复方案依据
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
def toks(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

# 重建原子句(无动词过滤版——检查"被过滤"假设用全量切分)
atoms_all = []   # (text, sess, filtered_out_by_verb)
atoms_kept = []
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
        has_verb = bool(re.search(r"\b(is|was|are|were|have|has|had|do|does|did|will|would|can|could|went|go|make|made|take|took|get|got|play|played|read|reads|started|finished|work|works|worked|live|lives|lived|like|likes|loves|adopted|won|signed|joined|planning|plan|bought|sold|graduated|studying|study|moving|moved)\b", p, re.I))
        atoms_all.append((p, sess, has_verb))
        if has_verb:
            atoms_kept.append((p, sess))
print("切分总量: 全部%d 动词过滤后%d (过滤率%.0f%%)" % (len(atoms_all), len(atoms_kept), 100 * (1 - len(atoms_kept) / len(atoms_all))), flush=True)

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

# 逐题: 在"全量切分"(含被过滤的)里找最佳重叠
from collections import Counter
reasons = Counter()
examples = []
for r in bucket1:
    q = qmap.get(r["qa_id"])
    sid = "loco-" + str(q.get("sample_id")) if q else ""
    best_ov = 0
    best_txt = ""
    best_filtered = False
    et_set = set()
    for e in q.get("evidence_messages") or []:
        et_set |= toks(e.get("text") or "")
    # 在该会话全部切分片段里找最大重叠(不限动词过滤, 不限top25位置)
    for p, sess2, fv in atoms_all:
        if sess2 != sid:
            continue
        at = toks(p)
        if not at:
            continue
        ov = len(at & et_set) / max(1, min(len(at), len(et_set)))
        if ov > best_ov:
            best_ov = ov
            best_txt = p
            best_filtered = (not fv)
    # 归因
    if best_ov >= 0.5:
        if best_filtered:
            reasons["被动词过滤器误杀(修复:放宽过滤)"] += 1
        else:
            reasons["词形/词序差异(重叠0.4~0.5边缘)"] += 1
    elif best_ov >= 0.3:
        reasons["部分覆盖(答案跨句,切分残缺)"] += 1
    else:
        reasons["真缺失(原文无≥30%重叠片段)"] += 1
    if len(examples) < 8:
        examples.append((best_ov, r["question"][:50], best_txt[:90]))

print()
print("== 52道未覆盖归因 ==", flush=True)
for k, v in reasons.most_common():
    print("  %s: %d" % (k, v), flush=True)
print()
for ov, q, t in examples:
    print("  [最大重叠%.0f%%] %s" % (ov * 100, q), flush=True)
    print("     最佳片段: %s" % t, flush=True)
print("WHY52_DONE", flush=True)
