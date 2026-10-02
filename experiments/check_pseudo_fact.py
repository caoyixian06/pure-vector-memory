# -*- coding: utf-8 -*-
"""check_pseudo_fact.py — 孪生摘要二次利用(伪fact域)离线验证
步骤:
①从mem.jsonl的summary字段切原子句(规则: 句号/分号切分, 每句含人名+动词干+≥4词)
②BGE嵌入全部原子句(零LLM)
③对桶一147道"不知道型"错题: 伪fact域检索 vs 现有top50, 看金证据的"原子句替身"能否进窗
④对照: 桶一里摘要确实提到证据内容的比例(上限)
输出: 伪fact可召回的桶一题数 / 147 = 这条路的真实上限
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
print("bge loaded", flush=True)

HERE = "C:/locomo_refined/memsys"
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

# ①收集summary记录
SUMS = []
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") == "raw":
        continue
    s = (r.get("summary") or "").strip()
    if len(s) >= 20:
        SUMS.append(r)
print("summary记录:", len(SUMS), flush=True)

# ②切原子句
STOPW = set("a an the is are was were be been being to of in on at for with about from by as and or but if so that this these those".split())
atoms = []  # (text, mid, session)
for r in SUMS:
    s = r.get("summary") or ""
    sess = r.get("session_id") or ""
    for piece in re.split(r"[。;;.!?\n]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 5 or len(words) > 40:
            continue
        if not re.search(r"\b(is|was|are|were|has|had|have|does|did|do|will|would|can|could|likes|enjoys|works|lives|moved|plans|wants|going|went|made|took|plays|reading|studying|mentions|shares|talks|says|tells)\b", p, re.I):
            continue
        atoms.append((p, r.get("memory_id"), sess))
print("原子句:", len(atoms), flush=True)

# ③嵌入
t0 = time.time()
AVEC = emb([a[0] for a in atoms])
print("embedded %.0fs" % (time.time() - t0), flush=True)

# ④桶一147道错题测试
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

def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

bucket1 = [r for r in R37.values() if r.get("llm_score") != 1 and is_dk(r.get("predicted_answer"))]
print("桶一错题:", len(bucket1), flush=True)

CONVSESS = {}
for a, mid, sess in atoms:
    CONVSESS.setdefault(sess, []).append(len(CONVSESS.get(sess, [])))
# 会话->原子句索引
sess_atoms = {}
for idx, (txt, mid, sess) in enumerate(atoms):
    sess_atoms.setdefault(sess, []).append(idx)

hit25 = hit10 = hit5 = 0
has_ev_atom = 0
detail = []
for r in bucket1:
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    sess = "loco-" + str(sid)
    idxs = sess_atoms.get(sess, [])
    if not idxs:
        continue
    qv = emb([r["question"]])[0]
    sims = AVEC[idxs] @ qv
    order = np.argsort(-sims)[:25]
    # 金证据内容是否被某原子句"覆盖"(词面重叠>=50%)
    def toks(s):
        return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
    found = False
    for rk, oi in enumerate(order):
        at = atoms[idxs[oi]][0]
        at_t = toks(at)
        for e in r.get("evidence_messages") or []:
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
        has_ev_atom += 1
    if len(detail) < 5:
        detail.append((r["question"][:60], found, atoms[idxs[order[0]]][0][:80] if len(order) else ""))

print()
print("=== 伪fact域对桶一的召回代理 ===")
print("原子句替身覆盖证据的题: %d/%d (%.0f%%)" % (hit25, len(bucket1), 100 * hit25 / max(1, len(bucket1))))
print("  其中进top5: %d | top10: %d" % (hit5, hit10))
print()
for q, f, top1 in detail:
    print("[%s] %s" % ("命中" if f else "未中", q))
    print("   伪fact top1:", top1)
print("PSEUDOFACT_DONE", flush=True)
