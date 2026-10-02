# -*- coding: utf-8 -*-
"""check_revpass.py — ①反向通行率(证据词被问题词命中) ②GLM读取失效率复核(推论C/D)
全离线零GLM
"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8"))]
def toks(s):
    STOP = set("a an the is are was were be been being to of in on at for with about from by as and or but if so not no that this these those it its i you he she they we my your his her their what when where who how why did do does did s t re ve ll d m".split())
    return [w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP and len(w) > 2]

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

# ① 反向通行率: 证据内容词(词干化)被问题内容词命中
rev_hits, rev_tot, fwd_hits, fwd_tot = 0, 0, 0, 0
low = []
for r in rows:
    evs = r.get("evidence_messages") or []
    if not evs:
        continue
    qt = set(stem(w) for w in toks(r.get("question")))
    for e in evs:
        et = [stem(w) for w in toks(e.get("text"))]
        for w in et:
            rev_tot += 1
            if w in qt:
                rev_hits += 1
    qwords_all = [stem(w) for w in toks(r.get("question"))]
    et_all = set()
    for e in evs:
        et_all |= set(stem(w) for w in toks(e.get("text")))
    for w in qwords_all:
        fwd_tot += 1
        if w in et_all:
            fwd_hits += 1
print("①反向通行率(证据词∈问题): %.1f%%  [正向(问题词∈证据): %.1f%%]" % (
    100 * rev_hits / max(1, rev_tot), 100 * fwd_hits / max(1, fwd_tot)))

# ② 读取失效率复核: r37错题中证据在窗(前25)的比例 → 读出率
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    rr = json.loads(l)
    MID.append(rr["mid"]); KIND.append(rr.get("kind") or "summary")
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
MID2I = {m: i for i, m in enumerate(MID)}
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
inwin = miss_inwin = miss_tot = 0
for r in rows:
    q = qmap.get(r["qa_id"])
    if not q:
        continue
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    hits = set()
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if not hits:
        continue
    qi = list(qmap).index(r["qa_id"]) if False else None
# 重算: 直接用target顺序
targets = {}
qids = list(qmap.keys())
for idx, r in enumerate(rows):
    q = qmap.get(r["qa_id"])
    if not q:
        continue
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    hits = set()
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits and idx < len(TOP50):
        targets[idx] = (r, hits)
for idx, (r, hs) in targets.items():
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    inwin += 1
    if r.get("llm_score") != 1:
        miss_tot += 1
        top = TOP50[idx][np.argsort(-RER[idx])][:25]
        if (hs | tws) & set(top.tolist()):
            miss_inwin += 1
print("②r37可匹配证据题: %d | 错题%d 其中证据在窗%d (%.0f%%)" % (
    inwin, miss_tot, miss_inwin, 100 * miss_inwin / max(1, miss_tot)))
print("  → 排序层剩余责任 = %.1f%%错题 | 读取层责任 = %.1f%%错题" % (
    100 * (miss_tot - miss_inwin) / max(1, miss_tot), 100 * miss_inwin / max(1, miss_tot)))
