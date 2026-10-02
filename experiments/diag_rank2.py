# -*- coding: utf-8 -*-
"""diag_rank2.py — 拆解余弦: 为什么证据排不进前5/前25
针对错题里 rank>25 的题,解剖:
A. 问题内容词在 top1 vs 证据句 的覆盖率(词面重叠差)
B. top5/top25 座位构成: 问句/寒暄反应/同话题陈述, 含人名比例, 同会话比例
C. 竞争池: 证据所在会话里含问题人名的记录数
D. 同turn的摘要孪生(twin)排名 vs 原句排名
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
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
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
        r = json.loads(l)
    except Exception:
        continue
    REC[r.get("memory_id")] = r

def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        v = r.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(r, ensure_ascii=False)

TEXTS = [rec_text(REC.get(m, {})) for m in MID]

STOP = set("a an the is are was were be been being do does did have has had "
           "i you he she it we they me him her us them my your his its our their "
           "what when where who whom why how which that this these those there "
           "to of in on at for with about from by as and or but if so not no "
           "did does do s t re ve ll d m".split())

def toks(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if w not in STOP and len(w) > 1]

WSET = [set(toks(t)) for t in TEXTS]

COURTESY = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)

def ev_text(e):
    for f in ("text", "message", "message_text"):
        v = e.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

rows = json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
qs_miss = [r for r in rows if r.get("llm_score") != 1]

def find_ev(row):
    sid = row.get("sample_id") or ("conv-" + str(row.get("conversation_idx")))
    prefix = "loco-" + sid + "_"
    cand = [i for i in range(N) if MID[i].startswith(prefix)]
    hits = []
    for e in row.get("evidence_messages") or []:
        t = normsub(ev_text(e))
        if len(t) < 15:
            continue
        for i in cand:
            if KIND[i] == "raw" and t[:40] in normsub(TEXTS[i]):
                hits.append(i)
                break
    return hits, prefix, cand

Q = []
miss_qs = [r["question"] for r in qs_miss]
for s in range(0, len(miss_qs), 64):
    enc = model.encode(miss_qs[s:s + 64])
    Q.append(np.asarray(enc["dense_vecs"], dtype=np.float32))
Q = np.concatenate(Q)
Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
S = Q @ D.T

# ---- 统计容器 ----
n_gt25 = 0
ov_top1, ov_ev = [], []
seat_q = seat_cour = seat_stmt = seat_same_conv = seat_name = 0
top5_total = 0
name_cnt_top25 = []
pool_sizes = []
twin_in_lib = twin_better = twin_le25 = twin_le5 = twin_total = 0
examples = []

CONV_IDX = {}
for i, m in enumerate(MID):
    CONV_IDX.setdefault(m.rsplit("_m", 1)[0], []).append(i)

for j, r in enumerate(qs_miss):
    hits, prefix, cand = find_ev(r)
    if not hits:
        continue
    srow = S[j]
    order = np.argsort(-srow)
    pos = {i: p for p, i in enumerate(order)}
    best = min(hits, key=lambda i: pos[i])
    rk = pos[best] + 1
    if rk <= 25:
        continue
    n_gt25 += 1
    qtoks = set(toks(r["question"]))
    # A. 词面重叠
    if qtoks:
        ov_top1.append(len(qtoks & WSET[order[0]]) / len(qtoks))
        ov_ev.append(len(qtoks & WSET[best]) / len(qtoks))
    # 人名(说话人)
    names = set()
    for f in ("speaker_a", "speaker_b"):
        if r.get(f):
            names.add(str(r[f]).lower())
    # B. top5 座位构成
    top5 = list(order[:5])
    top5_total += len(top5)
    for k in top5:
        t = TEXTS[k]
        if t.rstrip().endswith("?"):
            seat_q += 1
        elif COURTESY.search(t):
            seat_cour += 1
        else:
            seat_stmt += 1
        if MID[k].startswith(prefix):
            seat_same_conv += 1
        if any(n in TEXTS[k].lower() for n in names):
            seat_name += 1
    top25 = list(order[:25])
    name_cnt_top25.append(sum(1 for k in top25 if any(n in TEXTS[k].lower() for n in names)))
    # C. 竞争池
    pool = CONV_IDX.get(prefix, [])
    pool_sizes.append(sum(1 for i in pool if any(n in TEXTS[i].lower() for n in names)))
    # D. twin 摘要
    rec = REC.get(MID[best], {})
    tw = rec.get("raw_of")
    if tw and tw in MID2I:
        ti = MID2I[tw]
        twin_total += 1
        trk = pos[ti] + 1
        if trk < rk:
            twin_better += 1
        if trk <= 25:
            twin_le25 += 1
        if trk <= 5:
            twin_le5 += 1
    # 例子: 证据缺了问题的哪些词
    if len(examples) < 3 and qtoks:
        missing = sorted(qtoks - WSET[best])
        present = sorted(qtoks & WSET[best])
        examples.append((r["question"][:80], missing[:8], present[:8], TEXTS[best][:100], TEXTS[order[0]][:80]))

print("rank>25 错题数:", n_gt25)
print("A. 问题内容词覆盖率: top1=%.1f%%  证据句=%.1f%% (差%.1f个百分点)" % (
    100 * np.mean(ov_top1), 100 * np.mean(ov_ev), 100 * (np.mean(ov_top1) - np.mean(ov_ev))))
print("B. top5座位构成: 问句%d%% 寒暄/情绪反应%d%% 其他陈述%d%% ; 同会话%d%% ; 含问题人名%d%%" % (
    100 * seat_q / top5_total, 100 * seat_cour / top5_total, 100 * seat_stmt / top5_total,
    100 * seat_same_conv / top5_total, 100 * seat_name / top5_total))
print("   top25里含问题人名的记录数: 中位%.0f 条 (即人名就占掉一半座位)" % np.median(name_cnt_top25))
print("C. 证据同会话内含该人名的记录(竞争池): 中位%.0f 条" % np.median(pool_sizes))
print("D. 证据turn的摘要twin: 可查%d | 比原句排名高%d%% | twin进top25 %d%% | twin进top5 %d%%" % (
    twin_total, 100 * twin_better // max(twin_total, 1), 100 * twin_le25 // max(twin_total, 1), 100 * twin_le5 // max(twin_total, 1)))
print()
for q, miss, pres, evt, top1t in examples:
    print("例:", q)
    print("   证据句缺问题词:", miss, "| 有的词:", pres)
    print("   证据句:", evt)
    print("   第1名:", top1t)
