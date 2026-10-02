# -*- coding: utf-8 -*-
"""foundry70e.py — 解剖: evidence(session_index,message_index) <-> 记录order 映射规律"""
import io, json, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
def P(s):
    print(s)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q

# 记录order字段解析
def get_order(i):
    o = REC.get(MID[i], {}).get("order")
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except Exception:
            o = None
    return o if isinstance(o, list) else None

# 解剖3题: 打印evidence定位 vs flat命中记录的order
cnt = 0
for qa in Q:
    if cnt >= 3:
        break
    q = Q[qa]
    evs = q.get("evidence_messages") or []
    if not evs:
        continue
    P("===== %s =====" % qa)
    for em in evs[:4]:
        txt = em.get("text") or ""
        k = norm(txt)[:60]
        hits = [i for i in range(len(MID)) if len(k) >= 12 and k in RAWN[i]]
        P("  ev dia=%s sess=%s msg=%s spk=%s" % (em.get("dia_id"), em.get("session_index"), em.get("message_index"), em.get("speaker")))
        P("     text: %s" % txt[:90].replace("\n", " "))
        for i in hits[:4]:
            P("     hit#%d MID=%s order=%s raw: %s" % (i, MID[i], get_order(i), RAW[i][:90].replace("\n", " ")))
    cnt += 1

# 统计: 全库order覆盖情况
hasord = sum(1 for i in range(len(MID)) if get_order(i) is not None)
P("\n有order字段的记录: %d/%d" % (hasord, len(MID)))
lens = [len(get_order(i)) for i in range(len(MID)) if get_order(i)]
if lens:
    P("order长度分布: 1条=%d 多条=%d max=%d" % (
        sum(1 for x in lens if x == 1), sum(1 for x in lens if x > 1), max(lens)))
# 消息号是否全覆盖
allmsg = set()
for i in range(len(MID)):
    o = get_order(i)
    if o:
        allmsg |= set(o)
P("order并集覆盖消息号: %d 个 (min=%d max=%d)" % (len(allmsg), min(allmsg) if allmsg else -1, max(allmsg) if allmsg else -1))
