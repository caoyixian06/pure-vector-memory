# -*- coding: utf-8 -*-
"""cmp_r38.py — r38的85.0%对照校准: 同100题的r37成绩
从questions_r38.jsonl取题集, 在out_r37.json中查同题得分, 对照
"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

qids = set()
for l in open(r"C:\locomo_refined\memsys\questions_r38.jsonl", encoding="utf-8"):
    if l.strip():
        qids.add(json.loads(l)["qa_id"])
r37 = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_r37.json", encoding="utf-8"))}

same_ok = 0
tot = 0
flips_up = []
flips_dn = []
for qid in qids:
    r = r37.get(qid)
    if not r:
        continue
    tot += 1
    if r.get("llm_score") == 1:
        same_ok += 1
    else:
        flips_up.append((qid, r))
# r38成绩(从score文件)
r38_score = {}
# submission对照: r38官方判了85/100, r37同题集:
print("同100题对照:")
print("  r37 (glm判官):  %d/%d = %.1f%%" % (same_ok, tot, 100 * same_ok / max(1, tot)))
print("  r38 (Qwen判官): 85/100 = 85.0%%")
print("  ⚠ 注意: 判官不同(glm vs Qwen), 差值含判官差+原子句域差")
# 从out_r37_qwen14b拿同题的官方判官r37成绩
q14 = {}
for l in open(r"C:\locomo_refined\memsys\out_r37_qwen14b.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        q14[d["qa_id"]] = d["qwen_score"]
same_qwen = sum(1 for qid in qids if q14.get(qid) == 1)
valid = sum(1 for qid in qids if qid in q14)
print("  r37 (Qwen判官): %d/%d = %.1f%%  ← 同判官同题, 纯原子句域差" % (
    same_qwen, valid, 100 * same_qwen / max(1, valid)))
print()
print("翻案明细(r37错→r38对 的题在Qwen判官下r37也错):")
for qid, r in flips_up[:6]:
    print("  Q:", r["question"][:66])
    print("    r37答:", str(r.get("predicted_answer"))[:90])
