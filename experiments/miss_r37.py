# -*- coding: utf-8 -*-
"""miss_r37.py — r37(64.3%定版)错题整理: 分桶+分层+样本, 全离线零GLM"""
import io, json, sys, re
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
R = {r["qa_id"]: r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8"))}
OLD = {r["qa_id"]: r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))}

def is_dk(p):
    if not p:
        return True
    p = str(p).strip().lower()
    for kw in ["not sure", "don't know", "dont know", "unknown", "no information",
               "not mentioned", "not specified", "cannot", "无法", "不知道", "未提及"]:
        if kw in p:
            return True
    return len(p) <= 2

miss = [r for r in R.values() if r.get("llm_score") != 1]
print("r37错题总数:", len(miss), "/ 1382")
cat = defaultdict(lambda: [0, 0])
dk = fixed = newmiss = 0
still = []
for r in miss:
    c = str(r["category"])
    cat[c][1] += 1
    o = OLD.get(r["qa_id"])
    if o and o.get("llm_score") == 1:
        newmiss += 1
    if is_dk(r.get("predicted_answer")):
        dk += 1
    if o and o.get("llm_score") != 1:
        fixed += 1
        still.append(r)
print("分题型错题:", {c: "%d/%d(%.0f%%)" % (v[1], v[1], 100) if False else v[1] for c, v in sorted(cat.items())})
print("答不知道型: %d (%.0f%%)" % (dk, 100 * dk / len(miss)))
print("老错题(基线也错,被r37继承): %d | 新翻案(基线对r37错): %d | 基线错r37对: %d" % (
    fixed, newmiss, sum(1 for o in OLD.values() if o.get("llm_score") != 1) - fixed))
# 基线错题有多少被修好
old_miss = sum(1 for o in OLD.values() if o.get("llm_score") != 1)
print("(基线错%d → r37修回%d, 修复率%.0f%%)" % (old_miss, fixed, 100 * fixed / old_miss))
# 各题型错题率
print()
qs = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
ev_rank_cache = {}
print("=== 各题型 r37 错题率 ===")
for c in sorted(cat):
    tot = sum(1 for r in R.values() if str(r["category"]) == c)
    print("  cat%s: %d/%d 错 (%.0f%%)" % (c, cat[c][1], tot, 100 * cat[c][1] / tot))
print()
print("=== 不知道型样例4 ===")
n = 0
for r in miss:
    if is_dk(r.get("predicted_answer")) and n < 4:
        print("Q:", r["question"][:70], "| gold:", str(r["answer"])[:45])
        n += 1
print()
print("=== 实答错样例4(新翻进来的) ===")
n = 0
for r in miss:
    o = OLD.get(r["qa_id"])
    if o and o.get("llm_score") == 1 and not is_dk(r.get("predicted_answer")) and n < 4:
        print("Q:", r["question"][:70], "| gold:", str(r["answer"])[:45])
        print("   r37:", str(r.get("predicted_answer"))[:110])
        n += 1
# 落盘
with open(HERE + "/miss_r37.jsonl", "w", encoding="utf-8") as f:
    for r in miss:
        f.write(json.dumps(dict(qa_id=r["qa_id"], category=r["category"], question=r["question"],
                                answer=r["answer"], evidence_messages=r.get("evidence_messages"),
                                predicted_answer=r.get("predicted_answer")), ensure_ascii=False) + "\n")
print()
print("MISS_R37_WRITTEN", len(miss))
