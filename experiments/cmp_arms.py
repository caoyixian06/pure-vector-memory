# -*- coding: utf-8 -*-
"""cmp_arms.py — 臂A vs 基线 分题型对比 + 翻转分析"""
import io, json, sys
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

A = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_r33a.json", encoding="utf-8"))}
B = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_dense_all.json", encoding="utf-8"))}

catA, catB = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
flip_pos, flip_neg = [], []
for qid, a in A.items():
    b = B.get(qid)
    if not b:
        continue
    c = str(a.get("category"))
    catA[c][1] += 1; catB[c][1] += 1
    ao, bo = a.get("llm_score") == 1, b.get("llm_score") == 1
    catA[c][0] += ao; catB[c][0] += bo
    if ao and not bo:
        flip_pos.append(a)
    elif bo and not ao:
        flip_neg.append(a)

print("cat | 基线 | 臂A")
for c in sorted(catA):
    ob, nb = catB[c]; oa, na = catA[c]
    print(" %s  | %3d/%d(%.0f%%) | %3d/%d(%.0f%%)" % (
        c, ob, nb, 100 * ob / nb, oa, na, 100 * oa / na))
print("翻转: 基线错→臂A对 %d ; 基线对→臂A错 %d (净%d)" % (
    len(flip_pos), len(flip_neg), len(flip_pos) - len(flip_neg)))
print()
print("=== 基线对→臂A错 样例5 ===")
for r in flip_neg[:5]:
    print("Q:", r["question"][:75])
    print("   gold:", str(r["answer"])[:60], "| cat", r["category"])
    print("   臂A答:", str(r.get("predicted_answer"))[:100])
print()
print("=== 基线错→臂A对 样例3 ===")
for r in flip_pos[:3]:
    print("Q:", r["question"][:75], "| gold:", str(r["answer"])[:50])
