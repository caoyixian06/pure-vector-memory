# -*- coding: utf-8 -*-
"""autopsy_r34.py — r34崩盘验尸: 空答案数/分题型/翻转/错例样本"""
import io, json, sys
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
R = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_r34.json", encoding="utf-8"))}
B = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_dense_all.json", encoding="utf-8"))}
sub = {}
for l in open(r"C:\locomo_refined\memsys\submission_r34.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        sub[d["qa_id"]] = d.get("predicted_answer") or ""

empty = sum(1 for v in sub.values() if not v.strip())
short = sum(1 for v in sub.values() if 0 < len(v.strip()) <= 3)
print("空答案:", empty, " 超短答案:", short, "/", len(sub))
catR, catB = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
fp = fn = 0
neg = []
for qid, r in R.items():
    b = B.get(qid)
    if not b:
        continue
    c = str(r["category"])
    catR[c][1] += 1; catB[c][1] += 1
    ro, bo = r.get("llm_score") == 1, b.get("llm_score") == 1
    catR[c][0] += ro; catB[c][0] += bo
    if ro and not bo:
        fp += 1
    elif bo and not ro:
        fn += 1
        neg.append(r)
print("cat | 基线 | r34")
for c in sorted(catR):
    print(" %s | %3d/%d(%2.0f%%) | %3d/%d(%2.0f%%)" % (
        c, catB[c][0], catB[c][1], 100 * catB[c][0] / catB[c][1],
        catR[c][0], catR[c][1], 100 * catR[c][0] / catR[c][1]))
print("翻转: 基线错→对 %d, 基线对→错 %d" % (fp, fn))
print()
print("=== 基线对→r34错 样例6 ===")
for r in neg[:6]:
    print("Q:", r["question"][:70], "| cat", r["category"], "| gold:", str(r["answer"])[:40])
    print("   r34答:", str(r.get("predicted_answer"))[:130])
