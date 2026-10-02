# -*- coding: utf-8 -*-
import io, json, sys
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
A = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_r33b.json", encoding="utf-8"))}
AA = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_r33a.json", encoding="utf-8"))}
B = {r["qa_id"]: r for r in json.load(open(r"C:\locomo_refined\memsys\out_dense_all.json", encoding="utf-8"))}
cb, ca, caa = (defaultdict(lambda: [0, 0]) for _ in range(3))
for q, a in A.items():
    b, aa = B.get(q), AA.get(q)
    if not b:
        continue
    c = str(a["category"])
    cb[c][1] += 1; ca[c][1] += 1; caa[c][1] += 1
    ca[c][0] += (a["llm_score"] == 1)
    cb[c][0] += (b["llm_score"] == 1)
    if aa:
        caa[c][0] += (aa["llm_score"] == 1)
print("cat | 基线 | 臂A融合 | 臂B融合+精排")
for c in sorted(cb):
    print(" %s | %3d/%d(%2.0f%%) | %3d/%d(%2.0f%%) | %3d/%d(%2.0f%%)" % (
        c, cb[c][0], cb[c][1], 100 * cb[c][0] / cb[c][1],
        caa[c][0], caa[c][1], 100 * caa[c][0] / caa[c][1],
        ca[c][0], ca[c][1], 100 * ca[c][0] / ca[c][1]))
fp = sum(1 for q in A if A[q]["llm_score"] == 1 and B.get(q) and B[q]["llm_score"] != 1)
fn = sum(1 for q in A if A[q]["llm_score"] != 1 and B.get(q) and B[q]["llm_score"] == 1)
print("B vs 基线翻转: +%d/-%d" % (fp, fn))
sp = sum(1 for q, r in json.loads(open(r"C:\locomo_refined\memsys\submission_r33b.jsonl", encoding="utf-8").read() and "[]" or "[]") ) if False else None
import json as j2
sub = [j2.loads(l) for l in open(r"C:\locomo_refined\memsys\submission_r33b.jsonl", encoding="utf-8") if l.strip()]
suba = [j2.loads(l) for l in open(r"C:\locomo_refined\memsys\submission_r33a.jsonl", encoding="utf-8") if l.strip()]
print("second_pass: 臂A %d / 臂B %d" % (
    sum(1 for s in suba if s.get("second_pass")), sum(1 for s in sub if s.get("second_pass"))))
