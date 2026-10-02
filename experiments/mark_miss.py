# -*- coding: utf-8 -*-
"""mark_miss.py - 错题标记: 从判分结果筛 llm_score==0, 连同官方 evidence 落盘供归因。
用法: python mark_miss.py <out.json> <miss_out.jsonl>"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
out_path, miss_path = sys.argv[1], sys.argv[2]
CAT = {"1": "single-hop", "2": "temporal", "3": "multi-hop", "4": "open-domain"}
recs = json.load(open(out_path, encoding="utf-8"))
qs = {}
for l in open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    if l.strip():
        q = json.loads(l)
        qs[q["qa_id"]] = q
miss = [r for r in recs if r.get("llm_score") == 0]
with open(miss_path, "w", encoding="utf-8") as f:
    for r in miss:
        q = qs.get(r["qa_id"], {})
        f.write(json.dumps(dict(
            qa_id=r["qa_id"], category=CAT.get(str(r.get("category")), r.get("category")),
            question=r["question"], gold=r.get("answer"),
            pred=r.get("predicted_answer") or "",
            evidence=[m.get("text") for m in (q.get("evidence_messages") or [])],
            llm_reason=r.get("llm_reason")), ensure_ascii=False) + chr(10))
from collections import Counter
print("MISS_MARKED:", len(miss), "/", len(recs),
      dict(Counter(CAT.get(str(r["category"]), r["category"]) for r in miss)))
