# -*- coding: utf-8 -*-
"""inspect_lme2.py — 完整结构: 第一题全字段+第一个session结构"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = r"C:\locomo_refined\longmemeval\longmemeval_s_cleaned.json"
with open(p, encoding="utf-8") as f:
    d = json.load(f)
print("总题数:", len(d))
q = d[0]
print("字段:", list(q.keys()))
print("q_id:", q["question_id"])
print("q_type:", q["question_type"])
print("q_date:", q["question_date"])
print("answer:", q["answer"])
print("answer_session_ids:", q["answer_session_ids"])
print("haystack_dates[:3]:", q["haystack_dates"][:3])
print("haystack_session_ids[:3]:", q["haystack_session_ids"][:3])
print("haystack_sessions数量:", len(q["haystack_sessions"]))
s0 = q["haystack_sessions"][0]
print("session类型:", type(s0), "长度:", len(s0))
print("session[0]结构:", json.dumps(s0[0], ensure_ascii=False)[:400])
# 所有题目类型
from collections import Counter
c = Counter(x["question_type"] for x in d)
print("题型分布:", dict(c))
# haystack session总数(去重)
allsess = set()
for x in d:
    allsess |= set(x["haystack_session_ids"])
print("唯一session数:", len(allsess))
