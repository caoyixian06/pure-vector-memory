# -*- coding: utf-8 -*-
"""inspect_lme.py — LongMemEval数据结构探查(流式, 避免整载内存)"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
p = r"C:\locomo_refined\longmemeval\longmemeval_s_cleaned.json"
# 先看首字符判断结构
with open(p, encoding="utf-8") as f:
    head = f.read(2000)
print("HEAD:", head[:600], flush=True)
print("---", flush=True)
# 如果是list, 流式找第一个完整对象
d = json.loads(head[:0] or "[]")
