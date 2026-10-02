# -*- coding: utf-8 -*-
"""tag_read.py — TAG论文关键段提取"""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t = open(r"C:\locomo_refined\tag_paper.txt", encoding="utf-8").read()
print("总字符:", len(t))
# 找关键词上下文
for kw in ("LoCoMo", "granularit", "atomic", "GPT-", "reward", "navigator", "memory types", "Mem0", "ablation", "Ablation"):
    idxs = [m.start() for m in re.finditer(kw, t, re.I)]
    print(f"\n#### '{kw}' x{len(idxs)}", flush=True)
    for ix in idxs[:3]:
        seg = t[max(0, ix-150):ix+350].replace("\n", " ")
        print("  ...", seg[:420], flush=True)
