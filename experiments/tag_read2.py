# -*- coding: utf-8 -*-
"""tag_read2.py — TAG论文第二段提取"""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t = open(r"C:\locomo_refined\tag_paper.txt", encoding="utf-8").read()
for kw in ("Qwen", "embedding model", "fact extraction", "state-of-the-art", "imitation", "Limitations", "limitation", "LLaMA", "GPT", "answer generation", "granularity selection", "Table 1", "Table 3"):
    idxs = [m.start() for m in re.finditer(kw, t)]
    print(f"\n#### '{kw}' x{len(idxs)}", flush=True)
    for ix in idxs[:2]:
        seg = t[max(0, ix-130):ix+330].replace("\n", " ")
        print("  ...", seg[:400], flush=True)
