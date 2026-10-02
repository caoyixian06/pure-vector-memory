# -*- coding: utf-8 -*-
"""pdfx.py — TAG论文文本提取"""
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pypdf import PdfReader
r = PdfReader(r"C:\locomo_refined\tag_paper.pdf")
print("pages:", len(r.pages))
t = ""
for p in r.pages:
    t += p.extract_text() + "\n"
with open(r"C:\locomo_refined\tag_paper.txt", "w", encoding="utf-8") as f:
    f.write(t)
print("chars:", len(t))
