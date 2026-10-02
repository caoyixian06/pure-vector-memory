# -*- coding: utf-8 -*-
"""foundry38.py — 7反例解剖: 金片原文+问题全量拉取, 归因分类"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry38_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

VIOLS = ["conv-49#q0010", "conv-43#q0016", "conv-43#q0155", "conv-49#q0017",
         "conv-49#q0007", "conv-47#q0028", "conv-48#q0038"]
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q

for qa in VIOLS:
    q = Q[qa]
    P("=== %s (cat=%s)" % (qa, q.get("category")))
    P(" Q: %s" % q["question"])
    P(" A: %s" % "; ".join(str(x) for x in (q.get("answer") or [])))
    for em in (q.get("evidence_messages") or []):
        P("   E[%s]: %s" % (em.get("speaker") or "?", str(em.get("text") or "")[:100]))
    P("")
P("F38_DONE")
LOG.close()
