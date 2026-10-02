# -*- coding: utf-8 -*-
"""rejudge.py — 同一份 submission 判 3 次(官方 evaluate CLI 子进程), 报告分布。
用法: python rejudge.py <tag> <questions.jsonl> <submission.jsonl>"""
import io
import json
import subprocess
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
tag, qs, sub = sys.argv[1], sys.argv[2], sys.argv[3]
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
REPO = r"C:\locomo_refined\LoCoMo_refined-main"
KEY = "${GLM_KEY}"
HERE = r"C:\locomo_refined\memsys"

scores = []
for i in (1, 2, 3):
    out = os.path.join(HERE, "rejudge_%s_r%d.json" % (tag, i))
    cmd = [PY, "-X", "utf8", "src\\evaluate.py",
           "--questions-path", qs, "--predictions-path", sub,
           "--output-path", out, "--metrics", "llm",
           "--llm-judge", "refined", "--evaluator-model", "glm-5.3-flash",
           "--evaluator-base-url", "https://api.z.ai/api/paas/v4",
           "--evaluator-api-key", KEY, "--concurrency", "6", "--no-progress"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(out):
        print(tag, "round", i, "FAIL", r.stderr[-300:], flush=True)
        continue
    d = json.load(open(out, encoding="utf-8"))
    sc = sum(1 for x in d if x.get("llm_score") == 1)
    scores.append(sc)
    print(tag, "round%d = %d/100" % (i, sc), flush=True)
if scores:
    print("%s SUMMARY: %s mean=%.1f min=%d max=%d" %
          (tag, scores, sum(scores) / len(scores), min(scores), max(scores)), flush=True)
