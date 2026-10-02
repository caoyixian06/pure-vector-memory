# -*- coding: utf-8 -*-
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = {}
for l in open(r"C:\locomo_refined\memsys\out_r37_qwen14b.jsonl", encoding="utf-8"):
    if l.strip():
        dd = json.loads(l)
        d[dd["qa_id"]] = dd["qwen_score"]
rs = json.load(open(r"C:\locomo_refined\memsys\out_r37.json", encoding="utf-8"))
tot = ok = fail = 0
for r in rs:
    s = d.get(r["qa_id"], -1)
    if s == -1:
        fail += 1
    else:
        tot += 1
        ok += s
print("有效判分:", tot, "解析失败:", fail)
print("QWEN_OFFICIAL_SCORE", ok, "/", tot, "=", round(100 * ok / max(1, tot), 1), "%")
# 换算回1382口径(失败按0计 vs 按比例)
print("保守口径(失败计0):", ok, "/1382 =", round(100 * ok / 1382, 1), "%")
