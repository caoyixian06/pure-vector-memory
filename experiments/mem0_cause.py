# -*- coding: utf-8 -*-
"""mem0_cause.py — Mem0错因按r34病型分桶: 干扰项型 vs 时间盲型 vs 证据销毁型 vs 其他"""
import io, json, sys, re
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
rs = json.load(open(r"C:\locomo_refined\memsys\out_mem0_anti.json", encoding="utf-8"))

def is_dk(p):
    if not p:
        return True
    p = str(p).strip().lower()
    for kw in ["not sure", "don't know", "dont know", "unknown", "no information",
               "not mentioned", "not specified", "cannot", "not provided", "无法", "不知道", "未提及"]:
        if kw in p:
            return True
    return len(p) <= 2

def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2)

buckets = Counter()
examples = {"interference": [], "temporal": [], "destroyed": [], "other": []}
for r in rs:
    if r.get("llm_score") == 1:
        continue
    q = r.get("question") or ""
    gold = " ".join(r.get("answer") or [])
    pred = str(r.get("predicted_answer") or "")
    cat = str(r.get("category"))
    # 时间盲型: temporal题+预测是具体日期但错
    if cat == "3":
        buckets["时间盲型(cat3)"] += 1
        continue
    # 不知道型 → 证据没存(销毁或没检索到)
    if is_dk(pred):
        buckets["不知道型(证据销毁/未检索)"] += 1
        continue
    # 实答错: 分干扰项 vs 其他
    # 干扰项启发: 预测和gold有 topical 重叠(共享1个以上内容词)但不同(如不同日期/不同事件)
    gt, pt = toks(gold), toks(pred)
    inter = gt & pt
    if len(inter) >= 1:
        buckets["干扰项/近似错(答了相关但错)"] += 1
        if len(examples["interference"]) < 6:
            examples["interference"].append((q, gold, pred[:110]))
    else:
        buckets["其他实答错"] += 1
        if len(examples["other"]) < 4:
            examples["other"].append((q, gold, pred[:110]))

tot = sum(1 for r in rs if r.get("llm_score") != 1)
print("Mem0错题总数:", tot)
for k, v in buckets.most_common():
    print("  %-28s %2d (%.0f%%)" % (k, v, 100 * v / tot))
print()
print("=== 干扰项/近似错 样例 ===")
for q, g, p in examples["interference"]:
    print("Q:", q[:70])
    print("   gold:", g[:60])
    print("   pred:", p)
print()
print("=== 其他实答错 样例 ===")
for q, g, p in examples["other"]:
    print("Q:", q[:70], "| gold:", g[:50], "| pred:", p[:80])
