# -*- coding: utf-8 -*-
"""check_spread.py — 枚举题多证据的会话距离分布: 用户假设"碎片相隔很远/跨会话"验证"""
import io, json, sys, re
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

qs = [json.loads(l) for l in open(
    r"C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()]
multi = []
for q in qs:
    evs = q.get("evidence_messages") or []
    if len(evs) >= 2:
        multi.append(q)
print("多证据题:", len(multi), "/", len(qs))
diffs = []
sames = 0
pairs = 0
for q in multi:
    ss = [e.get("session_index") for e in q["evidence_messages"] if e.get("session_index") is not None]
    for a in range(len(ss)):
        for b in range(a + 1, len(ss)):
            d = abs(ss[a] - ss[b])
            diffs.append(d)
            pairs += 1
            if d == 0:
                sames += 1
diffs = np.array(diffs)
print("证据对总数:", pairs, " 同一session内: %d(%.0f%%)" % (sames, 100 * sames / pairs))
print("session间隔分布: 中位%.0f 均值%.1f p75=%.0f p90=%.0f 最大%d" % (
    np.median(diffs), diffs.mean(), np.percentile(diffs, 75), np.percentile(diffs, 90), diffs.max()))
print("间隔>=2个session的证据对: %.0f%%" % (100 * (diffs >= 2).mean()))
print("间隔>=4个session的证据对: %.0f%%" % (100 * (diffs >= 4).mean()))
# 每会话session总数参考
import collections
sesscnt = collections.Counter()
for q in qs:
    sid = q.get("sample_id")
    for e in q.get("evidence_messages") or []:
        sesscnt[sid] = max(sesscnt[sid], e.get("session_index") or 0)
print("各会话最大session号:", dict(sorted(sesscnt.items(), key=lambda x: str(x[0]))))
