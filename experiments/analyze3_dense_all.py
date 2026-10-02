# -*- coding: utf-8 -*-
"""analyze3_dense_all.py — 列举型金答案占比量化"""
import json, collections

rs = json.load(open(r'C:\locomo_refined\memsys\out_dense_all.json', encoding='utf-8'))

def is_list_answer(ans):
    a = str(ans[0]) if ans else ''
    return (',' in a) or (' and ' in a.lower()) or (len([x for x in a.split(',') if x.strip()]) >= 2)

stat = collections.defaultdict(lambda: [0, 0])  # cat -> [list_err, list_total_err]
okstat = collections.defaultdict(lambda: [0, 0])
for r in rs:
    c = str(r.get('category', '?'))
    ans = r.get('answer') or []
    if not ans:
        continue
    if is_list_answer(ans):
        if r.get('llm_score') == 1:
            okstat[c][0] += 1
        okstat[c][1] += 1
        if r.get('llm_score') != 1:
            stat[c][0] += 1
        stat[c][1] += 1

print('cat | list-type questions | list-type correct | list-type acc')
for c in sorted(okstat):
    o, n = okstat[c]
    print(c, '|', n, '|', o, '|', round(100 * o / n, 1) if n else '-')

# 全体: 列举题 vs 非列举题 正确率
lo = lt = wo = wt = 0
for r in rs:
    ans = r.get('answer') or []
    if not ans:
        continue
    isl = is_list_answer(ans)
    ok = r.get('llm_score') == 1
    if isl:
        lt += 1; lo += ok
    else:
        wt += 1; wo += ok
print('list-answer questions:', lo, '/', lt, '=', round(100 * lo / lt, 1))
print('single-answer questions:', wo, '/', wt, '=', round(100 * wo / wt, 1))
