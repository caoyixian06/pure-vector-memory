# -*- coding: utf-8 -*-
"""analyze2_dense_all.py — 错题模式深挖: 不知道型 vs 答错型, 分cat统计"""
import json, collections

rs = json.load(open(r'C:\locomo_refined\memsys\out_dense_all.json', encoding='utf-8'))

def is_dontknow(p):
    if not p:
        return True
    p = str(p).strip().lower()
    for kw in ['not sure', "don't know", 'dont know', 'unknown', 'no information', 'not mentioned',
               'not specified', 'cannot', 'not provided', 'not clear', 'insufficient', '无法', '不知道', '未提及']:
        if kw in p:
            return True
    return len(p) <= 2

mode = collections.defaultdict(lambda: collections.defaultdict(int))
tot = collections.defaultdict(int)
for r in rs:
    c = str(r.get('category', '?'))
    tot[c] += 1
    if r.get('llm_score') == 1:
        mode[c]['correct'] += 1
    else:
        p = r.get('predicted_answer')
        if is_dontknow(p):
            mode[c]['dontknow'] += 1
        else:
            mode[c]['wrong_answer'] += 1

print('cat | total | correct | dontknow | wrong_answer')
for c in sorted(tot):
    m = mode[c]
    print(c, '|', tot[c], '|', m['correct'], '|', m['dontknow'], '|', m['wrong_answer'])

# wrong_answer 里有多少 judge 认为是部分对(llm_score 0.5 之类)
scores = collections.Counter(str(r.get('llm_score')) for r in rs)
print('llm_score dist:', dict(scores))

# 抽样: cat1 错题前 12 条 (question / gold / predicted)
print()
print('=== cat1 wrong samples ===')
n = 0
for r in rs:
    if str(r.get('category')) == '1' and r.get('llm_score') != 1:
        print('Q:', r.get('question'))
        print('  gold:', r.get('answer'))
        print('  pred:', str(r.get('predicted_answer'))[:120])
        ev = r.get('evidence_messages') or []
        if ev:
            e0 = ev[0]
            print('  ev:', e0.get('speaker', '?'), ':', str(e0.get('message', e0.get('text', '')))[:100])
        print('  judge:', str(r.get('llm_reason'))[:160])
        n += 1
        if n >= 12:
            break
