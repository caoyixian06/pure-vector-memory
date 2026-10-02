# -*- coding: utf-8 -*-
"""analyze_dense_all.py — 全量1382错题分桶归因"""
import json, collections

rs = json.load(open(r'C:\locomo_refined\memsys\out_dense_all.json', encoding='utf-8'))
qs = {}
for l in open(r'C:\locomo_refined\memsys\questions_dense_all.jsonl', encoding='utf-8'):
    if l.strip():
        q = json.loads(l)
        qs[q['qa_id']] = q

cat = collections.defaultdict(lambda: [0, 0])
conv = collections.defaultdict(lambda: [0, 0])
miss = []
for r in rs:
    q = qs.get(r.get('qa_id'), {})
    ok = 1 if r.get('llm_score') == 1 else 0
    c = str(q.get('category', '?'))
    cat[c][0] += ok
    cat[c][1] += 1
    cv = str(q.get('conversation_idx', '?'))
    conv[cv][0] += ok
    conv[cv][1] += 1
    if not ok:
        miss.append(r)

print('=== by category ===')
for c in sorted(cat):
    o, n = cat[c]
    print('cat', c, o, '/', n, '=', round(100 * o / n, 1))
print('=== worst 8 conversations ===')
rank = sorted(conv.items(), key=lambda kv: kv[1][0] / kv[1][1])
for cv, (o, n) in rank[:8]:
    print('conv', cv, o, '/', n, '=', round(100 * o / n, 1))
print('=== best 3 ===')
for cv, (o, n) in rank[-3:]:
    print('conv', cv, o, '/', n, '=', round(100 * o / n, 1))

# 错题落盘(带题目与官方证据, 供归因)
out = []
for r in miss:
    q = qs.get(r.get('qa_id'), {})
    out.append({
        'qa_id': r.get('qa_id'),
        'category': q.get('category'),
        'conversation_idx': q.get('conversation_idx'),
        'question': q.get('question'),
        'answer': q.get('answer'),
        'evidence': q.get('evidence'),
        'predicted': r.get('predicted_answer') or (r.get('prediction') or {}).get('predicted_answer') if isinstance(r.get('prediction'), dict) else r.get('predicted_answer'),
        'raw_keys': list(r.keys()),
    })
with open(r'C:\locomo_refined\memsys\miss_dense_all.jsonl', 'w', encoding='utf-8') as f:
    for m in out:
        f.write(json.dumps(m, ensure_ascii=False) + '\n')
print('MISS_WRITTEN', len(out))
# 打印第一条错题的完整记录结构, 确认字段名
if rs:
    print('SAMPLE_KEYS', list(rs[0].keys()))
    print('SAMPLE', json.dumps(rs[0], ensure_ascii=False)[:400])
