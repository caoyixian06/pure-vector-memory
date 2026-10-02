# -*- coding: utf-8 -*-
"""unirank.bench.eval — 双口径评测
session@k (LME官方/FluctlightDB可比) + turn级全金@k (LoCoMo学长口径)。
纪律: 分折单位=数据生成单位(LoCoMo=会话/LME=haystack);
      任何"增益"必须同条件双基线(裸cos对照, 域过滤是隐藏变量值40pp)。
"""


def session_recall(order, sess_ids, gold_sessions, k):
    seen = set(sess_ids[r] for r in order[:k])
    return 1.0 if gold_sessions <= seen else 0.0


def turn_all_gold(order, gold_rows, k):
    return 1.0 if all(r in set(order[:k]) for r in gold_rows) else 0.0


def evaluate(orders, sess_ids, gold_sessions_by_q, ks=(5, 15)):
    """orders: {qid: 排序后的记录行号列表}。返回 {k: 平均session recall}。"""
    out = {}
    for k in ks:
        hits = [session_recall(orders[q], sess_ids, gold_sessions_by_q[q], k)
                for q in orders]
        out[k] = float(np.mean(hits)) if hits else 0.0
    return out
