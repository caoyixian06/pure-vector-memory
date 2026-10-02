# -*- coding: utf-8 -*-
"""unirank.index.dual_channel — 双cos通道池构造 [双库验证]
1024(BGE) + 256(qwen) 两通道并集 + 同会话邻域块。
零标注、零绝对阈值(top-K是相对序数参数, 跨库沿用同一组)。
"""
import numpy as np


def build_pool(D, V256, qvec1024, qvec256, sess_ids, topk=350, nb_anchor=30, nb_win=2, cap=700, domain=None):
    """构造单题候选池。

    D: (N,1024) 记录BGE向量(已l2归一)
    V256: (N,256) 记录qwen向量(已l2归一)
    qvec1024/qvec256: 问题向量
    sess_ids: 长N会话标签(用于邻域块)
    domain: 可选, 检索域(候选行号列表, 如haystack记录集; None=全库)
    返回: pool(list[int]), top30锚列表
    """
    rows = np.asarray(domain) if domain is not None else None
    c1 = (D @ qvec1024) if rows is None else (D[rows] @ qvec1024)
    c2 = (V256 @ qvec256) if rows is None else (V256[rows] @ qvec256)
    if rows is None:
        pool = set(np.argsort(-c1)[:topk].tolist()) | set(np.argsort(-c2)[:topk].tolist())
    else:
        pool = set(rows[np.argsort(-c1)[:topk]].tolist()) | set(rows[np.argsort(-c2)[:topk]].tolist())
    anchor = np.argsort(-c1)[:nb_anchor]
    for i in anchor:
        for off in range(-nb_win, nb_win + 1):
            if off == 0:
                continue
            j = int(i) + off
            if 0 <= j < len(sess_ids) and sess_ids[j] == sess_ids[int(i)]:
                pool.add(j)
    return sorted(pool)[:cap], anchor.tolist()
