# -*- coding: utf-8 -*-
"""unirank.core.ordinal — 池内分位变换: 通用排序库的心脏
原则: 零绝对数值。任何特征在进入排序器之前, 先变成"本题候选池内的相对位次"。
分布无关, 跨库直迁, 这一步值 79.2@5/92.8@15 (LoCoMo→LME 零标注)。
"""
import numpy as np


def pool_rank(Fm):
    """每列 -> 池内分位 rank/(n-1)。

    Fm: (n_candidates, n_features) 任意量纲
    返回: 同形状, 每列均匀分布于[0,1]
    """
    n = Fm.shape[0]
    out = np.empty_like(Fm)
    denom = max(1, n - 1)
    for j in range(Fm.shape[1]):
        order = np.argsort(Fm[:, j], kind="stable")
        out[order, j] = np.arange(n, dtype=np.float32) / denom
    return out


def pool_rank_grouped(Fm, groups):
    """组内分位: 适合"会话内排序"类特征(如同会话候选的相对位置)。

    groups: 长n的组标签数组(如会话id)
    每列在每个组内部分别做分位变换。
    """
    Fm = np.asarray(Fm, dtype=np.float32)
    out = np.zeros_like(Fm)
    ug = np.unique(groups)
    for g in ug:
        idx = np.where(groups == g)[0]
        out[idx] = pool_rank(Fm[idx])
    return out


def safe_merge(*cols):
    """把多个已是分位的列做等权平均 —— 结果仍是[0,1]的无量纲分数。
    (合并时不引入绝对数值)"""
    M = np.stack(cols, axis=1)
    return M.mean(axis=1)
