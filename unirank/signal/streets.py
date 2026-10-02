# -*- coding: utf-8 -*-
"""unirank.signal.streets — 语义街区derive (v0.2 零硬编码版)
运行时只提供 derive_street 原语: 分组mask由调用者**计算**得出,
本模块不内置任何词表(时间词/时态词表已移至 bench.autopsy 作离线发现工具)。
"""
import numpy as np


def derive_street(V, mask_hi, mask_lo, thresh=0.01):
    """街区derive原语: hi组均值-lo组均值 → (单位方向, 街区维数, 最强维)。
    mask必须来自计算(如分位/密度/投影), 不得来自词表。"""
    delta = V[mask_hi].mean(0) - V[mask_lo].mean(0)
    street = int((np.abs(delta) > thresh).sum())
    top = int(np.argmax(np.abs(delta)))
    u = delta / max(np.linalg.norm(delta), 1e-9)
    return u, street, top


def street_projection(V, direction):
    """记录/句子在街区方向上的投影分(纯计算, 运行时信号)。"""
    return V @ direction
