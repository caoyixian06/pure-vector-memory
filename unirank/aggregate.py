# -*- coding: utf-8 -*-
"""unirank.aggregate — 三级聚合: 原子句 → 记录 → 会话
每个信号只在其正确的层级生效(形态注记):
  G1:      句级均值判别=0(-0.02), 会话max=0.884 → 只用 max [形态]
  寒暄:    只在句级可见 [粒度]
  三团/团: 只在聚合粒度可见 [粒度]
"""
import re
import numpy as np
from .core.signal_features import q_stems, stem, QSTOP


def session_max_g1(question_text, sess_records, raw_texts):
    """会话max-G1 [双库/形态max]: 0.884 AUC — LME最强会话级信号之一。"""
    qs = q_stems(question_text)
    best = 0.0
    for r in sess_records:
        rsts = set(stem(w) for w in re.findall(r"[a-z']+", raw_texts[r].lower()))
        g = len(qs & rsts) / max(1, len(qs))
        if g > best:
            best = g
    return best


def aggregate_scores(scores_by_level, how):
    """通用聚合: how in {'max','mean','topk_mean'}。层级分数→上层分数。"""
    if not scores_by_level:
        return 0.0
    v = np.asarray(list(scores_by_level), dtype=np.float32)
    if how == "max":
        return float(v.max())
    if how == "mean":
        return float(v.mean())
    if how == "topk_mean":
        k = max(1, len(v) // 3)
        return float(np.sort(v)[-k:].mean())
    raise ValueError(how)
