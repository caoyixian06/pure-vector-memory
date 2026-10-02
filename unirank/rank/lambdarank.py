# -*- coding: utf-8 -*-
"""unirank.rank.lambdarank — listwise排序学习 [双库验证]
配置 = 2026-09-19 定版(冻结): trunc5/est500/lr0.08/leaves63
负例 = 头部陪衬采样(难例), 不是随机 — 两库验证的配方:
  LoCoMo随机负例同样可用, LME必须头部负例(随机负例学不会头部判别)。
特征必须先经 core.ordinal.pool_rank 分位化(零绝对数值宪法)。
"""
import numpy as np
import lightgbm as lgb

FROZEN = dict(objective="lambdarank", n_estimators=500, learning_rate=0.08,
              num_leaves=63, min_child_samples=20, lambdarank_truncation_level=5,
              random_state=0, verbosity=-1, n_jobs=4)


def head_negatives(pool_gold_flags, order=None, n_gold=8, head_depth=48, n_noise=40,
                   seed=31):
    """头部负例采样: gold行前n_gold + (池序/给定序)前head_depth非金行前n_noise。
    pool_gold_flags: 长n布尔(是否金); order: 候选排序(默认池序=cos序)。
    返回 (rows, labels)
    """
    n = len(pool_gold_flags)
    order = order if order is not None else list(range(n))
    gold_rows = [r for r, g in enumerate(pool_gold_flags) if g][:n_gold]
    gset = set(gold_rows)
    noise_rows = [r for r in order[:head_depth] if r not in gset][:n_noise]
    rows = gold_rows + noise_rows
    labels = [1 if r in gset else 0 for r in rows]
    return rows, labels


def train(feats_by_q, gold_flags_by_q, order_by_q=None):
    """feats: 每题(n_candidates, n_features) — 必须已分位化。"""
    trF, trY, trG = [], [], []
    for qi, Fm in feats_by_q.items():
        rows, labels = head_negatives(gold_flags_by_q[qi],
                                      order_by_q[qi] if order_by_q else None)
        for r, y in zip(rows, labels):
            trF.append(Fm[r])
            trY.append(y)
        trG.append(len(rows))
    rk = lgb.LGBMRanker(**FROZEN)
    rk.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
    return rk


def predict(rk, Fm):
    return rk.predict(Fm)
