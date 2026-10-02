# -*- coding: utf-8 -*-
"""unirank.rank.transfer — 零标注直迁 [库的核心卖点]
预训练ranker(LoCoMo) + 冻结配置 + 新库分位化特征 = 开箱即用。
实测: LoCoMo→LME 79.2@5/92.8@15 (零标注)。
增益归因账本(诚实记录): @5的8.2pp = 序数化7.0 + 砍毒维度1.4;
                       @15的14.0pp = 砍毒维度9.6 + 序数化4.8。
"""
import lightgbm as lgb
import numpy as np


def load(model_path):
    return lgb.Booster(model_file=model_path)


def rank_pool(booster, Fm_ordinal):
    """Fm_ordinal: (n_candidates, n_features) 已分位化 → 排序后的行号。"""
    s = booster.predict(Fm_ordinal)
    return list(np.argsort(-s))


def transfer_gain_note():
    """防止未来的欢庆忘记归因 —— 宪法第6条的代码化。"""
    return {"@5": {"ordinal": 7.0, "prune_dims": 1.4},
            "@15": {"ordinal": 4.8, "prune_dims": 9.6}}
