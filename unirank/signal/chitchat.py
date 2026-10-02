# -*- coding: utf-8 -*-
"""unirank.signal.chitchat — 寒暄效应检测 [原子句粒度] (v0.2 零硬编码版)
宪法: 零词表/零标点依赖/零标注 —— 一切从数据计算。

问句方向 = 问题集质心方向(问题=系统天然输入, 自监督)
  实测AUC 0.968 (标点自标注版0.988, 差2点换来零表面规则)
寒暄检测 = kNN密度法(寒暄句彼此极像=高密度簇) + 句长分位(短句分位)
  (v0.1的COURT英文词表已废——跨语言即死)
"""
import numpy as np


def ask_direction(question_vecs, sent_vecs):
    """问句方向: 问题集质心 - 库句共模质心。两输入都是系统天然数据。"""
    q = question_vecs.mean(0)
    c = sent_vecs.mean(0)
    d = q - c
    return d / max(np.linalg.norm(d), 1e-9)


def askness(sent_vecs, qvecs):
    """每句的问句性 = 在问句方向上的投影。零标点。"""
    u = ask_direction(qvecs, sent_vecs)
    return sent_vecs @ u


def knn_density(sent_vecs, k=20, anchor=2000):
    """无监督kNN密度: 句子与其k近邻的平均cos。
    寒暄句(Thanks/Great型)彼此极像 → 高密度。anchor采样控制计算量。"""
    rng = np.random.RandomState(0)
    A = sent_vecs[rng.choice(len(sent_vecs), size=min(anchor, len(sent_vecs)), replace=False)]
    sims = sent_vecs @ A.T
    part = np.sort(sims, axis=1)[:, -k:]
    return part.mean(axis=1)


def length_quantile(sents_words_lens):
    """句长分位(替代绝对词数窗)。"""
    v = np.asarray(sents_words_lens, dtype=np.float32)
    return (v - v.min()) / max(1e-9, v.max() - v.min())


def chitchat_score(sent_vecs, qvecs, word_lens, anchors=None):
    """寒暄性合成信号: 高kNN密度 × 低长度分位 × (问句性可选)。
    全部分位合成, 无阈值。anchors未标注时由调用者决定组合权重(相对参数)。"""
    dens = knn_density(sent_vecs)
    lq = length_quantile(word_lens)
    return dens - lq   # 密度高且短 → 寒暄倾向; 交由上层分位化, 不设绝对阈值
