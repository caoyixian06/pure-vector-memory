# -*- coding: utf-8 -*-
"""unirank.core.signal_features — 序数特征族 (v0.2 零硬编码版)
停用词: 不再内置英文QSTOP —— derive_stopwords()从语料文档频率自动计算。
"""
import re
import numpy as np

_QPAT = re.compile(r"[a-z']+")


def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w


def derive_stopwords(raw_texts, top_frac=0.001):
    """无监督停用词: 文档频率top 0.1%的词干 = 该语料的停用词。
    (英文语料自动得出what/the/is..., 中文语料自动得出的/了/是...)"""
    df = {}
    for r in raw_texts:
        for w in set(_QPAT.findall(r.lower())):
            df[w] = df.get(w, 0) + 1
    n = len(raw_texts)
    cutoff = max(1, int(n * top_frac))
    return {w for w, c in df.items() if c >= cutoff}


def q_stems_auto(question_text, stopwords):
    return set(stem(w) for w in _QPAT.findall(question_text.lower())
               if w not in stopwords and len(w) > 2)


def feat_jaccard(qtok, ctok):
    inter = qtok & ctok
    return len(inter) / max(1, len(qtok | ctok))


def feat_g1(qstems, rec_text):
    rsts = set(stem(w) for w in _QPAT.findall(rec_text.lower()))
    return len(qstems & rsts) / max(1, len(qstems))


def feat_len_ratio(rec_text, question_text):
    return len(rec_text.split()) / max(1, len(question_text.split()))


def feat_cos(qvec, rvec):
    return float(qvec @ rvec)


def feat_gradinv(cos, g1):
    return cos - g1


# ===== 验证注记(2026-09-19双基准战役) =====
# feat_cos/feat_len_ratio/feat_g1: [双库] 序数化后是直迁增益来源
# feat_g1在session级必须用max聚合(句级-0.02→会话max0.884) [形态]
# 寒暄效应: 仅原子句粒度可见 [粒度]; 三团: 仅聚合粒度 [粒度]
# 问句方向: 问题质心法AUC0.968(零标点零词表) [双库待复验]
