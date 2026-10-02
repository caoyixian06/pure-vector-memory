# -*- coding: utf-8 -*-
"""unirank.index.atom_split — 原子句切分 [保留标点版]
粒度因果链的工程化: 寒暄效应/话题信号只在原子句粒度可见(0.488 vs 0.482 → 0.524 vs 0.425);
三团只在聚合粒度可见。库内维护双层粒度。
"""
import re

_FIND = re.compile(r"[^.!?]*[.?]?")


def split_sentences(text, len_range=(4, 45)):
    """切句保留句尾标点。len_range默认为宽松英文窗; 正式管线用
    derive_len_range(语料)按句长分位自动得出(零绝对数值)。"""
    out = []
    for m in _FIND.finditer(text):
        p = m.group(0).strip()
        nw = len(p.split())
        if len_range[0] <= nw <= len_range[1]:
            out.append(p)
    return out


def derive_len_range(raw_texts, lo_q=0.05, hi_q=0.95):
    """无监督句长窗: 全库句子长度分布的[5%,95%]分位。"""
    lens = []
    for r in raw_texts:
        for m in _FIND.finditer(r):
            p = m.group(0).strip()
            if p:
                lens.append(len(p.split()))
    import numpy as np
    v = np.asarray(lens, dtype=np.float32)
    return (max(1, int(np.quantile(v, lo_q))), int(np.quantile(v, hi_q)))


def record_atoms(raw_texts, records=None):
    """批量: raw_texts[i] -> 该记录的原子句列表。
    records: 可选记录索引子集。返回 {rec_idx: [句子]}
    """
    idxs = records if records is not None else range(len(raw_texts))
    return {i: split_sentences(raw_texts[i]) for i in idxs}
