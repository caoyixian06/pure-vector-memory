# -*- coding: utf-8 -*-
"""rerank_stage0.py — 阶段0: bge-reranker 部署+确定性+速度+机理预览"""
import io, os, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

t0 = time.time()
from FlagEmbedding import FlagReranker
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("MODEL_LOADED %.1fs" % (time.time() - t0), flush=True)

# 机理预览: 三类候选(真答案/问句占座/寒暄占座)对同一问题
q = "What is Caroline's relationship status?"
pairs = [
    [q, "Caroline: Honestly, I'm single right now and quite happy focusing on myself."],
    [q, "Caroline: That's so nice! What pet do you have?"],
    [q, "Melanie: Thanks so much for your support, it means a lot to me!"],
    [q, "Caroline: I've known these friends for four years, they've been there through everything."],
]
s1 = rer.compute_score(pairs, batch_size=4)
print("PREVIEW answer=%.3f question_form=%.3f courtesy=%.3f name_mate=%.3f" % tuple(s1), flush=True)

# 确定性: 同输入3遍
runs = [rer.compute_score(pairs, batch_size=4) for _ in range(3)]
import numpy as np
arr = np.array(runs)
print("DETERMINISM max_diff=%.6g" % float(np.abs(arr - arr[0]).max()), flush=True)

# 速度: 50候选
speed_pairs = [[q, "Caroline: sentence %d about daily life and work and hobbies." % i] for i in range(50)]
t0 = time.time()
rer.compute_score(speed_pairs, batch_size=25)
dt = time.time() - t0
print("SPEED 50pairs=%.2fs → 1376题≈%.0f分钟" % (dt, dt * 1376 / 60), flush=True)
print("STAGE0_DONE", flush=True)
