# -*- coding: utf-8 -*-
"""replay37.py — r37判官活体重放: 同prompt同gold重发3题, 验证70.5%判分的可复现性"""
import io, json, re, sys, time, os, urllib.request, random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
QKEY = "${QWEN_KEY}"
API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
HERE = "C:/locomo_refined/memsys"

rows = json.load(open(HERE + "/out_r37.json", encoding="utf-8"))
PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""

rng = random.Random(3)
for r in rng.sample(rows, 3):
    gold = "; ".join(str(x) for x in (r.get("answer") or []))
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=r["question"], g=gold, p=str(r.get("predicted_answer") or ""))}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
    req = urllib.request.Request(API, json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + QKEY})
    with urllib.request.urlopen(req, timeout=120) as r2:
        d = json.loads(r2.read())
    txt = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", []))
    print("GOLD:", gold[:36], "| PRED:", str(r.get("predicted_answer"))[:36], "| 判官:", txt.strip()[:20], flush=True)
print("REPLAY_DONE", flush=True)
