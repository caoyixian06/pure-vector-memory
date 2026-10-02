# -*- coding: utf-8 -*-
"""judge_smoke.py — 验证真gold判官: 3题x预期(1/0/1)"""
import io, json, re, sys, time, urllib.request, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
QWEN_KEY = "${QWEN_KEY}"
QWEN_API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""
CASES = [
    ("When did Caroline go to the LGBTQ support group?", ["7 May 2023"], "7 May 2023", 1),
    ("When did Melanie paint a sunrise?", ["2022"], "Melanie painted a sunrise in May 2023.", 0),
    ("What fields would Caroline be likely to pursue in her education?",
     ["Counseling or mental health"], "Caroline would likely pursue counseling or mental health fields.", 1),
]
def judge(q, gold, pred):
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=q, g="; ".join(str(x) for x in gold), p=pred)}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
    req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + QWEN_KEY})
    with urllib.request.urlopen(req, timeout=120) as r2:
        d = json.loads(r2.read())
    txt = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", []))
    m = re.search(r'"score"\s*:\s*([01])', txt)
    return (int(m.group(1)) if m else -1), txt.strip()[:80]
ok = 0
for q, gold, pred, exp in CASES:
    s, raw = judge(q, gold, pred)
    ok += (s == exp)
    print("exp=%d got=%d raw=%s" % (exp, s, raw), flush=True)
print("SMOKE %d/3" % ok, flush=True)
