# -*- coding: utf-8 -*-
"""rejudge_now.py — 立即用真gold重判既有submission(纯API零GPU, 与af并行安全)"""
import io, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
QWEN_KEY = "${QWEN_KEY}"
QWEN_API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"

ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
qmap = {q["qa_id"]: q for q in ALLQS}

PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""

def judge(r):
    q = qmap.get(r["qa_id"], {})
    gold = "; ".join(str(x) for x in (q.get("answer") or []))
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=q.get("question", ""), g=gold, p=str(r.get("predicted_answer") or ""))}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
    for att in range(3):
        try:
            req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + QWEN_KEY})
            with urllib.request.urlopen(req, timeout=120) as r2:
                d = json.loads(r2.read())
            txt = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", []))
            m = re.search(r'"score"\s*:\s*([01])', txt)
            if m:
                return int(m.group(1))
        except Exception:
            time.sleep(2)
    return -1

TARGETS = [
    ("bf_full1382", HERE + "/submission_r40bf.jsonl"),
    ("r40b_500", HERE + "/submission_r40.jsonl"),
    ("r39_100glm", HERE + "/submission_r39.jsonl"),
    ("af_full1382", HERE + "/submission_r40af.jsonl"),
]
_js = {"n": 0}
def _jt(r):
    s = judge(r)
    _js["n"] += 1
    if _js["n"] % 200 == 0:
        print("progress", _js["n"], flush=True)
    return s

for tag, path in TARGETS:
    out = HERE + "/rejudge_%s.txt" % tag
    if not os.path.exists(path):
        print("skip(no file):", tag, flush=True)
        continue
    if os.path.exists(out):
        print("skip(done):", tag, flush=True)
        continue
    t0 = time.time()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    _js["n"] = 0
    with ThreadPoolExecutor(4) as jex:
        scores = list(jex.map(_jt, rows))
    ok = sum(1 for s in scores if s == 1)
    valid = sum(1 for s in scores if s in (0, 1))
    pct = round(100 * ok / max(1, valid), 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write("%s(gold) %d/%d = %.1f%%\n" % (tag, ok, valid, pct))
    print("REJUDGE", tag, ok, "/", valid, "=", pct, "%% %.0fs" % (time.time() - t0), flush=True)
print("REJUDGE_ALL_DONE", flush=True)
