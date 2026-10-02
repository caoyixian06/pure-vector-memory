# -*- coding: utf-8 -*-
"""judge_qwen.py — 官方Qwen3-14B判官重判r37 (LoCoMo-Refined官方口径)
- 接口: 阿里云百炼专属实例 OpenAI兼容 /chat/completions
- 模型: qwen3-14b, 非流式, temp=0, 思考默认开(解析content字段)
- 限流: RPM600/TPM1M — 并发20 + TPM滑动窗口保护
- 判分prompt: 官方Refined原则"宽容不矛盾, 完整不越权"
阶段1: 冒烟3题; 阶段2: 全量1382题, 断点续传
"""
import io, json, os, sys, re, time, threading, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
KEY = "${QWEN_KEY}"
MODEL = "qwen3-14b"
HERE = "C:/locomo_refined/memsys"

PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""

def call_judge(q, g, p, timeout=120):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(q=q, g=g, p=p)}],
        "temperature": 0.0, "enable_thinking": False,
        "max_tokens": 2048,
        "stream": False,
    }
    req = urllib.request.Request(API, json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r2:
        d = json.loads(r2.read())
    txt = ""
    for ch in d.get("choices", []):
        txt += ch.get("message", {}).get("content", "") or ""
    m = re.search(r'"score"\s*:\s*([01])', txt)
    if m:
        return int(m.group(1)), txt
    # 兜底: 单字符1/0
    m2 = re.search(r'\b([01])\b', txt)
    if m2:
        return int(m2.group(1)), txt
    return -1, txt

rows = json.load(open(HERE + "/out_r37.json", encoding="utf-8"))
print("题目:", len(rows), flush=True)

# 断点续传
OUTP = HERE + "/out_r37_qwen14b.jsonl"
done = {}
if os.path.exists(OUTP):
    for l in open(OUTP, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            done[d["qa_id"]] = d["qwen_score"]
print("已判:", len(done), flush=True)

todo = [r for r in rows if r["qa_id"] not in done]
print("待判:", len(todo), flush=True)

# TPM滑动窗口保护
LOCK = threading.Lock()
TOKWIN = []  # (timestamp, tokens)
def tpm_guard(est_tokens=2500):
    with LOCK:
        now = time.time()
        while TOKWIN and TOKWIN[0][0] < now - 60:
            TOKWIN.pop(0)
        used = sum(t for _, t in TOKWIN)
        if used + est_tokens > 900000:
            wait = 60 - (now - TOKWIN[0][0]) + 1 if TOKWIN else 5
            return wait
        TOKWIN.append((now, est_tokens))
        return 0

def work(r):
    for attempt in range(4):
        w = tpm_guard()
        if w:
            time.sleep(w)
            continue
        try:
            gold = "; ".join(str(x) for x in (r.get("answer") or []))
            s, raw = call_judge(r["question"], gold, str(r.get("predicted_answer") or ""))
            if s in (0, 1):
                with LOCK:
                    with open(OUTP, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(qa_id=r["qa_id"], qwen_score=s), ensure_ascii=False) + "\n")
                return s
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    with LOCK:
        with open(OUTP, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(qa_id=r["qa_id"], qwen_score=-1), ensure_ascii=False) + "\n")
    return -1

# 阶段1: 冒烟3题
print("== 阶段1 冒烟3题 ==", flush=True)
for r in todo[:3]:
    gold = "; ".join(str(x) for x in (r.get("answer") or []))
    t0 = time.time()
    s, raw = call_judge(r["question"], gold, str(r.get("predicted_answer") or ""))
    print("  score=%d (%.1fs) raw=%s" % (s, time.time() - t0, raw[-120:]), flush=True)
print("冒烟OK, 若上面score=-1请停; 否则继续全量", flush=True)

# 阶段2: 全量
t0 = time.time()
with ThreadPoolExecutor(20) as ex:
    results = list(ex.map(work, [r for r in todo if r["qa_id"] not in done][:1000000]))
ok = sum(1 for s in results if s in (0, 1))
print("JUDGE_QWEN_DONE 完成%d 成功%d 失败%d 用时%.0fs" % (
    len(results), ok, len(results) - ok, time.time() - t0), flush=True)
