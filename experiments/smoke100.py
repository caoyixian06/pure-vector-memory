# -*- coding: utf-8 -*-
"""smoke100.py — 100题x3遍x48并发判分试水: 并发通不通+逐题细节落盘供人工核对"""
import io, json, os, re, sys, time, urllib.request, threading
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
SUB = HERE + "/submission_r40bf.jsonl"
import sys as _s
NW = int(_s.argv[1]) if len(_s.argv) > 1 else 48
NQ = int(_s.argv[2]) if len(_s.argv) > 2 else 100

ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
qmap = {q["qa_id"]: q for q in ALLQS}
rows = [json.loads(l) for l in open(SUB, encoding="utf-8") if l.strip()][:NQ]
print("smoke rows=%d threads=%d tasks=%d" % (len(rows), NW, 3*len(rows)), flush=True)

PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""
WLOCK = threading.Lock()
CNT = {"n": 0, "err": 0, "lat": []}
def judge_one(args):
    pi, r = args
    q = qmap.get(r["qa_id"], {})
    gold = "; ".join(str(x) for x in (q.get("answer") or []))
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=q.get("question", ""), g=gold, p=str(r.get("predicted_answer") or ""))}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
    t0 = time.time()
    for att in range(4):
        try:
            req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + QWEN_KEY})
            with urllib.request.urlopen(req, timeout=120) as r2:
                d = json.loads(r2.read())
            txt = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", []))
            m = re.search(r'"score"\s*:\s*([01])', txt)
            if m:
                with WLOCK:
                    CNT["n"] += 1
                    CNT["lat"].append(time.time() - t0)
                return (pi, r["qa_id"], int(m.group(1)))
        except Exception as e:
            with WLOCK:
                CNT["err"] += 1
            time.sleep(1.5 * (att + 1))
    return (pi, r["qa_id"], -1)

tasks = [(1, r) for r in rows]
t0 = time.time()
with ThreadPoolExecutor(NW) as ex:
    results = list(ex.map(judge_one, tasks))
el = time.time() - t0
lat = sorted(CNT["lat"])
print("wall=%.1fs throughput=%.1f calls/s 重试/错误=%d 延迟p50/p95=%.2f/%.2fs" % (
    el, len(tasks)/el, CNT["err"], lat[len(lat)//2], lat[int(len(lat)*0.95)]), flush=True)

passes = {1: {}}
for pi, qa, s in results:
    passes[pi][qa] = s
pct = []
vals = list(passes[1].values())
ok = sum(1 for v in vals if v == 1); valid = sum(1 for v in vals if v in (0, 1))
pct.append(100.0*ok/max(1, valid))
una = -1
print("FULL single pass = %.1f%%" % pct[0], flush=True)

bad = sum(1 for qa in passes[1] if passes[1][qa] == -1)
with open(HERE + "/smoke100_detail.txt", "w", encoding="utf-8") as f:
    f.write(" Majority Vote | GOLD | PRED | QUESTION\n")
    for r in rows:
        q = qmap.get(r["qa_id"], {})
        s1 = passes[1][r["qa_id"]]
        f.write("%s %d | G:%s | P:%s | Q:%s\N{LF}" % (
            r["qa_id"], s1,
            "; ".join(str(x) for x in (q.get("answer") or []))[:60],
            str(r.get("predicted_answer") or "")[:100],
            q.get("question", "")[:70]))
print("detail written, parse_fail=%d" % bad, flush=True)
print("SMOKE100_DONE", flush=True)
