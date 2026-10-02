# -*- coding: utf-8 -*-
"""fast_validate.py — 48并发3遍真gold判分(专属实例吞吐不受限), 目标<5分钟
A. 答案体检(瞬时) B. 3遍判分并行 + 逐题一致率/多数票
"""
import io, json, os, re, sys, time, urllib.request, random, threading
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
NW = 48

ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
qmap = {q["qa_id"]: q for q in ALLQS}
rows = [json.loads(l) for l in open(SUB, encoding="utf-8") if l.strip()]
print("=== A. 答案体检 n=%d ===" % len(rows), flush=True)
lens = sorted(len(str(r.get("predicted_answer") or "")) for r in rows)
empty = sum(1 for L in lens if L == 0)
dk = sum(1 for r in rows if re.search(r"不知道|无法确定|没有相关信息|无法回答|not mentioned|cannot|unclear|no information",
                                      str(r.get("predicted_answer") or ""), re.I))
cjk = sum(1 for r in rows if re.search(r"[\u4e00-\u9fff]", str(r.get("predicted_answer") or "")))
print("空答=%d 长度p25/50/90=%d/%d/%d 不知道型=%d(%.1f%%) 含中文=%d" % (
    empty, lens[len(lens)//4], lens[len(lens)//2], lens[int(len(lens)*0.9)],
    dk, 100.0*dk/len(rows), cjk), flush=True)
rng = random.Random(7)
for r in rng.sample(rows, 8):
    q = qmap.get(r["qa_id"], {})
    print("Q:", q.get("question", "")[:58], "| GOLD:", "; ".join(str(x) for x in (q.get("answer") or []))[:40],
          "| PRED:", str(r.get("predicted_answer") or "")[:80], flush=True)

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
CNT = {"n": 0, "t0": time.time(), "err": 0}
def judge_one(args):
    pi, r = args
    q = qmap.get(r["qa_id"], {})
    gold = "; ".join(str(x) for x in (q.get("answer") or []))
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=q.get("question", ""), g=gold, p=str(r.get("predicted_answer") or ""))}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
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
                    if CNT["n"] % 500 == 0:
                        print("  %d/%d %.0fs (%.1f calls/s)" % (
                            CNT["n"], 3*len(rows), time.time()-CNT["t0"],
                            CNT["n"]/max(0.1, time.time()-CNT["t0"])), flush=True)
                return (pi, r["qa_id"], int(m.group(1)))
        except Exception:
            with WLOCK:
                CNT["err"] += 1
            time.sleep(1.5 * (att + 1))
    return (pi, r["qa_id"], -1)

print("=== B. 3遍x%d并发 ===" % NW, flush=True)
tasks = [(pi, r) for pi in (1, 2, 3) for r in rows]
t0 = time.time()
results = []
with ThreadPoolExecutor(NW) as ex:
    results = list(ex.map(judge_one, tasks))
el = time.time() - t0
print("judging wall=%.0fs (%.1f calls/s, retries=%d)" % (el, len(tasks)/max(0.1, el), CNT["err"]), flush=True)

passes = {1: {}, 2: {}, 3: {}}
for pi, qa, s in results:
    passes[pi][qa] = s
for p in (1, 2, 3):
    with open(HERE + "/qwen_val_p%d.jsonl" % p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"qa_id": r["qa_id"], "s": passes[p][r["qa_id"]]}, ensure_ascii=False) + "\n")
ids = [r["qa_id"] for r in rows]
pct_each = []
for p in (1, 2, 3):
    vals = [passes[p].get(i) for i in ids]
    ok = sum(1 for v in vals if v == 1); valid = sum(1 for v in vals if v in (0, 1))
    pct_each.append(round(100.0*ok/max(1, valid), 1))
trip = [(passes[1][i], passes[2][i], passes[3][i]) for i in ids]
valid_trip = [t for t in trip if all(x in (0, 1) for x in t)]
unanimous = sum(1 for t in valid_trip if t[0] == t[1] == t[2])
maj = sum(1 for t in valid_trip if sum(t) >= 2)
maj11 = sum(1 for t in valid_trip if sum(t) in (2, 3) and t[0] == 1)  # pass1=1且多数=1
print("=== 稳定性结论 ===", flush=True)
print("3遍各自: p1=%.1f%% p2=%.1f%% p3=%.1f%%" % tuple(pct_each), flush=True)
print("逐题3遍全一致: %d/%d = %.1f%%" % (unanimous, len(valid_trip), 100.0*unanimous/max(1, len(valid_trip))), flush=True)
print("多数票: %d/%d = %.1f%%" % (maj, len(valid_trip), 100.0*maj/max(1, len(valid_trip))), flush=True)
print("FAST_VALIDATE_DONE", flush=True)
