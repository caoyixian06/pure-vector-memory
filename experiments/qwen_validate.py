# -*- coding: utf-8 -*-
"""qwen_validate.py — bf(qwen3.8-27b)全量答题真实性验证
A. 答案本体体检: 空答/长度/不知道率/second-pass率/双语
B. 判分稳定性: 同submission真gold判3遍(pass1已有=60.3), 逐题一致率+flip分析
"""
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
SUB = HERE + "/submission_r40bf.jsonl"

ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
qmap = {q["qa_id"]: q for q in ALLQS}
rows = [json.loads(l) for l in open(SUB, encoding="utf-8") if l.strip()]
print("=== A. 答案本体体检 n=%d ===" % len(rows), flush=True)
lens = [len(str(r.get("predicted_answer") or "")) for r in rows]
empty = sum(1 for L in lens if L == 0)
dk = sum(1 for r in rows if re.search(r"不知道|无法确定|没有相关信息|无法回答|not mentioned|cannot|unclear|no information",
                                      str(r.get("predicted_answer") or ""), re.I))
cjk = sum(1 for r in rows if re.search(r"[\u4e00-\u9fff]", str(r.get("predicted_answer") or "")))
print("空答=%d 长度p25/50/90=%d/%d/%d 不知道型=%d(%0.1f%%) 含中文=%d" % (
    empty, sorted(lens)[len(lens)//4], sorted(lens)[len(lens)//2], sorted(lens)[int(len(lens)*0.9)],
    dk, 100.0*dk/len(rows), cjk), flush=True)
import random
rng = random.Random(7)
print("--- 随机8题抽视 ---", flush=True)
for r in rng.sample(rows, 8):
    q = qmap.get(r["qa_id"], {})
    print("Q:", q.get("question", "")[:60], flush=True)
    print("  GOLD:", "; ".join(str(x) for x in (q.get("answer") or []))[:80], flush=True)
    print("  PRED:", str(r.get("predicted_answer") or "")[:110], flush=True)

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

print("=== B. 判分稳定性: 再判2遍 ===", flush=True)
passes = {}
# pass1 已有(重判文件), 读回逐题分? 无逐题记录→重判一遍补记录, 共3遍全记录
for p in (1, 2, 3):
    out_scores = HERE + "/qwen_val_p%d.jsonl" % p
    if os.path.exists(out_scores):
        passes[p] = {json.loads(l)["qa_id"]: json.loads(l)["s"] for l in open(out_scores, encoding="utf-8") if l.strip()}
        print("pass%d from disk n=%d" % (p, len(passes[p])), flush=True)
        continue
    t0 = time.time()
    _js = {"n": 0}
    def _jt(r, p=p):
        s = judge(r)
        _js["n"] += 1
        if _js["n"] % 300 == 0:
            print("p%d %d/%d %.0fs" % (p, _js["n"], len(rows), time.time()-t0), flush=True)
        with open(out_scores, "a", encoding="utf-8") as f:
            f.write(json.dumps({"qa_id": r["qa_id"], "s": s}, ensure_ascii=False) + "\n")
        return s
    with ThreadPoolExecutor(4) as ex:
        scores = list(ex.map(_jt, rows))
    passes[p] = {r["qa_id"]: s for r, s in zip(rows, scores)}
    ok = sum(1 for s in scores if s == 1)
    valid = sum(1 for s in scores if s in (0, 1))
    print("PASS%d = %d/%d = %.1f%% (%.0fs)" % (p, ok, valid, 100.0*ok/max(1,valid), time.time()-t0), flush=True)

ids = [r["qa_id"] for r in rows]
trip = [(passes[1].get(i), passes[2].get(i), passes[3].get(i)) for i in ids]
valid_trip = [t for t in trip if all(x in (0, 1) for x in t)]
unanimous = sum(1 for t in valid_trip if t[0] == t[1] == t[2])
means = [sum(t)/3.0 for t in valid_trip]
pcts = [100.0*sum(1 for t in valid_trip if sum(t)/3.0 >= 0.5 and 1 in t)/len(valid_trip)]
pct_each = []
for p in (1, 2, 3):
    vals = [passes[p].get(i) for i in ids]
    ok = sum(1 for v in vals if v == 1); valid = sum(1 for v in vals if v in (0, 1))
    pct_each.append(100.0*ok/max(1, valid))
print("=== 稳定性结论 ===", flush=True)
print("3遍各自: p1=%.1f%% p2=%.1f%% p3=%.1f%%" % tuple(pct_each), flush=True)
print("逐题全一致率(3遍同分): %d/%d = %.1f%%" % (unanimous, len(valid_trip), 100.0*unanimous/max(1,len(valid_trip))), flush=True)
maj = sum(1 for t in valid_trip if sum(t) >= 2)
print("多数票口径: %d/%d = %.1f%%" % (maj, len(valid_trip), 100.0*maj/max(1,len(valid_trip))), flush=True)
print("QWEN_VALIDATE_DONE", flush=True)
