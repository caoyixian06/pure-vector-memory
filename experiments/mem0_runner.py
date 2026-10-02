# -*- coding: utf-8 -*-
"""mem0_runner.py - Mem0 行业对照(glm-5.3-flash 底座, anthropic套餐通道):
① ingest: 10会话按 Session 场次批量 add(mem0 官方 LoCoMo 口径), 说话人映射 user/assistant
② 答题: 100题固定子集 memory.search top5 → glm 回答 → submission
断点: 每场 add 后打 done 标记, 重跑跳过。"""
import io
import json
import os
import re
import sys
import time
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
KEY = "${GLM_KEY}"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
os.environ["ANTHROPIC_API_KEY"] = KEY

HERE = "C:/locomo_refined/memsys"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
SUBSET = HERE + "/subset100b.json"
MARK = HERE + "/mem0_done.txt"
NL = chr(10)

from mem0 import Memory

config = {
    "llm": {"provider": "anthropic", "config": {"model": "glm-5.3-flash", "temperature": 0.0, "max_tokens": 4000}},
    "embedder": {"provider": "openai", "config": {
        "model": "qwen3-embedding:latest",
        "openai_base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama", "embedding_dims": 256}},
    "vector_store": {"provider": "faiss", "config": {
        "collection_name": "locomo", "embedding_model_dims": 256,
        "path": HERE + "/mem0_store"}},
    "version": "v1.1",
}
m = Memory.from_config(config)

def glm(prompt, max_tokens=400):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
                                         json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": KEY, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            t = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

done = set()
if os.path.exists(MARK):
    done = set(l.strip() for l in open(MARK, encoding="utf-8") if l.strip())

data = json.load(open(RAW, encoding="utf-8"))

# ---- ① ingest(断点续传) ----
for ci, item in enumerate(data):
    sid = "loco-" + str(item.get("sample_id", ci))
    conv = item["conversation"]
    dates = {k: v for k, v in conv.items() if k.endswith("date_time")}
    keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("date_time"))
    speakers = []
    for k in keys:
        for u in conv[k]:
            if u["speaker"] not in speakers:
                speakers.append(u["speaker"])
    a, b = (speakers + ["A", "B"])[:2]
    for k in keys:
        tag = "%s|%s" % (sid, k)
        if tag in done:
            continue
        msgs = []
        for u in conv[k]:
            role = "user" if u["speaker"] == a else "assistant"
            msgs.append({"role": role, "content": "%s: %s" % (u["speaker"], u["text"])})
        try:
            m.add(msgs, user_id=sid)
        except Exception as e:
            print("ADD_FAIL", tag, repr(e)[:120], flush=True)
            time.sleep(3)
            try:
                m.add(msgs, user_id=sid)
            except Exception as e2:
                print("ADD_FAIL2", tag, repr(e2)[:120], flush=True)
        with open(MARK, "a", encoding="utf-8") as f:
            f.write(tag + NL)
        print("added", tag, len(msgs), flush=True)

print("INGEST_ALL_DONE", flush=True)

# ---- ② 答题(100题固定子集) ----
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
subset_ids = set(r["qa_id"] for r in json.load(open(SUBSET, encoding="utf-8")))
todo = [q for q in questions if q["qa_id"] in subset_ids]
print("questions:", len(todo), flush=True)

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期,禁止保留 yesterday/last year 等相对表述。")
sub = []
t0 = time.time()
for i, q in enumerate(todo):
    ci = q["conversation_idx"]
    sid = "loco-" + str(data[ci].get("sample_id", ci))
    try:
        hits = m.search(q["question"], filters={"user_id": sid}, limit=5)
        mems = [h.get("memory") or "" for h in (hits.get("results") or []) if h.get("memory")]
    except Exception as e:
        print("SEARCH_FAIL", q["qa_id"], repr(e)[:100], flush=True)
        mems = []
    ctx = NL.join("- " + x for x in mems) if mems else "(无记忆命中)"
    prompt = ("根据记忆回答。若无关回答「不知道」。" + NL + "记忆:" + NL + ctx + NL + NL +
              "问题: " + q["question"] + NL + TIME_RULE)
    sub.append(dict(qa_id=q["qa_id"], predicted_answer=glm(prompt)))
    if i % 10 == 0:
        print("answered", i, flush=True)
with open(HERE + "/mem0_sub100.jsonl", "w", encoding="utf-8") as f:
    for s in sub:
        f.write(json.dumps(s, ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in sub)
with open(HERE + "/mem0_q100.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("MEM0_SUBMISSION_DONE", len(sub), round(time.time() - t0, 1), "s", flush=True)
