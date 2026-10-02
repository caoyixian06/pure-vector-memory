# -*- coding: utf-8 -*-
"""mem0_smoke.py - 验证 mem0 配置: glm-5.3-flash(LLM) + ollama qwen3-embedding(嵌入)。"""
import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import os
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
KEY = "${GLM_KEY}"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
os.environ["ANTHROPIC_API_KEY"] = KEY
from mem0 import Memory

config = {
    "llm": {"provider": "anthropic", "config": {
        "model": "glm-5.3-flash", "temperature": 0.0, "max_tokens": 2000}},
    "embedder": {"provider": "openai", "config": {
        "model": "qwen3-embedding:latest",
        "openai_base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama", "embedding_dims": 256}},
    "vector_store": {"provider": "faiss", "config": {"collection_name": "smoke", "embedding_model_dims": 256}},
    "version": "v1.1",
}
m = Memory.from_config(config)
t0 = time.time()
r1 = m.add([{"role": "user", "content": "Maria: I had dinner with my mom yesterday at the new Italian place."},
            {"role": "assistant", "content": "That sounds lovely! How was the food?"}],
           user_id="smoke-1")
print("add1 ok %.1fs" % (time.time() - t0), str(r1)[:200], flush=True)
t0 = time.time()
r2 = m.add([{"role": "user", "content": "Maria: I started aerial yoga classes last week, it is amazing."},
            {"role": "assistant", "content": "Great to hear! How often do you go?"}],
           user_id="smoke-1")
print("add2 ok %.1fs" % (time.time() - t0), str(r2)[:200], flush=True)
t0 = time.time()
hits = m.search("Who did Maria have dinner with?", filters={"user_id": "smoke-1"}, limit=3)
print("search ok %.1fs" % (time.time() - t0), flush=True)
for h in hits.get("results", hits if isinstance(hits, list) else []):
    print("  hit:", str(h.get("memory"))[:80], flush=True)
print("MEM0_SMOKE_PASS", flush=True)
