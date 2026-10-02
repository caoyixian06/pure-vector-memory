# -*- coding: utf-8 -*-
"""r22_smoke.py - R22 接入冒烟(带断言, 任何一关不过 exit 非零)。
断言1: write 后库行数 = A + 伴生 B
断言2: 检索能命中原句通道(词面零重叠场景)
断言3: 同源去重生效(同 turn 摘要/原句不同时在榜)"""
import json, os, sys

HERE = "C:/locomo_refined/memsys"
os.environ["MEM_FILE"] = HERE + "/smoke.jsonl"
if os.path.exists(HERE + "/smoke.jsonl"):
    os.remove(HERE + "/smoke.jsonl")
sys.path.insert(0, HERE)
import sessionmem
print("version:", sessionmem.SESSIONMEM_VERSION, "raw_index:", sessionmem.RAW_INDEX, flush=True)
assert sessionmem.SESSIONMEM_VERSION == "r22-rawidx" and sessionmem.RAW_INDEX, "GATE1 FAIL"

t1 = "Caroline: 我刚读完 Sapiens 这本书，讲人类大历史的，读完特别震撼。"
t2 = "Melanie: 周末我们全家去湖边露营了，晚上烤棉花糖看星星。"
sessionmem.write_session("smoke-s1", "smoke", [t1, t2], allow_fallback_summary=True)

lines = [json.loads(l) for l in open(HERE + "/smoke.jsonl", encoding="utf-8") if l.strip()]
n_a = sum(1 for r in lines if r.get("kind") != "raw")
n_b = sum(1 for r in lines if r.get("kind") == "raw")
print("lib lines=%d A=%d B(raw)=%d" % (len(lines), n_a, n_b), flush=True)
assert len(lines) == 4 and n_a == 2 and n_b == 2, "GATE2a FAIL: A+B composition wrong"

res = sessionmem.retrieve("用户最喜欢的书是什么", current_session="smoke-s1", top_n=3)
ranks = []
for i, r in enumerate(res):
    print("rank%d [%s] raw=%s" % (i + 1, r.get("kind") or "summary", (r.get("raw") or "")[:40]), flush=True)
    if "sapiens" in (r.get("raw") or "").lower():
        ranks.append(i + 1)
assert ranks, "GATE2b FAIL: 原句通道未命中(词面零重叠场景)"
print("raw-channel hit at rank:", ranks, flush=True)

seen = {}
for r in res:
    k = r.get("raw_of") or r.get("memory_id")
    assert k not in seen, "GATE3 FAIL: 同源记录重复在榜 %s" % k
    seen[k] = True
print("dedup ok, meta.raw_dedup_removed=", sessionmem.search(
    "用户最喜欢的书是什么", current_session="smoke-s1", top_n=3)["meta"].get("raw_dedup_removed"), flush=True)
print("GATE1_PASS GATE2_PASS GATE3_PASS SMOKE_ALL_PASS", flush=True)
