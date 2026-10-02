# -*- coding: utf-8 -*-
"""远端巡检: 代码哈希 + memdata 各space记录数 + profile_all id前缀分布 + default全量dump。
输出写到 UTF-8 JSON 文件, 由本地拉回分析, 避开 GBK 控制台。"""
import hashlib
import json
import os
from collections import Counter

ROOT = r"C:\bci_eval_isolated\memory_v2"
MEMDATA = os.path.join(ROOT, "memdata")
OUT = os.path.join(ROOT, "inspect_out.json")

res = {"files": {}, "spaces": {}, "profile_all": {}, "default": {}}

for f in ["memory.py", "mcp_http.py", "mcp_server.py", "test_memory_v2.py",
          "migrate_global.py", "ingest_profile.py", "ingest_v2.py"]:
    p = os.path.join(ROOT, f)
    res["files"][f] = hashlib.sha256(open(p, "rb").read()).hexdigest() \
        if os.path.exists(p) else "MISSING"

for space in sorted(os.listdir(MEMDATA)):
    arc = os.path.join(MEMDATA, space, "archive.jsonl")
    if not os.path.exists(arc):
        res["spaces"][space] = {"archive": False}
        continue
    recs = [json.loads(l) for l in open(arc, encoding="utf-8").read()
            .splitlines() if l.strip()]
    info = {"archive": True, "size": os.path.getsize(arc),
            "n": len(recs), "s_hash_ids": 0}
    for r in recs:
        if str(r.get("id", "")).startswith("s#"):
            info["s_hash_ids"] += 1
    res["spaces"][space] = info

# profile_all 详细: id 前缀分布(去掉#后的序号)
arc = os.path.join(MEMDATA, "profile_all", "archive.jsonl")
recs = [json.loads(l) for l in open(arc, encoding="utf-8").read()
        .splitlines() if l.strip()]
pref = Counter()
for r in recs:
    i = str(r.get("id", ""))
    pref[i.split("#", 1)[0]] += 1
res["profile_all"]["prefix_counter"] = dict(pref)

def sample(rs, k=6):
    return [{"id": r.get("id"), "req": r.get("request_id"),
             "date": r.get("date"), "role": r.get("role"),
             "text": (r.get("text") or "")[:120]} for r in rs[:k]]

res["profile_all"]["sample_s_hash"] = sample(
    [r for r in recs if str(r.get("id", "")).startswith("s#")], 12)
res["profile_all"]["sample_other_rare"] = sample(
    [r for r in recs if not str(r.get("id", "")).startswith("s#")
     and pref[str(r.get("id", "")).split("#", 1)[0]] <= 3], 15)
res["profile_all"]["sample_common"] = sample(
    [r for r in recs if pref[str(r.get("id", "")).split("#", 1)[0]] > 3], 6)
res["profile_all"]["role_counter"] = dict(Counter(
    str(r.get("role")) for r in recs))
res["profile_all"]["has_session_key"] = sum(
    1 for r in recs if "session" in r)
res["profile_all"]["request_id_suspicious"] = sample(
    [r for r in recs if any(w in str(r.get("request_id", "")).lower()
                            for w in ("test", "verify", "smoke", "diag",
                                      "q", "ask", "check"))], 20)

# default 全量
arc = os.path.join(MEMDATA, "default", "archive.jsonl")
if os.path.exists(arc):
    drecs = [json.loads(l) for l in open(arc, encoding="utf-8").read()
             .splitlines() if l.strip()]
    res["default"] = {"n": len(drecs),
                      "records": [{"id": r.get("id"),
                                   "req": r.get("request_id"),
                                   "date": r.get("date"),
                                   "kind": r.get("kind"),
                                   "text": (r.get("text") or "")[:160],
                                   "situation": (r.get("situation") or "")[:80]}
                                  for r in drecs]}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("INSPECT_OK")
