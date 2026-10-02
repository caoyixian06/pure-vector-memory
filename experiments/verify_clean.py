# -*- coding: utf-8 -*-
"""清理后验证: state计数 + 旧自命中查询实测 + ask不增量证明。"""
import json
import urllib.request

MCP = "http://<WORKSTATION_IP>:8820/mcp"
_cid = [100]


def call(name, args):
    _cid[0] += 1
    req = {"jsonrpc": "2.0", "id": _cid[0], "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    r = urllib.request.Request(MCP, json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        d = json.loads(resp.read())
    res = d["result"]
    raw = res["content"][0]["text"]
    if name == "memory_ask":
        return raw, res.get("_meta", {})      # ask 返回prompt文本
    return json.loads(raw), res.get("_meta", {})


def state(space):
    st, _ = call("memory_state", {"space": space})
    return st


for sp in ("profile_all", "default"):
    st = state(sp)
    print(f"STATE {sp}: archive_size={st['archive_size']} "
          f"env_profile={st['env_profile']}")

st1 = state("profile_all")
q = "他撒过谎吗"
r, meta = call("memory_ask", {"request_id": "verify-clean-1", "text": q,
                              "space": "profile_all"})
st2 = state("profile_all")
print(f"\nASK[{q}] evidence_ids={meta.get('evidence')}")
print(f"ASK[{q}] size_before={st1['archive_size']} "
      f"size_after={st2['archive_size']} "
      f"delta={st2['archive_size'] - st1['archive_size']}")
print("--- prompt 证据行 ---")
for line in r["prompt"].splitlines():
    if line.startswith("[observed_order") or line.startswith("[经验") \
            or line.startswith("[候选"):
        print(line[:150])
print("--- prompt 末尾 ---")
print(r["prompt"][-120:])

q2 = "他的财务状况怎么样"
r2, meta2 = call("memory_ask", {"request_id": "verify-clean-2", "text": q2,
                                "space": "profile_all"})
st3 = state("profile_all")
print(f"\nASK[{q2}] evidence_ids={meta2.get('evidence')}")
print(f"ASK[{q2}] size_after={st3['archive_size']} "
      f"delta={st3['archive_size'] - st2['archive_size']}")
for line in r2["prompt"].splitlines():
    if line.startswith("[observed_order"):
        print(line[:150])
print("VERIFY_DONE")
