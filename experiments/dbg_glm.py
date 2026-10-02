# -*- coding: utf-8 -*-
"""dbg_glm.py — 单题伪标注debug: 打印GLM原始返回"""
import io, json, os, re, sys, time, hashlib
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k)
KEY = "${GLM_KEY}"
API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)

q = d[0]
gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
rows = []
for s2 in sorted(gold_h):
    for i in SID2ROWS[s2]:
        if not RAWS[i].startswith("[hdr") and len(RAWS[i]) > 30:
            rows.append(i)
rows = rows[:60]
ans_raw = q.get("answer")
ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
recs = "\n".join("[%d] %s" % (n, RAWS[i][:300]) for n, i in enumerate(rows))
prompt = ("Question: %s\nStandard answer: %s\n\nBelow are numbered records from a conversation. "
          "Select the record numbers that contain the information needed for the answer "
          "(may be multiple; pick only those directly containing answer content).\n\n%s\n\n"
          "Output ONLY the number list like [0,3,7]. If none, output [].") % (
    q["question"], ans_txt[:200], recs)
print("PROMPT len=%d rows=%d" % (len(prompt), len(rows)))
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.1, "max_tokens": 200}).encode()
req = urllib.request.Request(API, data=body,
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        out = json.loads(r.read())
    print("RAW CONTENT:", repr(out["choices"][0]["message"]["content"][:500]))
    print("finish:", out["choices"][0].get("finish_reason"))
except Exception as e:
    print("ERR:", str(e)[:300])
