# -*- coding: utf-8 -*-
"""foundry119a.py — LLM伪标注(A半250题): 金会话记录→GLM选出含答案的编号→turn级伪标签"""
import io, json, os, re, sys, time, hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k)
KEY = "${GLM_KEY}"
API = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LOG = io.open("C:/locomo_refined/foundry119a_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

t0 = time.time()
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
NOHDR = np_not = None
import numpy as np
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
def hay_key(q):
    return hashlib.md5(json.dumps(sorted(q["haystack_session_ids"]), ensure_ascii=False).encode()).hexdigest()
hk = {}
for qi, q in enumerate(d):
    hk.setdefault(hay_key(q), []).append(qi)
keys = sorted(hk)
fd = {k: i % 2 for i, k in enumerate(keys)}
qfold = np.array([fd[hay_key(q)] for q in d])
P("loaded %.0fs" % (time.time() - t0))

def glm_call(prompt, retries=5):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [
        {"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 200}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(API, data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + KEY})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
                return out["choices"][0]["message"]["content"]
        except Exception as e:
            if a == retries - 1:
                return ""
            time.sleep(6 * (a + 1))

def annotate(qi):
    q = d[qi]
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    rows = []
    for s2 in sorted(gold_h):
        for i in SID2ROWS[s2]:
            if NOHDR[i] and len(RAWS[i]) > 30:
                rows.append(i)
    if not rows:
        return None
    if len(rows) > 60:
        rows = rows[:60]
    ans_raw = q.get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    recs = "\n".join("[%d] %s" % (n, RAWS[i][:300]) for n, i in enumerate(rows))
    prompt = ("Question: %s\nStandard answer: %s\n\nBelow are numbered records from a conversation. "
              "Select the record numbers that contain the information needed for the answer "
              "(may be multiple; pick only those directly containing answer content).\n\n%s\n\n"
              "Output ONLY the number list like [0,3,7]. If none, output [].") % (
        q["question"], ans_txt[:200], recs)
    out = glm_call(prompt)
    if not out:
        return None
    m = re.findall(r"\d+", out)
    sel = [rows[int(x)] for x in m if int(x) < len(rows)]
    return {"qi": qi, "rows": [int(x) for x in rows], "sel": sel}

tasks = [qi for qi in range(len(d)) if qfold[qi] == 0]
P("A半=%d题 待伪标注" % len(tasks))
results = []
done = 0
with ThreadPoolExecutor(max_workers=2) as ex:
    for r in ex.map(annotate, tasks):
        done += 1
        if r is not None and r["sel"]:
            results.append(r)
        if done % 25 == 0:
            P("  %d/%d %.0fs sel均=%.1f" % (
                done, len(tasks), time.time() - t0,
                np.mean([len(r["sel"]) for r in results]) if results else 0))

json.dump(results, open("C:/locomo_refined/lme_pseudo_labels_A.json", "w"))
P("\n伪标注完成: %d/%d题有标签, 平均每题%.1f条选中 %.0fs" % (
    len(results), len(tasks), np.mean([len(r["sel"]) for r in results]), time.time() - t0))
P("落盘: lme_pseudo_labels_A.json")
P("F119A_DONE %.0fs" % (time.time() - t0))
