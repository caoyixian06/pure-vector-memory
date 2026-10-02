# -*- coding: utf-8 -*-
"""foundry113.py — 把文本找出来: 公式端到端展示(问题→找到的原文)"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry113_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

t0 = time.time()
# ================= LME 3题 =================
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
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
P("===== LME: 问题 → 公式找到的文本 =====")
for qi in (0, 5, 20):
    q = d[qi]
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    cq = D[domain] @ X[qi]
    order = domain[np.argsort(-cq)]
    P("\nQ%d: %s" % (qi, q["question"]))
    for r, i in enumerate(order[:5]):
        txt = RAWS[i][:200].replace(chr(10), " ")
        P("  [%d] %s" % (r + 1, txt))

# ================= LoCoMo 3题 =================
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
RAWS_C = [rec_raw(m) for m in MID]
D_C = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q_C = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q_C[q["qa_id"]] = q
X_C = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
P("\n\n===== LoCoMo: 问题 → 公式找到的文本 =====")
for k_i in (0, 100, 500):
    cq = D_C @ X_C[k_i]
    order = np.argsort(-cq)
    P("\nQ: %s" % Q_C[IDS[k_i]]["question"])
    for r, i in enumerate(order[:5]):
        txt = RAWS_C[int(i)][:200].replace(chr(10), " ")
        P("  [%d] %s" % (r + 1, txt))
P("\nF113_DONE %.0fs" % (time.time() - t0))
