# -*- coding: utf-8 -*-
"""foundry111.py — 同一条公式找turn级金证据(两基准):
全库记录cos排序 -> top记录=金句候选
验证(只看对错): LME=答案词干出现在top-k记录文本? LoCoMo=top-k命中金记录集合?"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry111_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)

t0 = time.time()
# ================= LME =================
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
P("LME loaded %.0fs" % (time.time() - t0))
lme = {1: 0, 3: 0, 5: 0}
nl = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = np.array([i for i, s2 in enumerate(SIDS) if s2 in hay_keys and NOHDR[i]])
    cq = D[domain] @ X[qi]
    order = domain[np.argsort(-cq)]
    ans_raw = q.get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    if not ans_stems:
        continue
    nl += 1
    for k2 in (1, 3, 5):
        hit = False
        for i in order[:k2]:
            rs = stems_of(RAWS[i])
            if len(ans_stems & rs) / len(ans_stems) >= 0.5:
                hit = True
                break
        if hit:
            lme[k2] += 1
    if qi % 100 == 0:
        P("  LME %d %.0fs" % (qi, time.time() - t0))
P("\n===== LME turn级(同一条cos公式, 验证=答案词干命中) =====")
P("top1含答案=%.1f%%  top3=%.1f%%  top5=%.1f%%  (n=%d)" % (
    100.0 * lme[1] / nl, 100.0 * lme[3] / nl, 100.0 * lme[5] / nl, nl))

# ================= LoCoMo =================
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
RAWN_C = [norm(x) for x in RAWS_C]
D_C = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q_C = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q_C[q["qa_id"]] = q
X_C = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
P("LoCoMo loaded %.0fs" % (time.time() - t0))
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q_C[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN_C[i] for k in keys))
lc = {1: 0, 3: 0, 5: 0}
nlc = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    nlc += 1
    cq = D_C @ X_C[k_i]
    order = np.argsort(-cq)
    for k2 in (1, 3, 5):
        if G & set(order[:k2].tolist()):
            lc[k2] += 1
    if k_i % 400 == 0:
        P("  LC %d %.0fs" % (k_i, time.time() - t0))
P("\n===== LoCoMo turn级(同一条cos公式, 验证=金记录命中) =====")
P("top1含金=%.1f%%  top3=%.1f%%  top5=%.1f%%  (n=%d)" % (
    100.0 * lc[1] / nlc, 100.0 * lc[3] / nlc, 100.0 * lc[5] / nlc, nlc))
P("对照: LoCoMo ranker(带训练) 核心@5=78.2")
P("F111_DONE %.0fs" % (time.time() - t0))
