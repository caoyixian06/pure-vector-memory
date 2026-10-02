# -*- coding: utf-8 -*-
"""weak_retrieval.py — 弱关联检索 + 多句证据关联性
Q1: 弱关联(闭合差+Δ+几何)做top50重排 → 排名质量变化(50错+50对)
Q2: 多句证据的内部关联: 共享独有词率/相邻率/同session率/向量互cos, vs 噪声对照组
"""
import io, json, os, sys, re, random
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
N = len(MID)
MID2I = {m: i for i, m in enumerate(MID)}
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if hits:
        targets[i] = hits
keys = sorted(targets.keys())
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evs)].mean(0) - D[np.array(nos)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ_D = (D @ DELTA).astype(np.float32)
bQ = emb([r["question"] for r in rows])

# ===== Q1: 弱关联重排(50错+50对) =====
print("== Q1 弱关联重排(50错+50对) ==", flush=True)
rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
X = emb([str(r["answer"][0]) for r in rows])

def eval_order(order_fn, sel_list, tag):
    m25 = o5 = cnt = 0
    for i in sel_list:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        allh = list(hs) + list(tws)
        if not all(h2 in pos for h2 in allh):
            continue
        rk = min(pos[h2] for h2 in allh) + 1
        cnt += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-26s 错题进25 %2d/50  对题进5 %2d/50" % (tag, m25, o5), flush=True)

def base_order(i):
    row_f = zs(SBS[i]) + zs(SQc[i]) + 0.5 * zs(VEXc[i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return newf

raw_idx = [i2 for i2 in range(N) if KIND[i2] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc0 = bQ.mean(0); qc0 /= np.linalg.norm(qc0)
u2 = stmt - qc0
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc = zc["SQ"], zc["VEX"]


def weak_order(i, w=1.0):
    # 弱关联重排: 闭合差(x用X[i]真答案=上界) + Δ
    # 在线上x未知——这里用"闭合差上界"测该信号天花板
    row = base_order(i)
    a = bQ[i]
    x = X[i]
    top = np.argsort(-row)[:50]
    newscore = []
    for j in top:
        c = D[j]
        ch = c - a
        t = (x - a) @ ch / (ch @ ch + 1e-9)
        dline = float(np.linalg.norm(x - (a + t * ch)))
        sc = -dline + w * float(c @ DELTA)
        newscore.append(sc)
    newscore = np.array(newscore)
    newf = row.copy()
    newf[top] = row[top] + w * 3.0 * zs(newscore)
    return np.argsort(-newf)

eval_order(lambda i: base_order(i), sel_miss + sel_ok, "基线r37")
eval_order(lambda i: weak_order(i, 1.0), sel_miss + sel_ok, "弱关联重排w=1(闭合差上界)")
eval_order(lambda i: weak_order(i, 2.0), sel_miss + sel_ok, "弱关联重排w=2")

# ===== Q2: 多句证据内部关联 vs 噪声 =====
print()
print("== Q2 多句证据内部关联 vs 噪声组 ==", flush=True)
import itertools
shared_word = {"ev": [], "no": []}
adj_rate = {"ev": [], "no": []}
sess_same = {"ev": [], "no": []}
cos_pair = {"ev": [], "no": []}
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    allh = hs | tws
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws]
    if len(ev) < 2 or len(no) < 2:
        continue
    # 证据组内部: 共享词率(词干交集/并集), 相邻率(CONVKEY相邻), 同session率, 向量互cos
    for (grp, tag) in ((ev, "ev"), (no[:2], "no")):
        for a2, b2 in itertools.combinations(grp, 2):
            ta, tb = toks_set(TEXTS[a2]), toks_set(TEXTS[b2])
            jac = len(ta & tb) / max(1, len(ta | tb))
            shared_word[tag].append(jac)
            adj_rate[tag].append(1.0 if CONVKEY[a2] == CONVKEY[b2] else 0.0)
            cos_pair[tag].append(float(D[a2] @ D[b2]))
            sa = MID[a2].rsplit("_", 1)[0]
            sb = MID[b2].rsplit("_", 1)[0]
            sess_same[tag].append(1.0 if sa == sb else 0.0)
        break  # 每题只取第一对
for tag in ("ev", "no"):
    print("  %s组: 词Jaccard=%.3f 相邻率=%.0f%% 同session率=%.0f%% 互cos=%.3f" % (
        tag.upper(), np.mean(shared_word[tag]), 100 * np.mean(adj_rate[tag]),
        100 * np.mean(sess_same[tag]), np.mean(cos_pair[tag])), flush=True)
print("WEAK_DONE", flush=True)
