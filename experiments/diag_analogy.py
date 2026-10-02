# -*- coding: utf-8 -*-
"""diag_analogy.py — 向量算术(king-man+woman)三连测
A 库内问句→下一句方向一致性 + 拆半迁移(类比性是否存在)
B 类比检索: 问题→最近问句turn→其下一句加权, 证据排名代理指标
C 人物方向投影: 证据vs噪声分离AUC
"""
import io, json, os, sys, re
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
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]
Q = [r["question"] for r in rows]
n = len(rows)
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
bQ = emb(Q)

# ---- A: 库内问答对方向 ----
qa_pairs = []
for j in range(N - 1):
    if KIND[j] != "raw" or CONVKEY[j + 1] != CONVKEY[j]:
        continue
    t = TEXTS[j].rstrip()
    if t.endswith("?") and len(t) > 25:
        qa_pairs.append((j, j + 1))
print("库内问句→下一句对:", len(qa_pairs), flush=True)
dvs = np.stack([D[b] - D[a] for a, b in qa_pairs])
dbar = dvs.mean(0)
dbar /= np.linalg.norm(dbar)
rng = np.random.default_rng(5)
perm = rng.permutation(len(dvs))
cos_own = np.array([float(dvs[i] @ dbar) for i in perm[:500]])
# 随机对照: 打乱的(问句,别人家的下一句)方向
rand_d = []
idx = rng.permutation(len(qa_pairs))
for k in range(500):
    a, _ = qa_pairs[idx[k]]
    _, b2 = qa_pairs[idx[(k + 7) % len(qa_pairs)]]
    if a != b2:
        rand_d.append(D[b2] - D[a])
rand_d = np.stack(rand_d)
cos_rand = np.array([float(v / np.linalg.norm(v) @ dbar) for v in rand_d])
print("A1 方向一致性: 自家对cos(d_i, d̄)=%.3f vs 随机拼对=%.3f" % (cos_own.mean(), cos_rand.mean()), flush=True)
# 拆半迁移: 用A半对求d̄, 施在B半: cos(next, norm(q+d̄)) vs cos(next, q)
half = len(qa_pairs) // 2
dbarA = dvs[:half].mean(0); dbarA /= np.linalg.norm(dbarA)
imp, base = [], []
for a, b in qa_pairs[half:half + 500]:
    qa_ = D[a] + dbarA
    qa_ /= np.linalg.norm(qa_)
    imp.append(float(D[b] @ qa_))
    base.append(float(D[b] @ D[a]))
imp, base = np.array(imp), np.array(base)
print("A2 拆半迁移: 加方向后cos %.3f→%.3f (%.0f%%的对改善)" % (
    base.mean(), imp.mean(), 100 * (imp > base).mean()), flush=True)

# ---- B: 类比检索(问题→最近问句→其下一句加权) ----
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
targets = {}
for i, r in enumerate(rows):
    sid = r.get("sample_id") or ("conv-" + str(r.get("conversation_idx")))
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in r.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        targets[i] = hits
print("matched:", len(targets), flush=True)

qidx = [a for a, b in qa_pairs]
Qmat = D[qidx]
sims = bQ @ Qmat.T  # (n, n_qa)
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]

def armB_order(i):
    top = TOP50[i]
    inside = np.argsort(-RER[i])
    return top, inside

def eval_blend(w_ana, topk_q=3, tag=""):
    ok_arr = np.array([rows[i].get("llm_score") == 1 for i, _ in targets.items()])
    keys = list(targets.keys())
    m25 = o5 = 0
    hit_adj = 0
    for i in keys:
        hits = set(targets[i])
        adj = set()
        for h in hits:
            for dd in (1, -1):
                k2 = h + dd
                if 0 <= k2 < N and CONVKEY[k2] == CONVKEY[h]:
                    adj.add(k2)
        top, inside = armB_order(i)
        s = np.zeros(N, dtype=np.float32)
        s[top[inside]] = np.linspace(50, 1, 50)
        # 类比加分: 最近k个问句的下一句
        nn = np.argsort(-sims[i])[:topk_q]
        for r0, qt in enumerate(nn):
            nxt = qa_pairs[qidx.index(qt) if False else [a for a, b in qa_pairs].index(qt)][1] if False else None
        # 直接用 qidx->pair 映射
        for r0, qt_idx in enumerate(nn):
            a, b2 = qa_pairs[qt_idx]
            w = float(sims[i][qt_idx]) * (1.0 - r0 / topk_q)
            s[b2] += w_ana * w * 10.0
        order = np.argsort(-s)
        pos = {x: p for p, x in enumerate(order)}
        rk = min(pos[h] for h in hits) + 1
        if rk <= 25 and rows[i].get("llm_score") != 1:
            m25 += 1
        if rk <= 5 and rows[i].get("llm_score") == 1:
            o5 += 1
        nn1 = np.argmax(sims[i])
        a1, b1 = qa_pairs[nn1]
        if b1 in hits or b1 in adj:
            hit_adj += 1
    print("  %-22s 错题进25 %d 对题进5 %d | 最近问句的下一句命中证据/邻句: %d/%d(%.0f%%)" % (
        tag, m25, o5, hit_adj, len(keys), 100 * hit_adj / len(keys)), flush=True)

print("== B 类比检索(叠在臂B排序上) ==", flush=True)
eval_blend(0.0, tag="基线(臂B排序)")
eval_blend(0.5, 1, "类比w=0.5 top1问句")
eval_blend(0.5, 3, "类比w=0.5 top3问句")
eval_blend(1.0, 3, "类比w=1.0 top3问句")

# ---- C: 人物方向 ----
names = sorted({str(r.get("speaker_a")) for r in rows if r.get("speaker_a")} |
               {str(r.get("speaker_b")) for r in rows if r.get("speaker_b")})
global_c = D[[i for i in range(N) if KIND[i] == "raw"]].mean(0)
PD = {}
for nm in names:
    ids = [i for i in range(N) if KIND[i] == "raw" and TEXTS[i].startswith(nm + ":")]
    if len(ids) >= 20:
        PD[nm] = D[ids].mean(0) - global_c
print("C 人物方向可建:", len(PD), "人", flush=True)
COUR = re.compile(r"\b(thank|thanks|congrat|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
fv, lab = [], []
for i in range(n):
    if i not in targets:
        continue
    nm = next((x for x in PD if re.search(r"\b%s\b" % re.escape(x), Q[i])), None)
    if not nm:
        continue
    hits = set(targets[i])
    tw = {}
    for h in hits:
        t2 = REC.get(MID[h], {}).get("raw_of")
        if t2 in MID2I:
            tw[MID2I[t2]] = 1
    for j in TOP50[i]:
        is_ev = j in hits or j in tw
        proj = float(D[j] @ (PD[nm] / (np.linalg.norm(PD[nm]) + 1e-9)))
        fv.append(proj)
        lab.append(is_ev)
fv = np.array(fv); lab = np.array(lab)
if lab.sum() > 20:
    def auc(x, p, q):
        r = np.argsort(np.concatenate([x[p], x[q]]))
    # 简化AUC
    pos = fv[lab]; neg = fv[~lab]
    r = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(r, dtype=np.float64); ranks[r] = np.arange(1, len(r) + 1)
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print("C 人物方向投影: EVID均值%.3f NOISE均值%.3f AUC=%.3f (n_ev=%d)" % (
        pos.mean(), neg.mean(), a, len(pos)), flush=True)
