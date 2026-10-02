# -*- coding: utf-8 -*-
"""diag_delta_anatomy.py — Δ的解剖: 它由什么成分构成?
D1 Δ的top维度上, 证据/噪声文本的词面差异(哪些词在证据侧高频)
D2 Δ与候选已知方向的相关性: 语域轴/日期轴/人名轴/长度轴/问句轴/肯定词轴
D3 Δ投影与文本特征的偏相关(控制精排分后还剩多少)
D4 说话人结构: 证据是否更可能是'轮到自己说话'的记录(对话结构信号)
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
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

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
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

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
TOP50 = z1["TOP50"]

evs, nos = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evs += [j for j in top if j in hs or j in tws]
    nos += [j for j in top if j not in hs and j not in tws][:8]
evs, nos = np.array(evs), np.array(nos)
DELTA = D[evs].mean(0) - D[nos].mean(0)
DELTA /= np.linalg.norm(DELTA)
print("pool: ev%d no%d" % (len(evs), len(nos)), flush=True)

# D2: 与候选轴的相关
def axis_of(texts_fn, name):
    texts = [texts_fn(k) for k in range(60)]
    V = emb(texts)
    a = V.mean(0) - emb([texts_fn(1000 + k) for k in range(60)]).mean(0)
    a /= np.linalg.norm(a) + 1e-9
    return float(DELTA @ a)

questions = [r["question"] for r in rows[:2000]]
qc = emb(questions).mean(0); qc /= np.linalg.norm(qc)
print("D2 相关轴:", flush=True)
print("  问题质心方向      : %+.3f" % float(DELTA @ qc))
stmt_c = D[[i for i in range(N) if KIND[i] == "raw"][:3000]].mean(0)
stmt_c /= np.linalg.norm(stmt_c)
print("  陈述质心方向      : %+.3f" % float(DELTA @ stmt_c))
# 长度轴: 长文本vs短文本
long_t = ["word " * 200] * 30
short_t = ["hi."] * 30
la = emb(long_t).mean(0) - emb(short_t).mean(0)
la /= np.linalg.norm(la)
print("  长度轴            : %+.3f" % float(DELTA @ la))
# 过去时/肯定叙述轴
past = ["I " + v + " it yesterday." for v in ["finished", "completed", "did", "made", "went", "saw", "started", "bought", "moved", "visited"] * 3]
fut = ["I will " + v + " it tomorrow." for v in ["finish", "complete", "do", "make", "go", "see", "start", "buy", "move", "visit"] * 3]
pa = emb(past).mean(0) - emb(fut).mean(0)
pa /= np.linalg.norm(pa)
print("  既成事实vs将来轴  : %+.3f" % float(DELTA @ pa))
# 信息密度轴: 具体名词句 vs 空泛寒暄
info = ["My favorite book is The Pragmatic Programmer by Hunt.", "We visited Kyoto in April 2019 for cherry blossoms.", "She works as a pediatric nurse at the children hospital.", "The concert tickets cost 45 dollars each last Friday.", "My brother graduated from MIT with computer science degree."] * 6
fluff = ["That sounds really great and awesome!", "Thanks so much for your support friend!", "Wow I cannot believe it amazing!", "Have a wonderful day and take care!", "It was so much fun hanging out together!"] * 6
ia = emb(info).mean(0) - emb(fluff).mean(0)
ia /= np.linalg.norm(ia)
print("  信息密度轴        : %+.3f" % float(DELTA @ ia))

# D1: 词面差异(证据侧高频内容词)
STOP = set("i you he she it we they me him her us them my your his its our their a an the is are was were be been being do does did have has had will would can could should to of in on at for with about from by as and or but if so not no that this these those there it's i'm don't didn't what when where who how why s t re ve ll d m just really very much more most some any all".split())
def toks(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if w not in STOP and len(w) > 2]
from collections import Counter
ce, cn = Counter(), Counter()
for j in evs[:2500]:
    ce.update(toks(TEXTS[j]))
for j in nos[:2500]:
    cn.update(toks(TEXTS[j]))
diff = []
for w in set(list(ce.keys())[:4000]):
    r_e = ce[w] / max(1, sum(ce.values()))
    r_n = cn[w] / max(1, sum(cn.values()))
    if r_e + r_n > 1e-5:
        diff.append((r_e - r_n, w))
diff.sort(reverse=True)
print("D1 证据侧高频词top20:", [w for _, w in diff[:20]], flush=True)
print("   噪声侧高频词top15:", [w for _, w in diff[-15:]], flush=True)

# D4: 对话结构 - 证据的说话人是否= 题目主角之一
sp_ev = sp_no = 0
for i in keys:
    hs = targets[i]
    names = [str(rows[i].get("speaker_a")), str(rows[i].get("speaker_b"))]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    for j in top:
        if j not in hs and j not in tws:
            continue
        t = TEXTS[j]
        if any(t.startswith(nm + ":") for nm in names if nm != "None"):
            sp_ev += 1
    cnt = 0
    for j in top:
        if j in hs or j in tws:
            continue
        t = TEXTS[j]
        if cnt >= 8:
            break
        cnt += 1
        if any(t.startswith(nm + ":") for nm in names if nm != "None"):
            sp_no += 1
print("D4 证据以主角开场: %.0f%% | 噪声以主角开场: %.0f%%" % (
    100 * sp_ev / max(1, sp_ev + 0), 100 * sp_no / max(1, sp_no)), flush=True)
