# -*- coding: utf-8 -*-
"""catch_real.py — 宿主链联合信号: 上界测试
J = ①问题内容词宿主(a侧) ∧ ②答案预期词宿主(x侧, 离线用真独有词=上界)
测: 单①(词票本质)/单②(覆盖)/联合①∧② 三者判别力(证据vs噪声, 配对池)
若联合>98%(超覆盖单挑) → 宿主链=真身证明
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
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
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
def toks_set(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

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

win1 = win2 = winJ = tot = 0
a1, a2, aJ = [], [], []
for i in keys:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    # ①问题内容词宿主(词干级)
    qst = set(stem(w) for w in toks_set(r["question"]))
    def host_q(j):
        t = set(stem(w) for w in toks_set(TEXTS[j]))
        return len(qst & t)
    # ②答案预期词宿主(真独有词, 词干级)
    aw = set(stem(w) for w in toks_set(str(r["answer"][0])))
    qw_raw = set(stem(w) for w in toks_set(r["question"]))
    expect = aw - qw_raw
    def host_x(j):
        if not expect:
            return 0
        t = set(stem(w) for w in toks_set(TEXTS[j]))
        return len(expect & t)
    h1E, h1N = host_q(ev[0]), host_q(no[0])
    h2E, h2N = host_x(ev[0]), host_x(no[0])
    # 联合: 双条件满足(≥1词宿主 且 ≥1预期词宿主)
    jE = 1 if (h1E >= 1 and h2E >= 1) else 0
    jN = 1 if (h1N >= 1 and h2N >= 1) else 0
    tot += 1
    if h1E > h1N:
        win1 += 1
    if h2E > h2N:
        win2 += 1
    if jE > jN:
        winJ += 1
    a1.append(h1E - h1N)
    a2.append(h2E - h2N)
    aJ.append(jE - jN)
print("配对样本:", tot, flush=True)
print("①问题词宿主差:      胜率=%.0f%%" % (100 * win1 / tot), flush=True)
print("②答案预期词宿主差:  胜率=%.0f%% (开卷上界)" % (100 * win2 / tot), flush=True)
print("①∧②联合宿主链:     胜率=%.0f%%  ← 若>②则真身=宿主链" % (100 * winJ / tot), flush=True)
# 联合的另一种形式: 和
winS = sum(1 for k in range(tot) if (a1[k] + 2 * a2[k]) > 0)
print("①+2×②加权和:       胜率=%.0f%%" % (100 * winS / tot), flush=True)
# 分桶: 联合在枚举题上的表现
we = wne = tote = 0
for k, i in enumerate(keys):
    r = rows[i]
    ans = str(r["answer"][0])
    if ("," not in ans) and (" and " not in ans.lower()):
        continue
    tote += 1
print("(枚举题占比见上轮, 联合优势若稳定则跨题型)", flush=True)
print("REALDONE", flush=True)
