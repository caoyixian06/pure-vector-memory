# -*- coding: utf-8 -*-
"""show_cluster.py — 拉出真实证据簇的原文实物
取3道多证据枚举题: 打印题目/金答案/证据簇原文/簇的四项指纹数字
"""
import io, json, os, sys, re, itertools
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

def TOP50_of(i):
    return TOP50[i]

# 找3道多证据枚举题(证据≥2, 答案是列表)
shown = 0
for i in keys:
    r = rows[i]
    ans = str(r["answer"][0])
    if ("," not in ans) and (" and " not in ans.lower()):
        continue
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    if len(hs | tws) < 2:
        continue
    shown += 1
    print("========== 例%d ==========" % shown, flush=True)
    print("题目:", r["question"], flush=True)
    print("金答案:", ans[:100], flush=True)
    print("证据簇原文(含孪生定位):", flush=True)
    ev_sorted = sorted(hs | tws)
    pairs = list(itertools.combinations(sorted(hs | tws), 2))
    for j in ev_sorted:
        spk = TEXTS[j].split(":")[0] if ":" in TEXTS[j][:30] else "?"
        print("  [%s] %s" % (spk, TEXTS[j][:110]), flush=True)
    # 四项指纹
    for a2, b2 in pairs:
        ta, tb = toks_set(TEXTS[a2]), toks_set(TEXTS[b2])
        jac = len(ta & tb) / max(1, len(ta | tb))
        adj = CONVKEY[a2] == CONVKEY[b2]
        cs = float(D[a2] @ D[b2])
        same_s = MID[a2].rsplit("_", 1)[0] == MID[b2].rsplit("_", 1)[0]
        print("    指纹对: 词J=%.2f 相邻=%s 同sess=%s 互cos=%.3f" % (jac, adj, same_s, cs), flush=True)
    # 与对照组(同池噪声)对比一个
    no = [j for j in TOP50[i] if j not in hs and j not in tws]
    if no:
        ta, tb = toks_set(TEXTS[ev_sorted[0]]), toks_set(TEXTS[no[0]])
        jac = len(ta & tb) / max(1, len(ta | tb))
        cs = float(D[ev_sorted[0]] @ D[no[0]])
        print("    [对照] 证据↔池内噪声: 词J=%.2f 互cos=%.3f" % (jac, cs), flush=True)
    if shown >= 3:
        break


