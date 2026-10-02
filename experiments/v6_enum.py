# -*- coding: utf-8 -*-
"""v6_enum.py — 枚举层信号扫描 + 融合显式化调优
E1 枚举题(518)信号谱系扫描: 每个已知信号在'枚举题的证据vs噪声'上的判别AUC
   信号: Δ投影 / 精排分 / bge-u2 / qwen / 词票 / 首提 / 会话头 / raw先验 / 占座(负) / 词覆盖(题宾语) / 兄弟相似(与top1)
E2 融合显式化: 融合各通道权重在枚举/非枚举子集分开调(粗网格, 拆半), 对照全局统一权重
E3 枚举题专用信号组合: 找到枚举层最强2信号, 合成枚举专用排序, 全量合成验证
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
rows_all = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))]
rows = [r for r in rows_all if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

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
N = len(MID)
MID2I = {m: i for i, m in enumerate(MID)}
RAWFLAG = np.array([1.0 if KIND[i] == "raw" else 0.0 for i in range(N)], dtype=np.float32)
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
FIRST = np.zeros(N, dtype=np.float32)
SESSPOS = np.zeros(N, dtype=np.float32)
seen = {}
for sess, ids in CONV.items():
    ids_sorted = sorted(ids, key=lambda j: MID[j])
    for pos, j in enumerate(ids_sorted):
        if KIND[j] != "raw":
            continue
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", TEXTS[j])
        nf = 0
        for nm in names:
            if nm not in seen.get(sess, set()):
                seen.setdefault(sess, set()).add(nm)
                nf = 1
        FIRST[j] = nf
        SESSPOS[j] = pos / max(1, len(ids_sorted))
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
SQc, VEXc, SBc = zc["SQ"], zc["VEX"], zc["SB"]
bQ = emb([r["question"] for r in rows])
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBSu = (l2n(bQ + u2) @ D.T).astype(np.float32)
PROJ_D = None

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

def is_enum(r):
    return ("," in str(r["answer"][0])) or (" and " in str(r["answer"][0]).lower())

# Δ投影
evA, noA = [], []
for i in keys[:len(keys) // 2]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evA += [j for j in top if j in hs or j in tws]
    noA += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evA)].mean(0) - D[np.array(noA)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ_D = (D @ DELTA).astype(np.float32)

def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def auc(x, plab):
    x = np.asarray(x, dtype=np.float64)
    plab = np.asarray(plab, dtype=bool)
    r = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[r] = np.arange(1, len(x) + 1)
    np_ = plab.sum()
    return (ranks[plab].sum() - np_ * (np_ + 1) / 2) / (np_ * (~plab).sum())

# E1: 枚举题信号谱系
print("== E1 枚举题信号谱系(证据vs噪声, 池内配对) ==", flush=True)
COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
sig = {k: ([], []) for k in ("Δ投影", "精排", "bge_u2", "qwen", "词票", "首提", "会话头早", "raw", "占座(负)", "近top1", "词覆盖(题宾语)")}
wins = {k: 0 for k in sig}
tot = 0
for i in keys:
    if not is_enum(rows[i]):
        continue
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i][np.argsort(-RER[i])]
    ev = [j for j in top if j in hs or j in tws]
    no = [j for j in top if j not in hs and j not in tws][:1]
    if not ev or not no:
        continue
    ce, cn = ev[0], no[0]
    qw = set(w[:4] for w in re.findall(r"[a-z']+", rows[i]["question"].lower()) if len(w) > 3)
    for k in sig:
        if k == "Δ投影":
            ve, vn = PROJ_D[ce], PROJ_D[cn]
        elif k == "精排":
            ve = RER[i][list(TOP50[i]).index(ce)] if ce in TOP50[i] else -7
            vn = RER[i][list(TOP50[i]).index(cn)] if cn in TOP50[i] else -7
        elif k == "bge_u2":
            ve, vn = SBSu[i][ce], SBSu[i][cn]
        elif k == "qwen":
            ve, vn = SQc[i][ce], SQc[i][cn]
        elif k == "词票":
            ve, vn = VEXc[i][ce], VEXc[i][cn]
        elif k == "首提":
            ve, vn = FIRST[ce], FIRST[cn]
        elif k == "会话头早":
            ve, vn = 1 - SESSPOS[ce], 1 - SESSPOS[cn]
        elif k == "raw":
            ve, vn = RAWFLAG[ce], RAWFLAG[cn]
        elif k == "占座(负)":
            te = TEXTS[ce]
            tn = TEXTS[cn]
            ve = -(1.0 if (te.rstrip().endswith("?") or COUR.search(te)) else 0.0)
            vn = -(1.0 if (tn.rstrip().endswith("?") or COUR.search(tn)) else 0.0)
        elif k == "近top1":
            ve, vn = float(D[ce] @ D[top[0]]), float(D[cn] @ D[top[0]])
        elif k == "词覆盖(题宾语)":
            ve = sum(1 for w in qw if w in TEXTS[ce].lower())
            vn = sum(1 for w in qw if w in TEXTS[cn].lower())
        sig[k][0].append(float(ve))
        sig[k][1].append(float(vn))
    tot += 1
print("枚举配对样本:", tot, flush=True)
for k, (e, n2) in sig.items():
    e, n2 = np.array(e), np.array(n2)
    w = int((e > n2).sum())
    x = np.concatenate([e, n2]); lb = np.zeros(len(x), dtype=bool); lb[:len(e)] = True
    print("  %-12s 胜率%.0f%% AUC=%.3f" % (k, 100 * w / tot, auc(x, lb)), flush=True)

# E2: 融合显式化调优(枚举/非枚举分开权重, 拆半)
print("== E2 融合显式化(分层权重, 拆半) ==", flush=True)
enum_keys = [i for i in keys if is_enum(rows[i])]
non_keys = [i for i in keys if not is_enum(rows[i])]
hE = len(enum_keys) // 2
hN = len(non_keys) // 2
def ev_sel(wBGE, wQ, wV, sel, tag):
    m25 = o5 = cnt = 0
    for i in sel:
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        row = wBGE * zs(SBSu[i]) + wQ * zs(SQc[i]) + wV * zs(VEXc[i])
        order = np.argsort(-row)
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
    print("  %-34s 错题进25 %3d  对题进5 %3d (n=%d)" % (tag, m25, o5, cnt), flush=True)
    return m25, o5
# 枚举子集: A半调权重
grid = [(1, 1, 0.5), (1, 1.5, 0.5), (1, 1, 1.0), (1, 1.5, 1.0), (0.5, 1.5, 0.5), (1, 2, 0.5)]
best = None
for g in grid:
    r = ev_sel(g[0], g[1], g[2], enum_keys[:hE], "枚举A半 w=%s" % (g,))
    if best is None or r[0] + r[1] > best[0]:
        best = (r[0] + r[1], g)
print("枚举A半最优权重:", best[1], flush=True)
gB = best[1]
ev_sel(gB[0], gB[1], gB[2], enum_keys[hE:], "  枚举B半验证")
ev_sel(1, 1, 0.5, enum_keys[hE:], "  枚举B半对照(全局权重)")
# 非枚举子集: 对照
bestN = None
for g in grid:
    r = ev_sel(g[0], g[1], g[2], non_keys[:hN], "非枚举A半 w=%s" % (g,))
    if bestN is None or r[0] + r[1] > bestN[0]:
        bestN = (r[0] + r[1], g)
print("非枚举A半最优权重:", bestN[1], flush=True)
gN = bestN[1]
ev_sel(gN[0], gN[1], gN[2], non_keys[hN:], "  非枚举B半验证")
ev_sel(1, 1, 0.5, non_keys[hN:], "  非枚举B半对照(全局权重)")
print("V6ENUM_DONE", flush=True)
