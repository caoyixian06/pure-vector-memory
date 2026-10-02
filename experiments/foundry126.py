# -*- coding: utf-8 -*-
"""foundry126.py — 动态窗口=断崖检测(用户令):
融合排序的分数序列, 最大跌幅处截窗 — 单金题缩窗提纯度, 多金题保持
对照: 固定5(命中65.6/纯度14.5) | 双口径: 纯度+命中"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry126_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
COURT = re.compile(r"\b(hi|hey|hello|thanks|thank you|great|awesome|cool|nice|sure|okay|ok|wow|sounds good|good to know)\b", re.I)

t0 = time.time()
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
RAWS = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAWS]
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
Q = {}
for l in io.open("C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
X = l2n(np.load(HERE + "/xz_cache.npz")["X"].astype(np.float32))
IDS = [str(x) for x in np.load(HERE + "/r40bf_ckpt.npz")["IDS"]]
CONVKEY = np.array([re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID])
UCONVS = sorted(set(CONVKEY))
CTX = np.load("C:/locomo_refined/ctx_token_vecs/ctx_vecs.npy")
meta = [json.loads(l) for l in io.open("C:/locomo_refined/ctx_token_vecs/tokens_meta.jsonl", encoding="utf-8")]
REC2TOK = {}
for ti, m in enumerate(meta):
    REC2TOK.setdefault(m["r"], []).append(ti)
CTXN = l2n(CTX.astype(np.float32))
P("loaded %.0fs" % (time.time() - t0))

import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
def ctx_tokens(text):
    with torch.no_grad():
        enc = tok([text[:1500]], truncation=True, max_length=64, return_tensors="pt").to("cuda")
        out = model(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        return l2n(out[mask].cpu().numpy().astype(np.float32))

def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    return set(i for i in range(len(MID)) if any(k in RAWN[i] for k in keys))

def zs(a):
    a = np.asarray(a)
    return (a - a.mean()) / (a.std() + 1e-9)

def cliff_window(scores, kmin=1, kmax=15):
    """断崖检测: 前 kmax 名的归一分序列, 最大跌幅处截窗(不早于kmin)"""
    s = np.asarray(scores[:kmax])
    if len(s) <= kmin:
        return max(kmin, len(s))
    # 归一到[0,1]后看跌幅
    rng = max(1e-9, s[0] - s[-1])
    sn = (s - s[-1]) / rng
    drops = sn[:-1] - sn[1:]
    # 跌幅显著定义: 超过最大跌幅的0.5倍且绝对>0.08
    di = int(np.argmax(drops))
    if drops[di] > 0.08 and sn[di] > 0.3:
        return max(kmin, di + 1)
    return kmax

win_lens = []
pur_fix = pur_dyn = 0.0
hit_fix = hit_dyn = 0
n = 0
for k_i, qa in enumerate(IDS):
    G = gold_set(qa)
    if not G:
        continue
    cq = D @ X[k_i]
    sess_score = {}
    for cv in UCONVS:
        sess_score[cv] = float(cq[CONVKEY == cv].max())
    top_conv = max(sess_score, key=sess_score.get)
    rows = np.where((CONVKEY == top_conv) & np.array(["_rbak" not in MID[i] for i in range(len(MID))]))[0]
    if len(rows) < 5:
        continue
    qctx = ctx_tokens(Q[qa]["question"])
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    clean = []
    sc = []
    for r in rows:
        r = int(r)
        nw = len(RAWS[r].split())
        if nw <= 8 or (nw <= 15 and COURT.search(RAWS[r])):
            continue
        tidx = REC2TOK.get(r)
        al = float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1.0
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[r].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        clean.append(r)
        sc.append([float(D[r] @ X[k_i]), al, g1, nw])
    if len(clean) < 5:
        continue
    S = np.asarray(sc)
    fuse = zs(S[:, 0]) + zs(S[:, 1]) + zs(S[:, 2]) + zs(S[:, 3])
    order = np.argsort(-fuse)
    od = [clean[i] for i in order]
    fsorted = fuse[order]
    kw = cliff_window(fsorted)
    win_lens.append(kw)
    n += 1
    gfix = sum(1 for i in od[:5] if i in G)
    gdyn = sum(1 for i in od[:kw] if i in G)
    pur_fix += gfix / 5.0
    pur_dyn += gdyn / max(1, kw)
    if gfix:
        hit_fix += 1
    if gdyn:
        hit_dyn += 1
    if k_i % 300 == 0:
        P("  %d 纯度dyn=%.1f%% hitdyn=%.1f%% 均窗=%.1f %.0fs" % (
            k_i, 100 * pur_dyn / max(1, n), 100 * hit_dyn / max(1, n), np.mean(win_lens), time.time() - t0))

P("\n===== 动态窗口(断崖检测) vs 固定5 (n=%d) =====" % n)
P("固定@5:    纯度=%.1f%%  命中=%.1f%%" % (100 * pur_fix / n, 100 * hit_fix / n))
P("断崖动态:  纯度=%.1f%%  命中=%.1f%%  平均窗长=%.1f (min1/max15)" % (
    100 * pur_dyn / n, 100 * hit_dyn / n, np.mean(win_lens)))
P("窗口分布: P25=%.0f 中位=%.0f P75=%.0f" % tuple(np.percentile(win_lens, [25, 50, 75])))
P("F126_DONE %.0fs" % (time.time() - t0))
