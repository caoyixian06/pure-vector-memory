# -*- coding: utf-8 -*-
"""foundry129.py — 降分母双武器(外部AI方案的可测部分):
A语义去重(候选池cos>0.90聚簇,簇代表竞争) B极值动态阈值(tau=mu+sigma*z(1-1/N))
对照: 固定top5命中65.6 | 全程零标签"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry129_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
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

from scipy.stats import norm as spnorm
res = {"base": 0, "dedup": 0, "dynthr": 0, "both": 0}
sizes_after = []
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
    if len(rows) < 10:
        continue
    qctx = ctx_tokens(Q[qa]["question"])
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
    clean_idx = []
    sc = []
    for rr, r in enumerate(rows):
        r = int(r)
        nw = len(RAWS[r].split())
        if nw <= 8 or (nw <= 15 and COURT.search(RAWS[r])):
            continue
        tidx = REC2TOK.get(r)
        al = float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1.0
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[r].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        clean_idx.append(rr)
        sc.append([float(D[r] @ X[k_i]), al, g1, nw])
    if len(clean_idx) < 8:
        continue
    S = np.asarray(sc)
    fuse = zs(S[:, 0]) + zs(S[:, 1]) + zs(S[:, 2]) + zs(S[:, 3])
    cres = [int(rows[rr]) for rr in clean_idx]
    order_all = [cres[i] for i in np.argsort(-fuse)]
    n += 1
    # base
    if any(i in G for i in order_all[:5]):
        res["base"] += 1
    # A: 语义去重 — 贪心簇(cos>0.90), 簇代表=簇内最高fuse, 取5簇展开代表
    sub_D = D[cres]
    SIM = sub_D @ sub_D.T
    fu = np.argsort(-fuse)
    taken = np.zeros(len(cres), dtype=bool)
    reps = []
    for i in fu:
        if taken[i]:
            continue
        reps.append(i)
        taken |= SIM[i] > 0.90
        if len(reps) >= 15:
            break
    sizes_after.append(len(reps))
    ord_rep = [cres[i] for i in sorted(reps, key=lambda j: -fuse[j])]
    if any(i in G for i in ord_rep[:5]):
        res["dedup"] += 1
    # B: 极值动态阈值 — tau = mu + sigma*z(1-1/N), 取超阈值的(截1-15)
    N = len(cres)
    zq = spnorm.ppf(1 - 1.0 / max(2, N))
    tau = fuse.mean() + fuse.std() * zq
    sel = np.where(fuse > tau)[0]
    if len(sel) == 0 or len(sel) > 15:
        sel = np.argsort(-fuse)[:5]
    ord_thr = [cres[i] for i in sel[np.argsort(-fuse[sel])]]
    if any(i in G for i in ord_thr[:5]):
        res["dynthr"] += 1
    # both: 去重后的代表上再用动态阈值
    rep_fuse = np.array([fuse[i] for i in sorted(reps, key=lambda j: -fuse[j])])
    Nr = len(rep_fuse)
    zqr = spnorm.ppf(1 - 1.0 / max(2, Nr))
    taur = rep_fuse.mean() + rep_fuse.std() * zqr
    selr = np.where(rep_fuse > taur)[0]
    if len(selr) == 0 or len(selr) > 15:
        selr = np.argsort(-rep_fuse)[:5]
    if any(i in G for i in [ord_rep[j] for j in selr]):
        res["both"] += 1
    if k_i % 300 == 0:
        P("  %d base=%.1f dedup=%.1f dyn=%.1f both=%.1f 去重后均=%.0f %.0fs" % (
            k_i, 100 * res["base"] / max(1, n), 100 * res["dedup"] / max(1, n),
            100 * res["dynthr"] / max(1, n), 100 * res["both"] / max(1, n),
            np.mean(sizes_after), time.time() - t0))

P("\n===== 降分母双武器(n=%d) =====" % n)
P("固定top5(基线):        命中=%.1f%%" % (100.0 * res["base"] / n))
P("A语义去重(cos>0.90簇): 命中=%.1f%%  去重后候选均=%.0f条(原%d)" % (
    100.0 * res["dedup"] / n, np.mean(sizes_after), 661))
P("B极值动态阈值:          命中=%.1f%%" % (100.0 * res["dynthr"] / n))
P("A+B组合:               命中=%.1f%%" % (100.0 * res["both"] / n))
P("F129_DONE %.0fs" % (time.time() - t0))
