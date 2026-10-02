# -*- coding: utf-8 -*-
"""foundry125.py — 窗口金纯度攻坚(用户: 增强信号+分离清洗→前5约80%是金):
纯度口径=前5条中金条目占比。臂: cos / 融合 / 清洗+融合 / 清洗+四信号"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry125_results.txt", "w", encoding="utf-8")
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

ARMS = ("cos", "fuse2", "clean_fuse2", "clean_fuse4")
purity = {a: 0.0 for a in ARMS}
hits5 = {a: 0 for a in ARMS}
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
    sc_cos, sc_align, sc_g1, sc_len = [], [], [], []
    for r in rows:
        r = int(r)
        sc_cos.append(float(D[r] @ X[k_i]))
        tidx = REC2TOK.get(r)
        sc_align.append(float((qctx @ CTXN[tidx].T).max(axis=1).mean()) if tidx else -1.0)
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAWS[r].lower()))
        sc_g1.append(len(qstems & rstems) / max(1, len(qstems)))
        sc_len.append(len(RAWS[r].split()))
    sc_cos, sc_align, sc_g1, sc_len = map(np.asarray, (sc_cos, sc_align, sc_g1, sc_len))
    f2 = zs(sc_cos) + zs(sc_align)
    f4 = zs(sc_cos) + zs(sc_align) + zs(sc_g1) + zs(sc_len)
    clean_mask = np.array([(len(RAWS[int(r)].split()) > 8 and not (len(RAWS[int(r)].split()) <= 15 and COURT.search(RAWS[int(r)]))) for r in rows])
    def topk(score, mask=None, k=5):
        idx = np.where(mask)[0] if mask is not None else np.arange(len(rows))
        if len(idx) < k:
            idx = np.arange(len(rows))
        sel = idx[np.argsort(-score[idx])[:k]]
        return [int(rows[i]) for i in sel]
    orders = {
        "cos": topk(sc_cos),
        "fuse2": topk(f2),
        "clean_fuse2": topk(f2, clean_mask),
        "clean_fuse4": topk(f4, clean_mask),
    }
    n += 1
    for a, od in orders.items():
        gold_in = sum(1 for i in od if i in G)
        purity[a] += gold_in / 5.0
        if gold_in > 0:
            hits5[a] += 1
    if k_i % 200 == 0:
        P("  %d 纯度: cos=%.1f%% f2=%.1f%% cf2=%.1f%% cf4=%.1f%% %.0fs" % (
            k_i, 100 * purity["cos"] / max(1, n), 100 * purity["fuse2"] / max(1, n),
            100 * purity["clean_fuse2"] / max(1, n), 100 * purity["clean_fuse4"] / max(1, n), time.time() - t0))

P("\n===== 窗口金纯度(n=%d, 前5条中金占比) =====" % n)
for a in ARMS:
    nm = {"cos": "记录cos", "fuse2": "cos+对齐", "clean_fuse2": "清洗+cos对齐", "clean_fuse4": "清洗+四信号"}[a]
    P("%-14s 纯度=%.1f%%  命中@5=%.1f%%" % (nm, 100.0 * purity[a] / n, 100.0 * hits5[a] / n))
P("目标: 纯度~80% | 参考: 金均%.1f条/题(含副本)→纯度理论天花板=%.0f%%" % (
    np.mean([len(gold_set(qa)) for qa in IDS[:200]]) * 0 + 5.7, min(100, 5.7 / 5 * 100)))
P("F125_DONE %.0fs" % (time.time() - t0))
