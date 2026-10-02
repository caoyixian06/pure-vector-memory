# -*- coding: utf-8 -*-
"""diag_heldout.py — 过拟合检验: 全组件零改动, 在原版LoCoMo题(held-out)上测增益迁移
held-out构建: 原版题(cat1-4) - 与精修版题目文本重复(含近似)的剔除
四层配置(全部冻结参数): a纯BGE / b融合u2 / c融合u2+PRF / d全约束
双集对比: 同脚本同口径在精修版全量重算四层 → 迁移率=held-out增益/refined增益
"""
import io, json, os, sys, re, time, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel, FlagReranker
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

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
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW /= np.maximum(np.linalg.norm(QW, axis=1, keepdims=True), 1e-9)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

# ---- 精修版题(全集) ----
refined_qs = [json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()]
ref_norm = {normsub(q["question"]) for q in refined_qs}
ref_toks = [set(re.findall(r"[a-z0-9]+", q["question"].lower())) for q in refined_qs]

# ---- 原版题构建 held-out ----
orig = json.load(open(r"C:/locomo_refined/locomo10.json", encoding="utf-8"))
held = []
for item in orig:
    sid = item["sample_id"]
    conv = item["conversation"]
    for qa in item.get("qa", []):
        if str(qa.get("category")) == "5":
            continue
        q = qa["question"]
        if normsub(q) in ref_norm:
            continue
        toks = set(re.findall(r"[a-z0-9]+", q.lower()))
        dup = False
        for rt in ref_toks:
            inter = len(toks & rt)
            if inter and inter / max(1, len(toks | rt)) >= 0.85:
                dup = True
                break
        if dup:
            continue
        evs = []
        for ev in qa.get("evidence", []):
            m = re.match(r"D(\d+):(\d+)", str(ev))
            if not m:
                continue
            sess = conv.get("session_%s" % m.group(1)) or []
            idx = int(m.group(2))
            if idx < len(sess):
                t = sess[idx]
                evs.append("%s: %s" % (t.get("speaker", "?"), t.get("text", "")))
        if evs:
            held.append(dict(sid=sid, question=q, answer=qa.get("answer"), evs=evs, cat=qa.get("category")))
print("held-out 原版题(去重后):", len(held), flush=True)

def find_hits(sid, evs):
    cand = CONV.get("loco-" + sid, [])
    hits = []
    for e in evs:
        t = normsub(e)
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    return hits

targets = []
for k, h in enumerate(held):
    hits = find_hits(h["sid"], h["evs"])
    if hits:
        targets.append((k, hits))
print("证据可匹配:", len(targets), "/", len(held), flush=True)
for k, hits in targets[:3]:
    h = held[k]
    print("样例: Q:", h["question"][:70], "| A:", str(h["answer"])[:40])
    print("      EV:", h["evs"][0][:90], flush=True)

# ---- 冻结组件重算 ----
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
STOP = set("a an the is are was were be been being do does did have has had i you he she it we they me him her us them my your his its our their what when where who whom why how which that this these those there to of in on at for with about from by as and or but if so not no s t re ve ll d m".split())
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
CLEAN = {}
for w in WH.keys():
    c = w.strip(":").lower()
    if c and c not in CLEAN:
        CLEAN[c] = w
HOST_IDX = {w: [MID2I[h] for h in hs if h in MID2I] for w, hs in WH.items()}
def word_votes(q):
    v = np.zeros(N, dtype=np.float32)
    for w in re.findall(r"[a-z']+", q.lower()):
        if w in STOP or len(w) <= 1:
            continue
        key = CLEAN.get(w)
        if key:
            v[HOST_IDX[key]] += 1.0
    return v
def ollama_embed(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
    return l2n(np.concatenate(out))
def emb_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

# u2 冻结: 用精修版问题质心(部署时就是这样)
ref_qtexts = [q["question"] for q in refined_qs]
bQ_ref = emb_bge(ref_qtexts)
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ_ref.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
print("u2 frozen(精修题质心)", flush=True)

def run_set(qtexts, sid_list, tag):
    n = len(qtexts)
    bQ = emb_bge(qtexts)
    wQ = ollama_embed(qtexts)
    SBS0 = (l2n(bQ + u2) @ D.T).astype(np.float32)
    FUSED = np.stack([zs(SBS0[i]) + zs(QW @ wQ[i]) + 0.5 * zs(word_votes(qtexts[i])) for i in range(n)])
    # rerank top50(基于FUSED)
    TOP50 = np.zeros((n, 50), dtype=np.int64)
    RER = np.zeros((n, 50), dtype=np.float32)
    for i in range(n):
        top = np.argsort(-FUSED[i])[:50]
        TOP50[i] = top
        sc = rer.compute_score([[qtexts[i], TEXTS[j]] for j in top], batch_size=50)
        RER[i] = np.asarray(sc, dtype=np.float32)
    ZRER = np.zeros((n, N), dtype=np.float32)
    for i in range(n):
        v = np.full(N, float(RER[i].min()) - 1.0, dtype=np.float32)
        v[TOP50[i]] = RER[i]
        ZRER[i] = zs(v)
    # PRF
    qq = l2n(bQ + u2).copy()
    exp = np.zeros_like(qq)
    for i in range(n):
        top = TOP50[i][np.argsort(-RER[i])[:3]]
        c = D[top].mean(0)
        exp[i] = c / (np.linalg.norm(c) + 1e-9)
    qq = l2n(qq + 0.5 * exp)
    SBSp = (qq @ D.T).astype(np.float32)
    BASE = np.stack([zs(SBSp[i]) + zs(QW @ wQ[i]) + 0.5 * zs(word_votes(qtexts[i])) for i in range(n)])
    ADJ = np.zeros((n, N), dtype=np.float32)
    TWINV = np.zeros((n, N), dtype=np.float32)
    for i in range(n):
        order = np.argsort(-BASE[i])[:5]
        for j in order:
            if TEXTS[j].rstrip().endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
                ADJ[i][j + 1] += 1.0
        order50 = np.argsort(-BASE[i])[:50]
        for r0, j in enumerate(order50):
            tw = TWIN.get(j)
            if tw is not None:
                TWINV[i][tw] += max(0.0, 1.0 - r0 / 50.0)
    cfgs = {
        "a纯BGE": lambda i: D @ bQ[i],
        "b融合u2": lambda i: FUSED[i],
        "c融合u2+PRF": lambda i: BASE[i],
        "d全约束": lambda i: BASE[i] + 0.5 * ZRER[i] + 0.5 * ADJ[i] + 0.1 * TWINV[i],
    }
    print("==== %s (n=%d) ====" % (tag, n), flush=True)
    for name, f in cfgs.items():
        rks = []
        for i in range(n):
            s = f(i)
            order = np.argsort(-s)
            pos = {x: p for p, x in enumerate(order)}
            rks.append(min(pos[h] for h in tgt_of[i]) + 1)
        rks = np.array(rks)
        print("  %-14s 进25 %.1f%% 进5 %.1f%% 均名次%.0f" % (
            name, 100 * (rks <= 25).mean(), 100 * (rks <= 5).mean(), rks.mean()), flush=True)

# held-out的targets索引
tgt_of = {k: hits for k, hits in targets}
hq = [held[k]["question"] for k, _ in targets]
run_set(hq, None, "HELD-OUT 原版题")

# 精修版全量同口径(证据匹配)
ref_targets = []
for qi, q in enumerate(refined_qs):
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    hits = []
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.append(j)
                break
    if hits:
        ref_targets.append((qi, hits))
tgt_of = {qi: hits for qi, hits in ref_targets}
bQ_ref2 = [refined_qs[qi]["question"] for qi, _ in ref_targets]
run_set(bQ_ref2, None, "REFINED 精修题(同口径)")
