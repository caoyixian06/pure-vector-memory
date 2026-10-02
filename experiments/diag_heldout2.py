# -*- coding: utf-8 -*-
"""diag_heldout2.py — held-out再解剖: 按标签可信度(证据文本含答案)分层,拆开"标签烂"与"过拟合"
可信=答案内容词≥50%出现在证据句里; 分别在可信/不可信子集看四层增益
精修版同口径对照(修复KeyError)
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
def toks(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

refined_qs = [json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()]
ref_norm = {normsub(q["question"]) for q in refined_qs}
ref_toks = [toks(q["question"]) for q in refined_qs]
STOP = set("a an the is are was were be been being do does did have has had i you he she it we they me him her us them my your his its our their what when where who whom why how which that this these those there to of in on at for with about from by as and or but if so not no s t re ve ll d m likely probably yes no".split())

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
        qt = toks(q)
        dup = any(len(qt & rt) / max(1, len(qt | rt)) >= 0.85 for rt in ref_toks)
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
            held.append(dict(sid=sid, question=q, answer=qa.get("answer"), evs=evs))

def plausible(h):
    at = {w for w in toks(h["answer"]) if w not in STOP and len(w) > 1}
    if not at:
        return True
    evt = toks(" ".join(h["evs"]))
    return len(at & evt) / len(at) >= 0.5

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

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
CLEAN = {}
for w in WH.keys():
    c = w.strip(":").lower()
    if c and c not in CLEAN:
        CLEAN[c] = w
HOST_IDX = {w: [MID2I[h2] for h2 in hs if h2 in MID2I] for w, hs in WH.items()}
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

bQ_ref = emb_bge([q["question"] for q in refined_qs])
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ_ref.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc

def run_pairs(pairs, tag):
    """pairs: list of (question_text, hits)"""
    n = len(pairs)
    qtexts = [p[0] for p in pairs]
    bQ = emb_bge(qtexts)
    wQ = ollama_embed(qtexts)
    SBS0 = (l2n(bQ + u2) @ D.T).astype(np.float32)
    FUSED = np.stack([zs(SBS0[i]) + zs(QW @ wQ[i]) + 0.5 * zs(word_votes(qtexts[i])) for i in range(n)])
    TOP50 = np.zeros((n, 50), dtype=np.int64)
    RER = np.zeros((n, 50), dtype=np.float32)
    for i in range(n):
        top = np.argsort(-FUSED[i])[:50]
        TOP50[i] = top
        sc = rer.compute_score([[qtexts[i], TEXTS[j]] for j in top], batch_size=50)
        RER[i] = np.asarray(sc, dtype=np.float32)
    ZRER = []
    for i in range(n):
        v = np.full(N, float(RER[i].min()) - 1.0, dtype=np.float32)
        v[TOP50[i]] = RER[i]
        ZRER.append(zs(v))
    qq = l2n(bQ + u2).copy()
    exp = np.zeros_like(qq)
    for i in range(n):
        top = TOP50[i][np.argsort(-RER[i])[:3]]
        c = D[top].mean(0)
        exp[i] = c / (np.linalg.norm(c) + 1e-9)
    qq = l2n(qq + 0.5 * exp)
    SBSp = (qq @ D.T).astype(np.float32)
    BASE = np.stack([zs(SBSp[i]) + zs(QW @ wQ[i]) + 0.5 * zs(word_votes(qtexts[i])) for i in range(n)])
    cfgs = {
        "a纯BGE": lambda i: D @ bQ[i],
        "b融合u2": lambda i: FUSED[i],
        "c+PRF": lambda i: BASE[i],
        "d全约束": lambda i: BASE[i] + 0.5 * ZRER[i] + 0.5 * adjv(i, BASE) + 0.1 * twv(i, BASE),
    }
    def adjv(i, B):
        v = np.zeros(N, dtype=np.float32)
        order = np.argsort(-B[i])[:5]
        for j in order:
            if TEXTS[j].rstrip().endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
                v[j + 1] += 1.0
        return v
    def twv(i, B):
        v = np.zeros(N, dtype=np.float32)
        order = np.argsort(-B[i])[:50]
        for r0, j in enumerate(order):
            tw = TWIN.get(j)
            if tw is not None:
                v[tw] += max(0.0, 1.0 - r0 / 50.0)
        return v
    print("==== %s (n=%d) ====" % (tag, n), flush=True)
    out = {}
    for name, f in cfgs.items():
        rks = []
        for i in range(n):
            s = f(i)
            order = np.argsort(-s)
            pos = {x: p for p, x in enumerate(order)}
            rks.append(min(pos[h2] for h2 in pairs[i][1]) + 1)
        rks = np.array(rks)
        print("  %-10s 进25 %.1f%% 进5 %.1f%% 均名次%.0f" % (
            name, 100 * (rks <= 25).mean(), 100 * (rks <= 5).mean(), rks.mean()), flush=True)
        out[name] = ((rks <= 25).mean(), (rks <= 5).mean(), rks.mean())
    return out

pl = [(h, find_hits(h["sid"], h["evs"])) for h in held]
pl = [(h, hh) for h, hh in pl if hh]
plaus = [(h["question"], hh) for h, hh in pl if plausible(h)]
unpl = [(h["question"], hh) for h, hh in pl if not plausible(h)]
print("held-out %d = 可信标签 %d + 不可信标签 %d" % (len(pl), len(plaus), len(unpl)), flush=True)
run_pairs(plaus, "HELD-OUT 可信标签(证据含答案)")
run_pairs(unpl, "HELD-OUT 不可信标签(证据不含答案)")

# 精修版同口径全量
ref_pairs = []
for q in refined_qs:
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
        ref_pairs.append((q["question"], hits))
print("refined可匹配:", len(ref_pairs), flush=True)
run_pairs(ref_pairs, "REFINED 精修题(同口径全量)")
