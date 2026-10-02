import io, json, os, sys, re, bisect
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
np.seterr(all="ignore")
BASE = "C:/locomo_refined/memsys"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def nrm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

def strip_spk(t):
    return re.sub(r"^[A-Za-z .']+\s*:\s*", "", t)

def parse_mid(m):
    mm = re.search(r"_(?:rbak(\d+)k|m(\d+))$", m)
    return int(mm.group(1) or mm.group(2)) if mm else -1

# ---------- static loads ----------
V = np.load(BASE + "/mem_bge_dense.npz")["dense"]
MIDS = [json.loads(l)["mid"] for l in open(BASE + "/mem_bge_sparse.jsonl", encoding="utf-8")]
MID2ROW = {m: i for i, m in enumerate(MIDS)}
RECS = [json.loads(l) for l in open(BASE + "/mem.jsonl", encoding="utf-8")]
F = np.load(BASE + "/fusion_cache.npz")
SB, SQ, VEX = F["SB"], F["SQ"], F["VEX"]
RR = np.load(BASE + "/rerank_stage1.npz")
TOP50, RER = RR["TOP50"], RR["RER"]
O37 = json.load(open(BASE + "/out_r37.json", encoding="utf-8"))

NREC = len(RECS)
CONV_OF_MID = {}
TURN_OF_MID = {}
KIND_OF_MID = {}
RAWTEXT_OF_MID = {}
for r in RECS:
    m = r["memory_id"]
    CONV_OF_MID[m] = r["session_id"]
    TURN_OF_MID[m] = parse_mid(m)
    KIND_OF_MID[m] = r.get("kind") or "sum"
    RAWTEXT_OF_MID[m] = r.get("raw") or ""

# speaker: from raw prefix; summary inherits from twin raw
SPK_OF_MID = {}
RAW_OF = {}
for r in RECS:
    if r.get("kind") == "raw":
        RAW_OF[r.get("raw_of")] = r["memory_id"]
        mm = re.match(r"^([A-Za-z .']+)\s*:", r.get("raw") or "")
        SPK_OF_MID[r["memory_id"]] = mm.group(1).strip() if mm else "?"
for r in RECS:
    if (r.get("kind") or "sum") == "sum":
        SPK_OF_MID[r["memory_id"]] = SPK_OF_MID.get(RAW_OF.get(r["memory_id"], ""), "?")

# session segmentation via header records "[Session N - date]" (kind sum, turn t)
HDR_TURNS = {}
for r in RECS:
    if (r.get("kind") or "sum") == "sum" and (r.get("raw") or "").strip().startswith("[Session"):
        HDR_TURNS.setdefault(r["session_id"], []).append(parse_mid(r["memory_id"]))
for k in HDR_TURNS:
    HDR_TURNS[k] = sorted(set(HDR_TURNS[k]))

def sess_idx_of(mid):
    hs = HDR_TURNS.get(CONV_OF_MID[mid], [])
    t = TURN_OF_MID[mid]
    return bisect.bisect_right(hs, t) if hs else 0

# text -> raw mids (evidence matching)
NIDX = {}
for m, t in RAWTEXT_OF_MID.items():
    if m in MID2ROW and KIND_OF_MID[m] == "raw":
        NIDX.setdefault(nrm(strip_spk(t)), []).append(m)

# twin pairs (raw, summary) both in sparse
TWIN_PAIRS = []
for r in RECS:
    if r.get("kind") == "raw" and r["memory_id"] in MID2ROW and r.get("raw_of") in MID2ROW:
        TWIN_PAIRS.append((MID2ROW[r["memory_id"]], MID2ROW[r["raw_of"]]))
TWIN_PAIRS = np.array(TWIN_PAIRS)

# ---------- question meta ----------
if os.path.exists(BASE + "/dd_qmeta.json"):
    QMETA = json.load(open(BASE + "/dd_qmeta.json", encoding="utf-8"))
else:
    QMETA = []
    for i, q in enumerate(O37):
        conv = "loco-" + q["qa_id"].split("#")[0]
        ev_mids, ev_sess = [], set()
        for em in (q.get("evidence_messages") or []):
            for m in NIDX.get(nrm(em["text"]), []):
                if CONV_OF_MID[m] == conv:
                    ev_mids.append(m)
            if em.get("session_index") is not None:
                ev_sess.add(int(em["session_index"]))
        ev_mids = sorted(set(ev_mids))
        QMETA.append({
            "i": i, "qa_id": q["qa_id"], "conv": conv, "cat": q.get("category"),
            "score": q.get("llm_score"), "q": q.get("question"), "answer": q.get("answer"),
            "n_evmsg": len(q.get("evidence_messages") or []),
            "ev_mids": ev_mids, "ev_rows": [MID2ROW[m] for m in ev_mids],
            "ev_nsess": len(ev_sess),
        })
    json.dump(QMETA, open(BASE + "/dd_qmeta.json", "w", encoding="utf-8"))

# ---------- query embeddings + row mapping ----------
if os.path.exists(BASE + "/dd_qemb.npy"):
    QEMB = np.load(BASE + "/dd_qemb.npy")
else:
    from FlagEmbedding import BGEM3FlagModel
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    texts = [q["question"] for q in O37]
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    QEMB = l2n(np.concatenate(out))
    np.save(BASE + "/dd_qemb.npy", QEMB)

if os.path.exists(BASE + "/dd_S.npy"):
    S = np.load(BASE + "/dd_S.npy")
else:
    S = (QEMB @ V.T).astype(np.float32)
    np.save(BASE + "/dd_S.npy", S)

if os.path.exists(BASE + "/dd_qrow.json"):
    QR = json.load(open(BASE + "/dd_qrow.json", encoding="utf-8"))
    ROW2Q = QR["row2q"]
else:
    a = SB - SB.mean(1, keepdims=True)
    b = S - S.mean(1, keepdims=True)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    M = a @ b.T  # 1376 x 1382
    ROW2Q = [int(x) for x in np.argmax(M, axis=1)]
    conf = [float(M[j, ROW2Q[j]]) for j in range(M.shape[0])]
    json.dump({"row2q": ROW2Q, "conf": conf}, open(BASE + "/dd_qrow.json", "w"))
    del M
QROW = [-1] * len(O37)
for j, qi in enumerate(ROW2Q):
    if QROW[qi] == -1:
        QROW[qi] = j

def pools(i):
    """returns (row:int, evset:set, top50:np.array, noise:np.array, rer:np.array)"""
    j = QROW[i]
    if j < 0:
        return -1, set(), None, None, None
    t50 = TOP50[j]
    ev = set(QMETA[i]["ev_rows"])
    noise = np.array([r for r in t50 if r not in ev], dtype=int)
    return j, ev, t50, noise, RER[j]

def auc(y, s):
    y = np.asarray(y, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    p, n = y > 0.5, y <= 0.5
    if p.sum() == 0 or n.sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    ss = s[order]
    i = 0
    while i < len(ss):
        k = i
        while k + 1 < len(ss) and ss[k + 1] == ss[i]:
            k += 1
        if k > i:
            ranks[order[i:k + 1]] = (i + k + 2) / 2.0
        i = k + 1
    return float((ranks[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * n.sum()))

def folds(n, k=5, seed=7):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [np.sort(idx[i::k]) for i in range(k)]

def folds_by_conv(items_conv, k=5, seed=7):
    convs = sorted(set(items_conv))
    fl = folds(len(convs), k, seed)
    c2f = {}
    for fi, fl_ in enumerate(fl):
        for c in fl_:
            c2f[convs[c]] = fi
    return [np.array([i for i, c in enumerate(items_conv) if c2f[c] == fi]) for fi in range(k)]

def logit_auc(X, y, k=5, seed=7, l2=1.0, iters=300, lr=0.1):
    """5-fold CV logistic regression AUC (numpy). Returns (auc_list, mean)"""
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    fl = folds(len(y), k, seed)
    aucs = []
    for fi in range(k):
        te = fl[fi]; tr = np.concatenate([fl[x] for x in range(k) if x != fi])
        w = np.zeros(Xs.shape[1])
        for _ in range(iters):
            z = Xs[tr] @ w
            p = 1 / (1 + np.exp(-z))
            g = Xs[tr].T @ (p - np.asarray(y)[tr]) + l2 * w
            g[-1] -= l2 * w[-1]
            w -= lr * g / len(tr)
        s = Xs[te] @ w
        aucs.append(auc(np.asarray(y)[te], s))
    return aucs, float(np.nanmean(aucs))
