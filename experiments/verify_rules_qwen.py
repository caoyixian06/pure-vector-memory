# -*- coding: utf-8 -*-
"""verify_rules_qwen.py — 三条"问题规则"在qwen3-embedding-256空间重验
同verify_rules.py逻辑, 全部换成qwen3-embedding(ollama, 256维)
"""
import io, json, os, sys, re, random, time, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 32):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 32], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
        print("  emb batch", s, flush=True)
    return l2n(np.concatenate(out))

HERE = "C:/locomo_refined/memsys"
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
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

# 第一人称事实句
fp_sents = []
for k in range(N):
    if KIND[k] != "raw":
        continue
    t = rec_text(REC.get(MID[k], {}))
    if re.match(r"^[A-Z][a-z]+:\s*(I|My|We)\b", t) and 30 < len(t) < 200 and "?" not in t:
        if re.search(r"\b(have|has|had|am|was|went|got|like|love|work|live|study|bought|moved|started|finished)\b", t, re.I):
            fp_sents.append((k, t))
print("第一人称事实句:", len(fp_sents), flush=True)
rng = random.Random(9)
sample = rng.sample(fp_sents, min(60, len(fp_sents)))
print("== V1q 人称失配(qwen256空间) ==", flush=True)
win_fp = win_tp = tie = 0
texts_to_emb_fp, texts_to_emb_tp = [], []
meta = []
for k, t in sample[:60]:
    spk, body = t.split(":", 1)
    body = body.strip()
    q_fp = "What did I " + re.sub(r"^(have|had|like|love|want|need|start)\b", r"\1", body.lower(), flags=re.I)[:90]
    q_tp = "What did " + spk + " " + re.sub(r"^(have|had|like|love|want|need|start)\b", r"\1", body.lower(), flags=re.I)[:90]
    meta.append((k, q_fp, q_tp))
    texts_to_emb_fp.append(q_fp)
    texts_to_emb_tp.append(q_tp)
# 库句子的qwen向量: 用REC自带的vector字段(每条记录自带256向量)
QWLIB = np.zeros((N, 256), dtype=np.float32)
for k2 in range(N):
    v = REC.get(MID[k2], {}).get("vector")
    if v:
        QWLIB[k2] = v
QWLIB = l2n(QWLIB)
KV = []
for k, t in sample[:60]:
    KV.append(np.asarray(REC.get(MID[k], {}).get("vector"), dtype=np.float32))
QF = emb(texts_to_emb_fp)
QT = emb(texts_to_emb_tp)
for idx, (k, q_fp, q_tp) in enumerate(meta):
    kv = KV[idx] / (np.linalg.norm(KV[idx]) + 1e-9)
    r_fp = int(((QWLIB @ QF[idx]) > (QF[idx] @ kv)).sum()) + 1
    r_tp = int(((QWLIB @ QT[idx]) > (QT[idx] @ kv)).sum()) + 1
    if r_fp < r_tp:
        win_fp += 1
    elif r_tp < r_fp:
        win_tp += 1
    else:
        tie += 1
print("  第一人称查询胜率=%.0f%% 第三人称=%.0f%% 平=%.0f%%" % (
    100 * win_fp / len(meta), 100 * win_tp / len(meta), 100 * tie / len(meta)), flush=True)

# V2q: 相对时间(qwen空间)
print("== V2q 相对时间失配(qwen256) ==", flush=True)
DATE_SENTS = []
for k in range(N):
    if KIND[k] != "raw":
        continue
    t = rec_text(REC.get(MID[k], {}))
    m = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,4}\b", t)
    if m and "?" not in t and 40 < len(t) < 220:
        DATE_SENTS.append((k, t))
print("含日期句:", len(DATE_SENTS), flush=True)
sample2 = rng.sample(DATE_SENTS, min(40, len(DATE_SENTS)))
qrels, qabs, kvs = [], [], []
for k, t in sample2:
    q_rel = "When did " + re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,4}\b", "last month", t, flags=re.I)[:100]
    q_abs = "When did " + t[:90]
    qrels.append(q_rel); qabs.append(q_abs)
    kvs.append(np.asarray(REC.get(MID[k], {}).get("vector"), dtype=np.float32))
QR = emb(qrels); QA = emb(qabs)
wr = wa = tie2 = 0
for idx, kv in enumerate(kvs):
    kv = kv / (np.linalg.norm(kv) + 1e-9)
    r_rel = int(((QWLIB @ QR[idx]) > (QR[idx] @ kv)).sum()) + 1
    r_abs = int(((QWLIB @ QA[idx]) > (QA[idx] @ kv)).sum()) + 1
    if r_rel < r_abs:
        wr += 1
    elif r_abs < r_rel:
        wa += 1
    else:
        tie2 += 1
print("  相对词查询胜=%.0f%% 绝对词=%.0f%% 平=%.0f%%" % (
    100 * wr / max(1, len(kvs)), 100 * wa / max(1, len(kvs)), 100 * tie2 / max(1, len(kvs))), flush=True)

# V3q: 相对词与月份(qwen词向量域)
print("== V3q 相对词vs月份(qwen词向量) ==", flush=True)
rel_words = ["yesterday", "last week", "last month", "next week", "tomorrow", "recently"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
rw_list = ["apple", "democracy", "guitar", "cloud", "pencil", "justice"]
allw = rel_words + months + rw_list
EMB = emb(allw)
EMB = l2n(EMB)
for ridx, rw in enumerate(rel_words):
    d_month = float(np.max(EMB[ridx] @ EMB[len(rel_words):len(rel_words)+12].T))
    d_rand = float(np.max(EMB[ridx] @ EMB[len(rel_words)+12+ridx][None].T))
    print("  %-11s vs月份max=%.3f vs随机=%.3f" % (rw, d_month, d_rand), flush=True)
print("QVERIFY_DONE", flush=True)
