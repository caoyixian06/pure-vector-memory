# -*- coding: utf-8 -*-
"""lme_build.py — LongMemEval-S 建库(独立目录 longmemeval_mem, 绝不混LoCoMo库)
阶段: A解析去重(CPU,每次重算~2min) → B bge-m3 dense1024分块断点(GPU) → C ollama 256向量(GPU)
      → D 词向量+原子句预嵌入(分块断点, GPU)
无GLM摘要(用户红线), 无硬规则过滤(向量规则留给检索层)
产物: mem.jsonl / rec_vec256.npy / mem_bge_dense.npz / mem_bge_sparse.jsonl
      / word_vecs.npz / word_hosts.json / atom_cache.npz / lme_index.json
"""
import io, json, os, re, sys, time, hashlib
import urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
os.makedirs(OUT, exist_ok=True)
CAP = 6000

def wjson(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def rjson(p):
    return json.load(open(p, encoding="utf-8"))

# ===== Stage A: 解析+内容去重 (CPU, 确定性, 每次重算) =====
t0 = time.time()
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"],
                      q["haystack_sessions"])) for q in d], [])
sess, sdate, sid2h = {}, {}, {}
for sid, dt, turns in pairs:
    h = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
    sid2h[sid] = h
    if h not in sess:
        sess[h] = turns
        sdate[h] = dt
hashes = list(sess.keys())
SIDS, MIDS, KINDS, RAWS = [], [], [], []
for si, h in enumerate(hashes):
    dt = (sdate[h] or "")[:10]
    sid = "lme-s" + h[:12]
    SIDS.append(sid)
    MIDS.append("lme-b%s-h" % h[:12]); KINDS.append("hdr")
    RAWS.append("[Session %d — %s]" % (si + 1, dt))
    for ti, t in enumerate(sess[h]):
        role = t.get("role") or "user"
        MIDS.append("lme-b%s-t%04d" % (h[:12], ti)); KINDS.append("raw")
        SIDS.append(sid)
        RAWS.append(("user: " if role == "user" else "assistant: ") + str(t.get("content") or "").strip())
print("StageA: pairs=%d real_sess=%d records=%d %.0fs" % (
    len(pairs), len(hashes), len(RAWS), time.time() - t0), flush=True)
wjson(OUT + "/_sid2h.json", sid2h)

# ===== Stage B: bge-m3 dense 1024 (GPU, 分块断点) =====
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
os.makedirs(OUT + "/_bge_parts", exist_ok=True)
CH = 16384
nchunk = (len(RAWS) + CH - 1) // CH
done_b = {int(x) for x in (open(OUT + "/_bge_done.txt").read().split()
                           if os.path.exists(OUT + "/_bge_done.txt") else "")}
t0 = time.time()
for c in range(nchunk):
    if c in done_b:
        continue
    part = [r[:CAP] for r in RAWS[c * CH:(c + 1) * CH]]
    out = []
    for s in range(0, len(part), 64):
        out.append(np.asarray(bge.encode(part[s:s + 64])["dense_vecs"], dtype=np.float32))
    np.save(OUT + "/_bge_parts/p%04d.npy" % c, np.concatenate(out))
    done_b.add(c)
    open(OUT + "/_bge_done.txt", "w").write(" ".join(map(str, sorted(done_b))))
    print("StageB chunk %d/%d %.0fs" % (c + 1, nchunk, time.time() - t0), flush=True)
D = np.concatenate([np.load(OUT + "/_bge_parts/p%04d.npy" % c) for c in range(nchunk)])
np.savez_compressed(OUT + "/mem_bge_dense.npz", dense=D)
print("StageB done", D.shape, flush=True)
del bge
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
print("bge unloaded, VRAM freed", flush=True)

# ===== Stage C: ollama qwen3-embed 256 (GPU, 分块断点) =====
def oemb(texts, dims=256):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest", "input": texts[s:s + 64],
                           "dimensions": dims}).encode()
        for attempt in range(4):
            req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                break
            except Exception as e:
                print("oemb retry %d: %s" % (attempt + 1, str(e)[:80]), flush=True)
                if attempt == 3:
                    raise
                time.sleep(8 * (attempt + 1))
    return out

os.makedirs(OUT + "/_v256_parts", exist_ok=True)
CV = 16384
nc2 = (len(RAWS) + CV - 1) // CV
done_c = {int(x) for x in (open(OUT + "/_v256_done.txt").read().split()
                           if os.path.exists(OUT + "/_v256_done.txt") else "")}
t0 = time.time()
for c in range(nc2):
    if c in done_c:
        continue
    part = [r[:CAP] for r in RAWS[c * CV:(c + 1) * CV]]
    np.save(OUT + "/_v256_parts/q%04d.npy" % c,
            np.concatenate(oemb(part)))
    done_c.add(c)
    open(OUT + "/_v256_done.txt", "w").write(" ".join(map(str, sorted(done_c))))
    print("StageC chunk %d/%d %.0fs" % (c + 1, nc2, time.time() - t0), flush=True)
V = np.concatenate([np.load(OUT + "/_v256_parts/q%04d.npy" % c) for c in range(nc2)])
np.save(OUT + "/rec_vec256.npy", V)
print("StageC done", V.shape, flush=True)

# ===== mem.jsonl + sparse =====
with open(OUT + "/mem.jsonl", "w", encoding="utf-8") as f:
    for i in range(len(RAWS)):
        f.write(json.dumps({"memory_id": MIDS[i], "kind": "raw", "session_id": SIDS[i],
                            "raw": RAWS[i][:CAP], "order": i}, ensure_ascii=False) + NL)
with open(OUT + "/mem_bge_sparse.jsonl", "w", encoding="utf-8") as f:
    for i in range(len(RAWS)):
        f.write(json.dumps({"mid": MIDS[i], "kind": KINDS[i]}) + NL)
print("mem.jsonl written", len(RAWS), flush=True)

# ===== Stage D: 词向量 + 原子句预嵌入 (GPU, 分块断点) =====
COUR = re.compile(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank|cool|nice)\b", re.I)
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
vocab, hosts, atoms = {}, {}, []
for i, raw in enumerate(RAWS):
    if KINDS[i] != "raw":
        continue
    for w in re.findall(r"[a-z0-9']+", raw.lower()):
        if len(w) < 2:
            continue
        c0 = stem(w)
        if c0 not in vocab:
            vocab[c0] = len(vocab)
            hosts[c0] = set()
        hosts[c0].add(MIDS[i])
    for piece in re.split(r"[.!?]+", raw):
        p = piece.strip()
        ws = p.split()
        if len(ws) < 4 or len(ws) > 45:
            continue
        if len(ws) <= 5 and COUR.search(p):
            continue
        atoms.append((p, i))
words = sorted(vocab, key=lambda w: vocab[w])
print("vocab=%d atoms=%d" % (len(words), len(atoms)), flush=True)

os.makedirs(OUT + "/_w256_parts", exist_ok=True)
WCH = 8192
nw = (len(words) + WCH - 1) // WCH
done_w = {int(x) for x in (open(OUT + "/_w256_done.txt").read().split()
                           if os.path.exists(OUT + "/_w256_done.txt") else "")}
t0 = time.time()
for c in range(nw):
    if c in done_w:
        continue
    np.save(OUT + "/_w256_parts/w%04d.npy" % c,
            np.concatenate(oemb(words[c * WCH:(c + 1) * WCH])))
    done_w.add(c)
    open(OUT + "/_w256_done.txt", "w").write(" ".join(map(str, sorted(done_w))))
    print("StageD words chunk %d/%d %.0fs" % (c + 1, nw, time.time() - t0), flush=True)
WV = np.concatenate([np.load(OUT + "/_w256_parts/w%04d.npy" % c) for c in range(nw)])
np.savez_compressed(OUT + "/word_vecs.npz", V=WV)
wjson(OUT + "/word_hosts.json", {w: sorted(hosts[w]) for w in words})
print("StageD words done", WV.shape, flush=True)

try:
    _body = json.dumps({"model": "qwen3-embedding:latest", "input": ["x"], "keep_alive": 0}).encode()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/embed", data=_body,
                                  headers={"Content-Type": "application/json"}), timeout=60).read()
    print("ollama unloaded for bge atoms", flush=True)
    time.sleep(5)
except Exception as _e:
    print("ollama unload skip:", str(_e)[:60], flush=True)
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("bge reloaded for atoms", flush=True)
os.makedirs(OUT + "/_atom_parts", exist_ok=True)
ACH = 16384
na = (len(atoms) + ACH - 1) // ACH
done_a = {int(x) for x in (open(OUT + "/_atom_done.txt").read().split()
                           if os.path.exists(OUT + "/_atom_done.txt") else "")}
t0 = time.time()
for c in range(na):
    if c in done_a:
        continue
    part = [atoms[c * ACH + k][0][:300] for k in range(min(ACH, len(atoms) - c * ACH))]
    out = []
    for s in range(0, len(part), 64):
        out.append(np.asarray(bge.encode(part[s:s + 64])["dense_vecs"], dtype=np.float32))
    np.save(OUT + "/_atom_parts/a%04d.npy" % c, np.concatenate(out))
    done_a.add(c)
    open(OUT + "/_atom_done.txt", "w").write(" ".join(map(str, sorted(done_a))))
    print("StageD atoms chunk %d/%d %.0fs" % (c + 1, na, time.time() - t0), flush=True)
AV = np.concatenate([np.load(OUT + "/_atom_parts/a%04d.npy" % c) for c in range(na)])
np.savez_compressed(OUT + "/atom_cache.npz", V=AV,
                    HOST=np.array([a[1] for a in atoms]))
print("StageD atoms done", AV.shape, flush=True)

# ===== Stage E: 发现字段(方法迁移, 常数本地derive) =====
import re as _re
# E1 邻域ID: 同会话±1/±2记录的行号(检索时零成本取邻域块)
# 记录布局: 每会话1条hdr + N条turn, hdr行跳过, turn行按顺序
SESS_OF_ROW = []
_s = None
for i, m in enumerate(MIDS):
    if KINDS[i] == "hdr":
        _s = i
    SESS_OF_ROW.append(_s)
NEIGH = {}
for i in range(len(RAWS)):
    if KINDS[i] != "raw":
        continue
    nb = []
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(RAWS) and KINDS[j] == "raw" and SESS_OF_ROW[j] == SESS_OF_ROW[i]:
            nb.append(j)
    NEIGH[i] = nb

# E2 说话人字段: 从"user: "/"assistant: "前缀提取
SPEAKER = {}
for i, r in enumerate(RAWS):
    if KINDS[i] != "raw":
        continue
    mm = _re.match(r"(user|assistant):\s", r)
    SPEAKER[i] = mm.group(1) if mm else ""

# E3 时间街区derive(LME本地): 含时间表达的记录 vs 不含, 256维逐维Δ
MONTHS_E = ["january","february","march","april","may","june","july","august","september","october","november","december"]
def _has_time(txt):
    tl = str(txt).lower()
    return any(m2 in tl for m2 in MONTHS_E) or bool(_re.search(r"(19|20)\d\d", tl))
TLAB_E = np.array([_has_time(r) for r in RAWS])
tproj = np.load(OUT + "/_v256_parts", allow_pickle=False) if False else None
# V已在StageC算好(记录256向量)
Vrec = V  # (n_records, 256)
tmask = TLAB_E & (KINDS_arr == None) if False else TLAB_E
pos_v = Vrec[TLAB_E]
neg_v = Vrec[~TLAB_E]
if len(pos_v) >= 30 and len(neg_v) >= 30:
    tdelta = pos_v.mean(0) - neg_v.mean(0)
    TIME_BLOCK = sorted(np.argsort(-np.abs(tdelta))[:24].tolist())
    sens = (Vrec[:, TIME_BLOCK] * np.sign(tdelta[TIME_BLOCK])).mean(1)
else:
    TIME_BLOCK = []
    sens = np.zeros(len(RAWS))

# E4 RTOK缓存(三团Jaccard用)
RTOK_LME = [set(_re.findall(r"[a-z]{4,}", str(r).lower())) for r in RAWS]

# E5 DF表(锚扩展用)
DF_LME = {}
for rt in RTOK_LME:
    for w in rt:
        DF_LME[w] = DF_LME.get(w, 0) + 1

np.savez_compressed(OUT + "/discovery_fields.npz",
                    neigh_idx=np.array([NEIGH.get(i, []) for i in range(len(RAWS))], dtype=object),
                    time_block=np.array(TIME_BLOCK),
                    sens=sens.astype(np.float32))
with open(OUT + "/speakers.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in SPEAKER.items()}, f)
with open(OUT + "/rtok_cache.json", "w", encoding="utf-8") as f:
    json.dump({str(i): sorted(rt) for i, rt in enumerate(RTOK_LME)}, f)
with open(OUT + "/df_table.json", "w", encoding="utf-8") as f:
    json.dump(DF_LME, f)
print("StageE done: neigh=%d time_block=%s(dims) speakers=%d rtok=%d df=%d" % (
    len(NEIGH), len(TIME_BLOCK), len(SPEAKER), len(RTOK_LME), len(DF_LME)), flush=True)

# ===== lme_index.json: eval侧映射(qid → 去重后session哈希) =====
if not os.path.exists(OUT + "/lme_index.json"):
    qmap = [{"question_id": q["question_id"], "question_type": q["question_type"],
             "question": q["question"], "question_date": q.get("question_date"),
             "answer": q.get("answer"), "answer_sids": q.get("answer_session_ids"),
             "sid2h": {sid: sid2h.get(sid) for sid in q["haystack_session_ids"]}}
            for q in d]
    wjson(OUT + "/lme_index.json", qmap)
print("LME_BUILD_ALL_DONE sessions=%d records=%d questions=%d" % (
    len(hashes), len(RAWS), len(d)), flush=True)
