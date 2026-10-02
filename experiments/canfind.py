# -*- coding: utf-8 -*-
"""canfind.py — 检索成功测试(不看排名, 只看找没找到)
50错+50对, 全库11753条直接检索(不做会话限定!), 三种方法:
  M0 r37基线: 融合top50里有没有证据
  M1 双空间互证: 词级侦察+句级, top50里有没有
  M2 互证+原子句域: M1 + 原子句命中宿主加成
关键: 会话仍限定(LoCoMo题目自带sample_id, 合法), 但看的是"找没找到"而非名次
另测: 完全不限定会话的裸检索(全库11753)——最严苛条件
"""
import io, json, os, sys, re, random
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
def toks_list(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if len(w) > 2]
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
STOPW = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None

atoms = []
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") != "raw":
        continue
    s = (r.get("raw") or "").strip()
    if s.startswith("[Session"):
        continue
    sess = r.get("session_id") or ""
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if len(words) <= 5 and re.search(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank)\b", p, re.I):
            continue
        atoms.append((p, sess, r.get("memory_id")))
A2WV = []
sess2idx = {}
for k2, (p, sess, mid) in enumerate(atoms):
    vs = [wv(w) for w in toks_list(p)]
    vs = [v for v in vs if v is not None]
    A2WV.append(np.stack(vs) if vs else np.zeros((1, 256), dtype=np.float32))
    sess2idx.setdefault(sess, []).append(k2)
A2HOSTMID = [a[2] for a in atoms]
print("原子句:", len(atoms), flush=True)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
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
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
bQ = emb([r["question"] for r in rows])
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

rng = random.Random(2026)
miss_keys = [i for i in keys if rows[i].get("llm_score") != 1]
ok_keys = [i for i in keys if rows[i].get("llm_score") == 1]
sel_miss = rng.sample(miss_keys, 50)
sel_ok = rng.sample(ok_keys, 50)
sel = sel_miss + sel_ok
print("样本: 错50 对50", flush=True)

# 三方法: 全都是"问题丢进库, 看证据是否出现在top50"
def m0_r37(i):
    # r37基线: 会话内融合+精排 top50
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    return set(np.argsort(-newf)[:50].tolist())

def scout(i, sess):
    qws = [wv(w) for w in toks_list(rows[i]["question"]) if w not in STOPW]
    qws = [v for v in qws if v is not None]
    idxs = sess2idx.get(sess, [])
    if not qws or not idxs:
        return {}
    Qm = np.stack(qws)
    out = {}
    for k in idxs:
        wvm = A2WV[k]
        if wvm.shape[0] == 1 and wvm[0].sum() == 0:
            continue
        out[k] = float((Qm @ wvm.T).max(axis=1).mean())
    return out

def m1_dual(i):
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    base = m0_r37(i)
    sc = scout(i, sess)
    newf = np.zeros(N, dtype=np.float32)
    newf[list(base)] = np.linspace(50, 1, 50)
    if sc:
        vals = np.array(list(sc.values()))
        zv = zs(np.concatenate([vals, [0.0]]))[:-1]
        for k2, (k, v) in enumerate(sc.items()):
            j = MID2I.get(A2HOSTMID[k])
            if j is not None:
                newf[j] += 2.5 * zv[min(k2, len(zv) - 1)]
    return set(np.argsort(-newf)[:50].tolist())

def m2_atom(i):
    # M1 + 原子句域二次: 侦察top原子句的"下一句"也带回
    s1 = m1_dual(i)
    sid = rows[i].get("sample_id") or ("conv-" + str(rows[i].get("conversation_idx")))
    sess = "loco-" + str(sid)
    sc = scout(i, sess)
    if not sc:
        return s1
    topk = sorted(sc, key=lambda k: -sc[k])[:3]
    ext = set(s1)
    for k in topk:
        nxt = k + 1
        if nxt < len(atoms) and atoms[nxt][1] == atoms[k][1]:
            j = MID2I.get(A2HOSTMID[nxt])
            if j is not None:
                ext.add(j)
    return ext

stats = {"M0": [0, 0], "M1": [0, 0], "M2": [0, 0]}
miss_fail_m0 = []
for i in sel:
    hs = targets[i]
    ismiss = rows[i].get("llm_score") != 1
    s0, s1, s2 = m0_r37(i), m1_dual(i), m2_atom(i)
    for nm, s in (("M0", s0), ("M1", s1), ("M2", s2)):
        if hs & s:
            stats[nm][0] += 1
        if ismiss and not (hs & s):
            if nm == "M0":
                miss_fail_m0.append(i)
print()
print("== 检索成功测试(证据∈top50, 全库N=%d) ==" % N, flush=True)
print("  M0 r37基线:    %d/100" % stats["M0"][0], flush=True)
print("  M1 双空间互证: %d/100" % stats["M1"][0], flush=True)
print("  M2 互证+邻句:  %d/100" % stats["M2"][0], flush=True)
# 错题子集(r37原本失败的)
fails = set(miss_fail_m0)
print()
print("== M0检索失败的错题(%d道)能否被M1/M2救回 ==" % len(fails), flush=True)
s1s = s2s = 0
for i in fails:
    s1 = m1_dual(i)
    s2 = m2_atom(i)
    hs = targets[i]
    s1s += bool(hs & s1)
    s2s += bool(hs & s2)
print("  M1救回: %d/%d | M2救回: %d/%d" % (s1s, len(fails), s2s, len(fails)), flush=True)
print("CANFIND_DONE", flush=True)
