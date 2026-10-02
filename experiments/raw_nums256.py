# -*- coding: utf-8 -*-
"""raw_nums256.py — 256维空间的同样解剖 + 槽位的数字底账
S1 256维三组解剖: 答案/证据/噪声的数组关系(共模? 相关? 与1024对照)
S2 256维的Δ256为什么失败: 逐维差幅 vs 1024的0.013, 谱结构对比
S3 槽位数字底账: 日期槽(When题的证据session日期 vs 问题日期键的匹配率分布)
S4 人名槽: 问题人名在证据句vs噪声句中的出现率差(真正的槽位信号强度)
S5 时间槽的槽位-名次曲线: 带日期的记录在top榜的实际名次优势
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
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
rows = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))
        if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb_bge(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def emb_qwen(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
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
REC = {}
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
def rec_text(r):
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text(REC.get(m, {})) for m in MID]
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
MID2I = {m: i for i, m in enumerate(MID)}
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

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
TOP50 = z1["TOP50"]

ev_ids, no_ids = [], []
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    ev_ids += [j for j in top if j in hs or j in tws]
    no_ids += [j for j in top if j not in hs and j not in tws][:6]
ev_ids, no_ids = np.array(ev_ids), np.array(no_ids)
A256 = emb_qwen([str(r["answer"][0]) for r in rows])
E256 = QW[ev_ids]
NO256 = QW[no_ids]
print("S1 256维三组解剖:", flush=True)
me, mn, ma = E256.mean(0), NO256.mean(0), A256.mean(0)
print("  证据vs噪声 跨维相关=%.3f  [1024版=0.986]" % np.corrcoef(me, mn)[0, 1], flush=True)
print("  证据vs答案=%.3f | 噪声vs答案=%.3f  [1024: 0.875/0.854]" % (
    np.corrcoef(me, ma)[0, 1], np.corrcoef(mn, ma)[0, 1]), flush=True)
diff = me - mn
print("S2 Δ256: |差|最大=%.4f 中位=%.4f  [1024: 0.0133/0.0027]" % (
    np.abs(diff).max(), np.median(np.abs(diff))), flush=True)
top5 = np.argsort(-np.abs(diff))[:5]
print("   top5维:", top5.tolist(), "值:", np.round(diff[top5], 4).tolist(), flush=True)
# 回归: 答案 = a*证据 + b*噪声
X = np.stack([me, mn]).T
coef, *_ = np.linalg.lstsq(X, ma, rcond=None)
r2 = 1 - np.sum((ma - X @ coef) ** 2) / np.sum((ma - ma.mean()) ** 2)
print("   答案≈%.2f×证据+%.2f×噪声 (R²=%.3f) [1024: 0.51/0.51/-0.35q R²见主跑]" % (coef[0], coef[1], r2), flush=True)

# 谱: 256空间的维有效秩
U, S, _ = np.linalg.svd(QW[:4000] - QW[:4000].mean(0), full_matrices=False)
eff = (S ** 2).sum() / (S[0] ** 2)
print("S2b 256空间有效秩=%.0f/256  [1024: 22/1024]" % eff, flush=True)

# S3 日期槽底账
MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
def norm_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(1).lower() in MONTHS:
        return "%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()])
    return ""
SESS_DATE = {}
cur_d = ""
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    try:
        rr = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session \d+ — (.+?)\]", (rr.get("raw") or "").strip())
    if m:
        cur_d = m.group(1).strip()
    SESS_DATE[rr.get("memory_id")] = cur_d
qfile = [json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()]
qmap = {q["qa_id"]: q for q in qfile}
when_q = [r for r in rows if re.match(r"(?i)^when\b", r["question"])]
date_keys = set()
exact = month = none_ = 0
for r in when_q:
    q = qmap.get(r["qa_id"])
    qd = q.get("session_date") or ""
    for m in re.finditer(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})", r["question"], re.I):
        date_keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", r["question"], re.I):
        date_keys.add("%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()]))
    if not date_keys:
        none_ += 1
print("S3 日期槽: When题%d道, 问题含显式日期键%d道 (%.0f%%无键=日期槽无输入)" % (
    len(when_q), len(when_q) - none_, 100 * none_ / max(1, len(when_q))), flush=True)

# S4 人名槽强度
def names_of(r):
    return {str(r.get(f)).lower() for f in ("speaker_a", "speaker_b") if r.get(f)}
ev_name = no_name = 0
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    nms = names_of(rows[i])
    top = TOP50[i]
    for j in top:
        t = TEXTS[j].lower()
        hitn = any(n in t for n in nms)
        if j in hs or j in tws:
            ev_name += hitn
        else:
            no_name += hitn
print("S4 人名槽: 证据句含主角名%.0f%% | 噪声句含主角名%.0f%% (差=%.0fpp=槽位真实强度)" % (
    100 * ev_name / max(1, ev_name + 0) if False else 0, 0, 0), flush=True)
# 上式复杂, 直接算两个比例
ev_cnt = sum(1 for i in keys for j in TOP50[i]
             if (j in targets[i]) or (j in {MID2I[REC[MID[h]].get("raw_of")] for h in targets[i] if REC.get(MID[h], {}).get("raw_of") in MID2I}))
print("  (证据句含名比例需按池算, 见下行)", flush=True)
ev_hit = ev_name
# 正确版本: 分别统计
tot_ev = tot_no = 0
for i in keys:
    hs = targets[i]
    tws = {MID2I[REC[MID[h]].get("raw_of")] for h in hs if REC.get(MID[h], {}).get("raw_of") in MID2I}
    nms = names_of(rows[i])
    for j in TOP50[i]:
        t = TEXTS[j].lower()
        h = any(n in t for n in nms)
        if j in hs or j in tws:
            tot_ev += 1; ev_name += h
        else:
            tot_no += 1; no_name += h
print("S4 人名槽(正确): 证据含名%.1f%% vs 噪声含名%.1f%% → 槽位区分度=%.1fpp" % (
    100 * ev_name / tot_ev, 100 * no_name / tot_no, 100 * (ev_name / tot_ev - no_name / tot_no)), flush=True)

# S5 日期记录的名次优势: 问题含日期键时, 日期匹配记录的平均名次 vs 不匹配
adv_m, adv_n = [], []
for i in keys:
    r = rows[i]
    if not re.match(r"(?i)^when\b", r["question"]):
        continue
    q = qmap.get(r["qa_id"])
    dks = set()
    for m in re.finditer(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})", r["question"], re.I):
        dks.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1))))
    if not dks:
        continue
    for rk, j in enumerate(TOP50[i][:25], start=1):
        rd = norm_date(SESS_DATE.get(MID[j]) or "")
        if not rd:
            continue
        if rd in dks or any(k.startswith(rd) or rd.startswith(k) for k in dks):
            adv_m.append(rk)
        else:
            adv_n.append(rk)
print("S5 日期匹配记录平均名次=%.1f vs 不匹配=%.1f (差=%.1f位)" % (
    np.mean(adv_m) if adv_m else -1, np.mean(adv_n) if adv_n else -1,
    (np.mean(adv_n) - np.mean(adv_m)) if adv_m and adv_n else -1), flush=True)
print("R256_DONE", flush=True)
