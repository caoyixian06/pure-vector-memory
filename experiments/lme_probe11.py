# -*- coding: utf-8 -*-
"""lme_probe11.py — 三死信号的向量空间验尸+粒度受控实验:
①三团: 同批句子聚合回记录粒度,团是否恢复(粒度属性的直接证据)
②问号的向量对应: 问句vs陈述句的可分性
③时态的向量对应: ed密度方向derive(类时间街区)+金噪投影差"""
import io, json, os, re, sys, time, hashlib
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def P(s):
    print(s)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)
P("loaded %.0fs" % (time.time() - t0))

# 采样: 记录级(金会话12条+陪衬12条/题), 保留句子归属
rng = np.random.RandomState(0)
gold_recs, noise_recs = [], []   # (qi, rec_idx, is_gold)
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    gold_keys = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    nr = []
    for k2 in set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h):
        if k2 not in gold_keys:
            nr += SID2ROWS.get(k2, [])
    if not gs or len(nr) < 20:
        continue
    for i in rng.choice(sorted(gs), size=min(8, len(gs)), replace=False):
        gold_recs.append((qi, int(i)))
    for i in rng.choice(sorted(set(nr)), size=min(8, len(nr)), replace=False):
        noise_recs.append((qi, int(i)))
P("金记录=%d 陪衬记录=%d %.0fs" % (len(gold_recs), len(noise_recs), time.time() - t0))

# 记录文本→句子
def split_sents(rec):
    out = []
    for m in re.finditer(r"[^.!?]*[.?]?", RAWS[rec]):
        p = m.group(0).strip()
        if 4 <= len(p.split()) <= 45:
            out.append(p)
    return out

rec_sents = {}   # rec -> [句子]
for _, rec in gold_recs + noise_recs:
    if rec not in rec_sents:
        rec_sents[rec] = split_sents(rec)
all_sents = [s for rec in rec_sents for s in rec_sents[rec]]
sent_owner = [rec for rec in rec_sents for _ in rec_sents[rec]]
P("句子总数=%d(记录数=%d) %.0fs" % (len(all_sents), len(rec_sents), time.time() - t0))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
SV = []
for st in range(0, len(all_sents), 64):
    SV.append(np.asarray(bge.encode(all_sents[st:st + 64])["dense_vecs"], dtype=np.float32))
SV = l2n(np.concatenate(SV))
P("句子嵌入 %.0fs" % (time.time() - t0))

# ===== ① 粒度受控: 句子粒度三团 vs 句子聚合回记录的三团 =====
gold_set = set(r for _, r in gold_recs)
def triad(pick_gold, pick_noise, vecs, owners):
    rng2 = np.random.RandomState(1)
    gg, gn_, nn = [], [], []
    G = [i for i in range(len(owners)) if owners[i] in gold_set]
    N = [i for i in range(len(owners)) if owners[i] not in gold_set]
    for _ in range(3000):
        a, b = rng2.choice(G, 2) if len(G) > 1 else (0, 0)
        gg.append(float(vecs[a] @ vecs[b]))
    for _ in range(3000):
        a = rng2.choice(G); b = rng2.choice(N)
        gn_.append(float(vecs[a] @ vecs[b]))
    for _ in range(3000):
        a, b = rng2.choice(N, 2)
        nn.append(float(vecs[a] @ vecs[b]))
    return np.mean(gg), np.mean(gn_), np.mean(nn)

gg1, gn1, nn1 = triad(None, None, SV, sent_owner)
# 聚合回记录: 记录向量=其句子均值
rec_list = [r for r in rec_sents if rec_sents[r]]
rec_vec = np.zeros((len(rec_list), SV.shape[1]), dtype=np.float32)
rec_of_sent = []
for ri, rec in enumerate(rec_list):
    idxs = [i for i, o in enumerate(sent_owner) if o == rec]
    rec_vec[ri] = l2n(SV[idxs].mean(0, keepdims=True))[0]
gold_rset = set(rec_list) & gold_set
gg2, gn2, nn2 = triad(None, None, rec_vec, rec_list)
P("\n①粒度受控三团(cos):")
P("  句子粒度:   金金=%.3f 金噪=%.3f 噪噪=%.3f (差%.3f)" % (gg1, gn1, nn1, gg1 - gn1))
P("  聚合回记录: 金金=%.3f 金噪=%.3f 噪噪=%.3f (差%.3f)" % (gg2, gn2, nn2, gg2 - gn2))
P("  (同批数据,聚合后差恢复→粒度属性实锤)")

# ===== ② 问号的向量对应: 问句vs陈述句可分性 =====
is_q = np.array([s.rstrip().endswith("?") for s in all_sents])
if is_q.sum() >= 50 and (~is_q).sum() >= 50:
    qc = l2n(SV[is_q].mean(0, keepdims=True))[0]
    sc = l2n(SV[~is_q].mean(0, keepdims=True))[0]
    P("\n②问号向量对应: 问句n=%d 陈述n=%d 中心cos=%.3f  方向差=%.3f" % (
        is_q.sum(), (~is_q).sum(), float(qc @ sc), np.linalg.norm(qc - sc)))
    qproj = SV @ (qc - sc)
    from bisect import bisect_left
    def auc(pos, neg):
        allv = sorted(list(pos) + list(neg))
        return (sum(bisect_left(allv, v) + 1 for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    P("  问句方向投影AUC: %.3f (问句vs陈述句在向量空间的可分性)" % auc(qproj[is_q], qproj[~is_q]))

# ===== ③ 时态向量对应: ed密度方向derive =====
ed_dens = np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w.endswith("ed")) / max(1, len(s.split())) for s in all_sents])
hi = ed_dens > np.quantile(ed_dens, 0.85)
lo = ed_dens == 0
if hi.sum() >= 30:
    delta = l2n(SV[hi].mean(0, keepdims=True))[0] - l2n(SV[lo].mean(0, keepdims=True))[0]
    proj = SV @ delta
    P("\n③时态向量对应: 高ed句n=%d 低edn=%d  Δ方向derive完成" % (hi.sum(), lo.sum()))
    P("  高ed投影均值=%.3f 低ed=%.3f (方向有效性)" % (proj[hi].mean(), proj[lo].mean()))
    # 金vs噪在该方向的差
    proj_g = [proj[i] for i in range(len(sent_owner)) if sent_owner[i] in gold_set]
    proj_n = [proj[i] for i in range(len(sent_owner)) if sent_owner[i] not in gold_set]
    P("  金句ed投影=%.3f 陪衬句=%.3f (词面打平,向量投影是否有差)" % (np.mean(proj_g), np.mean(proj_n)))
P("done %.0fs" % (time.time() - t0))
