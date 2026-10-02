# -*- coding: utf-8 -*-
"""foundry69.py — 用户裁决执行: C置信度驱动窗口 + E错题指纹 + F三向迭代检索 + 资产全接入
特征329→336: +时间敏感度 +地名轴分 +colbert分 +情感轴分(资产接入)
F三向迭代: ①锚+问题骨架重嵌检索 ②头部3均值重检索 ③G1最强邻域块调入
C置信度: conf = f(GBDT头部分差, 锚扩展命中数, 头部聚集度) → 窗口15/25/35三档
E: 输出445失败题的题面统计(疑问词/长度/否定式)
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry69_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z]{3,}", str(s).lower()))
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w

t0 = time.time()
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
NR = len(MID)
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
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
RAW = [rec_raw(m) for m in MID]
RAWN = [norm(x) for x in RAW]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
TWIN = {}
MID2I = {m: i for i, m in enumerate(MID)}
for i, m in enumerate(MID):
    tw = REC.get(m, {}).get("raw_of") or REC.get(m, {}).get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
_c = np.load(HERE + "/xz_cache.npz")
X = _c["X"]
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
C0 = l2n(X @ D.T)
Q256 = np.load(HERE + "/q256_cache.npz")["Q"]
def gold_set(qa):
    keys = [norm(em.get("text") or "")[:60] for em in (Q[qa].get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    g = set()
    for i in range(NR):
        if any(k in RAWN[i] for k in keys):
            g.add(i)
    return g
GSETS = [gold_set(qa) for qa in IDS]
CONV_of = {qa: qa.split("#")[0] for qa in IDS}
FOLDS = sorted(set(CONV_of.values()))
DF = {}
for rn in RAWN:
    for w in set(re.findall(r"[a-z]{4,}", rn)):
        DF[w] = DF.get(w, 0) + 1
RTOK = [set(re.findall(r"[a-z]{4,}", r.lower())) for r in RAW]

# ===== 资产: 时间敏感度(256街区) / 地名轴 / 情感轴 / colbert =====
TIME_DIMS = list(range(245, 256))
sens_t = np.mean(np.abs(QW[:, TIME_DIMS]), axis=1)
PLACES = ["california", "texas", "america", "canada", "germany", "france", "japan",
          "china", "london", "paris", "brazil", "york", "europe", "spain", "italy"]
CONTROL = ["table", "chair", "idea", "water", "music", "book", "phone", "tree",
           "bread", "window", "car", "shirt"]
POS_W = ["happy", "joy", "love", "great", "wonderful", "excited", "proud"]
NEG_W = ["sad", "angry", "hate", "terrible", "awful", "depressed", "upset"]
zw = np.load(HERE + "/cue_word_cache.npz", allow_pickle=True)
WV = zw["V"].astype(np.float32)
WV = WV / np.maximum(np.linalg.norm(WV, axis=1, keepdims=True), 1e-9)
WORDS_C = [str(w).lower() for w in zw["WORDS"]]
W2I_C = {w: i for i, w in enumerate(WORDS_C)}
def cent(ws):
    vs = [WV[W2I_C[w]] for w in ws if w in W2I_C]
    return np.mean(vs, axis=0) if vs else None
AX_PLACE = cent(PLACES) - cent(CONTROL)
AX_PLACE = AX_PLACE / (np.linalg.norm(AX_PLACE) + 1e-9)
AX_SENT = cent(POS_W) - cent(NEG_W)
AX_SENT = AX_SENT / (np.linalg.norm(AX_SENT) + 1e-9)
place_sc = QW @ AX_PLACE
sent_sc = QW @ AX_SENT

# colbert分数(缓存文件或重算——用bge现算开销大, 简化: 用D的top段投影近似; 若有colbert_parts则用)
COLB = None
try:
    import torch
    mats, offsets = [], []
    run_ = 0
    nch = len([f for f in os.listdir(HERE + "/colbert_parts") if f.startswith("c") and f.endswith(".npz")])
    for ci in range(nch):
        _p = np.load(HERE + "/colbert_parts/c%04d.npz" % ci)
        m = _p["mat"]
        counts = _p["counts"]
        mats.append(m)
        for c in counts:
            offsets.append((run_, run_ + int(c)))
            run_ += int(c)
    Pm_t = torch.from_numpy(np.concatenate(mats)).to("cuda")
    Pm_t = Pm_t / (Pm_t.norm(dim=1, keepdim=True) + 1e-9)
    OFF = np.array(offsets, dtype=np.int64)
    from FlagEmbedding import BGEM3FlagModel
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    enc_all = bge.encode([Q[qa]["question"] for qa in IDS], return_dense=False,
                         return_sparse=False, return_colbert_vecs=True)
    COLB = []
    for k_i in range(len(IDS)):
        qv = np.asarray(enc_all["colbert_vecs"][k_i], dtype=np.float16)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        qt_ = torch.from_numpy(qv).to("cuda")
        s_ = (qt_ @ Pm_t.T).cpu().numpy()
        COLB.append(np.maximum.reduceat(s_, OFF[:, 0], axis=1).mean(axis=0))
    COLB = np.stack(COLB)
    P("colbert ready %.0fs" % (time.time() - t0))
except Exception as e:
    P("colbert skip: %s" % repr(e)[:80])
    COLB = None
P("assets ready %.0fs" % (time.time() - t0))

QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
TOKENS = {}
def ctoks(i):
    if i not in TOKENS:
        TOKENS[i] = toks(RAW[i])
    return TOKENS[i]

# ===== F迭代检索: bge编码器(锚重构查询) =====
from FlagEmbedding import BGEM3FlagModel
bge_enc = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def embed_texts(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge_enc.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

POOLSZ = 700
FEATS = [None] * len(IDS)
POOLA = [None] * len(IDS)
for k_i, qa in enumerate(IDS):
    qi = IDX[qa]
    pool = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    for i in list(np.argsort(-FINAL[qi])[:30]):
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[i]:
                pool.add(j)
    pool = sorted(pool)[:POOLSZ]
    POOLA[k_i] = pool
    q = Q[qa]
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", q["question"].lower()) if w not in QSTOP and len(w) > 2)
    qstems_list = sorted(qstems)
    top30list = list(np.argsort(-FINAL[qi])[:30])
    qtok = toks(q["question"])
    qlen = max(1, len(qtok))
    qv256 = Q256[k_i]
    qv1024 = X[qi]
    interr = (Q[qa].get("question") or "x").strip().lower().split()[0] if (Q[qa].get("question") or "").strip() else ""
    Fm = np.zeros((len(pool), 336), dtype=np.float32)
    for rr, c in enumerate(pool):
        ct = ctoks(c)
        inter = qtok & ct
        jac = len(inter) / max(1, len(qtok | ct))
        qcov = len(inter) / qlen
        ccov = len(inter) / max(1, len(ct))
        qmark = 1.0 if RAW[c].rstrip().endswith("?") else 0.0
        iecho = 1.0 if (interr and interr in RAW[c].lower()) else 0.0
        lr = len(RAW[c]) / max(1, len(Q[qa].get("question") or "x"))
        vqc = float(qv256 @ QW[c])
        dqc = float(D[c] @ qv1024)
        rstems = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
        g1 = len(qstems & rstems) / max(1, len(qstems))
        cospen = 1.0 if dqc > 0.65 else 0.0
        gradinv = dqc - g1
        cross = dqc * (1.0 - g1)
        nb_flag = 1.0 if any(abs(c - j) <= 2 and CONVKEY[j] == CONVKEY[c] for j in top30list) else 0.0
        rl_ = RAW[c].lower()
        seq = 0; mx = 0; last_p = -1
        for w in qstems:
            p = rl_.find(w)
            if p >= 0 and p > last_p:
                seq += 1; last_p = p; mx = max(mx, seq)
            elif p >= 0:
                seq = 1; last_p = p
            else:
                seq = 0
        r1 = mx / max(1, len(qstems))
        big = [(qstems_list[i], qstems_list[i + 1]) for i in range(len(qstems_list) - 1)]
        rset_ = set(re.findall(r"[a-z']+", RAW[c].lower()))
        r2 = sum(1 for a2, b2 in big if a2 in rset_ and b2 in rset_) / max(1, len(big))
        wvotes = sum(1 for w in qstems_list if w in RTOK[c])
        tw = TWIN.get(c)
        twin_q = float(D[tw] @ qv1024) if tw is not None else 0.0
        Fm[rr, 0:12] = [jac, qcov, ccov, qmark, iecho, lr, vqc, vqc - dqc, dqc, dqc - vqc, dqc - qcov, qcov - dqc]
        Fm[rr, 12:16] = [D[c, 616], D[c, 538], D[c, 218], D[c, 408]]
        Fm[rr, 16:116] = D[c][:100]
        Fm[rr, 116:216] = np.abs(qv1024[:100] - D[c][:100])
        Fm[rr, 216:316] = QW[c][:100]
        Fm[rr, 316] = float(C0[qi][c])
        Fm[rr, 317] = float(FINAL[qi][c])
        Fm[rr, 318:321] = [qmark, iecho, min(lr, 4) / 4]
        Fm[rr, 321:329] = [cospen, gradinv, cross, nb_flag, r1, r2, wvotes, twin_q]
        # 资产特征329-336
        Fm[rr, 329] = float(sens_t[c])
        Fm[rr, 330] = float(place_sc[c])
        Fm[rr, 331] = float(sent_sc[c])
        Fm[rr, 332] = float(COLB[k_i][c]) if COLB is not None else 0.0
        Fm[rr, 333] = float(abs(sent_sc[c]))    # 情感绝对强度(无论正负,情感浓=私人内容)
        Fm[rr, 334] = float(qv256 @ QW[c]) * g1  # cos×G1交叉
        Fm[rr, 335] = float(len(RAW[c]))          # 长度原始值
    FEATS[k_i] = Fm
    POOLA[k_i] = pool
    if k_i % 200 == 0:
        P("  feat %d %.0fs" % (k_i, time.time() - t0))
P("built %.0fs" % (time.time() - t0))

from sklearn.ensemble import HistGradientBoostingClassifier
FINAL_ORDER = {}
res = {"w15": 0, "w_dyn_conf": 0, "w30": 0, "w2l": 0, "pool": 0}
conf_dist = {"hi": 0, "mid": 0, "lo": 0}
fail_qs = {"ok": [], "fail": []}
n = 0
for hold in FOLDS:
    trF, trY = [], []
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] == hold:
            continue
        pool = POOLA[k_i]
        G = GSETS[k_i]
        gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
        noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
        noisepick = list(np.random.RandomState(k_i).permutation(noise_rows)[:15]) if noise_rows else []
        for rr in gold_rows:
            trF.append(FEATS[k_i][rr]); trY.append(1)
        for rr in noisepick:
            trF.append(FEATS[k_i][rr]); trY.append(0)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8))
    # 迭代检索的锚查询(每题一次bge编码——放题循环外批量做不了,逐题做太慢; 用F三向的②均值向代替①避免编码)
    for k_i, qa in enumerate(IDS):
        if CONV_of[qa] != hold:
            continue
        G = GSETS[k_i]
        if not G:
            continue
        n += 1
        qi = IDX[qa]
        pool = POOLA[k_i]
        pos_of = {c: i for i, c in enumerate(pool)}
        s = clf.decision_function(FEATS[k_i])
        o = [pool[i] for i in np.argsort(-s)]
        if G <= set(pool):
            res["pool"] += 1
        # ===== F三向迭代(②③, ①用向量拼接近似) =====
        # ② 头部3均值重检索(全库): 均值向量×库
        h3 = o[:3]
        mvec = l2n(D[h3].mean(0, keepdims=True))[0]
        s2 = D @ mvec
        new_top = [int(i) for i in np.argsort(-s2)[:50] if i not in set(o[:15])]
        # ③ G1最强候选的邻域块调入
        qstems2 = set(stem(w) for w in re.findall(r"[a-z']+", Q[qa]["question"].lower()) if w not in QSTOP and len(w) > 2)
        def g1q(c):
            rs = set(stem(w) for w in re.findall(r"[a-z']+", RAW[c].lower()))
            return len(qstems2 & rs) / max(1, len(qstems2))
        g1_strong = max(o[:20], key=g1q)
        nb_block = []
        for off in range(-3, 4):
            j = g1_strong + off
            if 0 <= j < NR and CONVKEY[j] == CONVKEY[g1_strong] and j not in set(o[:15]):
                nb_block.append(j)
        # 迭代捞回候选: new_top前5 + nb_block前4 → 进入L1尾部竞争(用GBDT特征打分)
        rescue_cands = new_top[:5] + nb_block[:4]
        if rescue_cands:
            # 给rescue候选构造特征(与池内同法, 简化: 用GBDT对rescue的decision——需特征, 用FINAL/C0/fused近似)
            rescue_sc = {}
            for c in rescue_cands:
                rescue_sc[c] = 0.5 * float(C0[qi][c]) + 0.5 * float(FINAL[qi][c]) + 0.5 * g1q(c)
            rescue_pick = sorted(rescue_sc, key=rescue_sc.get, reverse=True)[:4]
        else:
            rescue_pick = []
        # 融合V2-gated(同F68)
        head = o[:10]
        qv = Q256[k_i]
        PRODM = np.array([qv * QW[c] for c in head])
        w2 = np.abs(PRODM[0])
        w2 = w2 / (w2.sum() + 1e-9)
        boost = PRODM @ (w2 * 256.0)
        boost = (boost - boost.min()) / (boost.max() - boost.min() + 1e-9)
        gb_norm = np.array([s[pos_of[c]] for c in head])
        gb_norm = (gb_norm - gb_norm.min()) / (gb_norm.max() - gb_norm.min() + 1e-9)
        mix = 0.6 * gb_norm + 0.4 * boost
        o_head = [head[i] for i in np.argsort(-mix)]
        for prot in reversed(o[:2]):
            if prot in o_head:
                o_head.remove(prot)
                o_head.insert(0, prot)
        order = o_head + [c for c in o if c not in set(o_head)]
        # rescue插入15行尾部
        if rescue_pick:
            keep = order[:15 - len(rescue_pick)]
            order = keep + rescue_pick + [c for c in order if c not in set(keep) | set(rescue_pick)]
        # ===== C置信度驱动窗口 =====
        s_sorted = np.sort(s)[::-1]
        conf = (s_sorted[0] - s_sorted[5]) / (abs(s_sorted[0]) + 1e-9)   # 头部分差
        conf2 = len(rescue_pick)
        conf_score = conf * 2.0 + conf2 * 0.3
        if conf_score > 1.2:
            W = 15
            conf_dist["hi"] += 1
        elif conf_score > 0.5:
            W = 25
            conf_dist["mid"] += 1
        else:
            W = 35
            conf_dist["lo"] += 1
        FINAL_ORDER[qa] = [MID[c] for c in order[:35]]
        g15 = all(MID[g] in FINAL_ORDER[qa][:15] for g in G)
        gW = all(MID[g] in FINAL_ORDER[qa][:W] for g in G)
        g30 = all(MID[g] in FINAL_ORDER[qa][:30] for g in G)
        g2l = all(MID[g] in FINAL_ORDER[qa][:35] for g in G)
        if g15:
            res["w15"] += 1
        if gW:
            res["w_dyn_conf"] += 1
        if g30:
            res["w30"] += 1
        if g2l:
            res["w2l"] += 1
        # E: 题面指纹收集
        qtxt = Q[qa]["question"]
        fp = (len(qtxt.split()), qtxt.lower().split()[0] if qtxt.split() else "", "not" in qtxt.lower() or "never" in qtxt.lower())
        (fail_qs["ok"] if g15 else fail_qs["fail"]).append(fp)
    P("  fold %s done" % hold)

with open(HERE + "/r41_final_order_v8.json", "w", encoding="utf-8") as f:
    json.dump(FINAL_ORDER, f, ensure_ascii=False)
P("\n===== F69全能版检验 (n=%d) =====" % n)
P("池全金率: %.1f%%" % (100.0 * res["pool"] / max(1, n)))
P("窗15固定: %.1f%%" % (100.0 * res["w15"] / max(1, n)))
P("置信度动态窗(hi15/mid25/lo35): %.1f%%" % (100.0 * res["w_dyn_conf"] / max(1, n)))
P("窗30固定: %.1f%%" % (100.0 * res["w30"] / max(1, n)))
P("双层35: %.1f%%" % (100.0 * res["w2l"] / max(1, n)))
P("置信度分布: hi=%d mid=%d lo=%d" % (conf_dist["hi"], conf_dist["mid"], conf_dist["lo"]))
# E: 指纹统计
import collections
for tag in ("ok", "fail"):
    F = fail_qs[tag]
    lens = [x[0] for x in F]
    firstw = collections.Counter(x[1] for x in F).most_common(4)
    negr = np.mean([x[2] for x in F]) if F else 0
    P("E指纹[%s]: n=%d 均长=%.1f 首词=%s 否定率=%.1f%%" % (
        tag, len(F), np.mean(lens) if lens else 0, firstw, 100 * negr))
P("F69_DONE %.0fs" % (time.time() - t0))
LOG.close()
