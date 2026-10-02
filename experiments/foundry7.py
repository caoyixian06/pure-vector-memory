# -*- coding: utf-8 -*-
"""foundry7.py — 词向量线索解码器: 已验证结构在真实题目词表上的广度+意图解码率
零API零GPU(纯256词向量)。对照法防共模假象(目标种子集 vs 中性名词集)。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry7_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()

t0 = time.time()
import urllib.request
def oemb(texts):
    out = []
    for s in range(0, len(texts), 64):
        for att in range(4):
            try:
                body = json.dumps({"model": "qwen3-embedding:latest", "input": texts[s:s + 64],
                                   "dimensions": 256}).encode()
                req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                break
            except Exception as e:
                print("oemb retry", repr(e)[:60], flush=True)
                time.sleep(3 * (att + 1))
        else:
            raise RuntimeError("oemb fail at %d" % s)
    return np.concatenate(out)
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
VALENCE_POS = ["happy", "joy", "love", "great", "wonderful", "excited", "proud", "amazing"]
VALENCE_NEG = ["sad", "angry", "hate", "terrible", "awful", "depressed", "upset", "boring"]
NAMES = ["john", "mary", "james", "sarah", "mike", "dave", "caroline", "melanie",
         "nate", "joanna", "deborah", "james", "evan", "sam", "maria", "gina",
         "calvin", "andrew", "audrey", "jolene", "tim", "john"]
PLACES = ["california", "texas", "america", "canada", "germany", "france", "japan",
          "china", "london", "paris", "brazil", "york", "europe", "spain", "italy"]
CONTROL = ["table", "chair", "idea", "water", "music", "book", "phone", "tree",
           "bread", "window", "car", "shirt"]
CACHE = HERE + "/cue_word_cache.npz"
if os.path.exists(CACHE):
    _c = np.load(CACHE, allow_pickle=True)
    WORDS, V = list(_c["WORDS"]), _c["V"]
else:
    Q0 = {}
    for l in io.open(QS, encoding="utf-8"):
        q = json.loads(l)
        Q0[q["qa_id"]] = q
    qtoks = set()
    for q in Q0.values():
        for w in re.findall(r"[a-z]+", str(q.get("question")).lower()):
            if len(w) > 2:
                qtoks.add(w)
    SEEDS_ALL = (MONTHS + WEEK + VALENCE_POS + VALENCE_NEG + NAMES + PLACES
                 + CONTROL + ["when", "where", "who", "why", "how", "because"])
    WORDS = sorted(qtoks | set(SEEDS_ALL))
    V = oemb(WORDS)
    np.savez(CACHE, WORDS=np.array(WORDS), V=V)
V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
W2I = {w: i for i, w in enumerate(WORDS)}
P("词表(题目实词,ollama现嵌)=%d" % len(WORDS))

def vec(w):
    i = W2I.get(w)
    return V[i] if i is not None else None
def centroid(ws):
    vs = [vec(w) for w in ws if vec(w) is not None]
    if not vs:
        return None
    M = np.stack(vs)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

# ===== 种子结构 =====


Mv = centroid(MONTHS)
Wv = centroid(WEEK)
Nm = centroid(NAMES)
Pl = centroid(PLACES)
CV = centroid(CONTROL)
P("种子加载完成 %.0fs" % (time.time() - t0))

# ===== 结构复验(在真词表上) =====
P("\n===== 结构复验 =====")
mi = [W2I.get(m) for m in MONTHS]
mi = [i for i in mi if i is not None]
Mm = V[mi]
adj = [float(Mm[i] @ Mm[(i + 1) % 12]) for i in range(12)]
far = [float(Mm[i] @ Mm[(i + 6) % 12]) for i in range(12)]
P("月环: 相邻cos=%.3f 远隔cos=%.3f (真序应 adj>far)" % (np.mean(adj), np.mean(far)))
for a, b, tag in [(VALENCE_POS, VALENCE_NEG, "情感价轴"), (NAMES, CONTROL, "人名vs中性"),
                  (PLACES, CONTROL, "地名vs中性"), (MONTHS, CONTROL, "月份vs中性")]:
    A, B = centroid(a), centroid(b)
    if A is None or B is None:
        continue
    axis = A.mean(0) - B.mean(0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    pa = float((A.mean(0) @ axis) / (np.linalg.norm(A.mean(0)) + 1e-9))
    pb = float((B.mean(0) @ axis) / (np.linalg.norm(B.mean(0)) + 1e-9))
    P("%s: 种子侧投影=%.3f 对照侧投影=%.3f (差=%.3f)" % (tag, pa, pb, pa - pb))

# ===== 真实题目扫描(对照法) =====
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
def toks(s):
    return re.findall(r"[a-z]+", str(s).lower())

def cue_scores(tokens):
    ts = {}
    best = {"time": 0.0, "who": 0.0, "place": 0.0, "val": 0.0}
    for w in tokens:
        i = W2I.get(w)
        if i is None:
            continue
        v = V[i]
        ct = float(np.max(V[[W2I[m] for m in MONTHS if m in W2I]] @ v)) if any(m in W2I for m in MONTHS) else 0.0
        cc = float(np.max(V[[W2I[m] for m in CONTROL if m in W2I]] @ v))
        cn = float(np.max(V[[W2I[m] for m in NAMES if m in W2I]] @ v))
        cw = float(np.max(V[[W2I[m] for m in WEEK if m in W2I]] @ v))
        cp = float(np.max(V[[W2I[m] for m in PLACES if m in W2I]] @ v))
        contrast_t = ct - cc
        contrast_p = max(cn, cp) - cc
        if contrast_t > best["time"]:
            best["time"] = contrast_t
        if contrast_p > best["who"]:
            best["who"] = contrast_p
        va = float(v @ (centroid(VALENCE_POS).mean(0) - centroid(VALENCE_NEG).mean(0)))
        if abs(va) > abs(best["val"]):
            best["val"] = va
        ts[w] = (contrast_t, contrast_p)
    return best, ts

P("\n===== 题目扫描(每题最强线索, 对照法) =====")
gold_type = {}
for qa, q in Q.items():
    ans = " ".join(str(x) for x in (q.get("answer") or [])).lower()
    mon = any(m in ans for m in MONTHS) or re.search(r"\b(19|20)\d\d\b", ans)
    if mon:
        gold_type[qa] = "time"
        continue
    names_hit = [n for n in set(NAMES) if n in ans]
    if names_hit:
        gold_type[qa] = "who"
        continue
    pl_hit = [p for p in PLACES if p in ans]
    if pl_hit:
        gold_type[qa] = "place"
        continue
    gold_type[qa] = "other"

AX_T = centroid(MONTHS).mean(0) - centroid(CONTROL).mean(0)
AX_T = AX_T / (np.linalg.norm(AX_T) + 1e-9)
AX_W = centroid(NAMES).mean(0) - centroid(CONTROL).mean(0)
AX_W = AX_W / (np.linalg.norm(AX_W) + 1e-9)
AX_P = centroid(PLACES).mean(0) - centroid(CONTROL).mean(0)
AX_P = AX_P / (np.linalg.norm(AX_P) + 1e-9)
AX_V = centroid(VALENCE_POS).mean(0) - centroid(VALENCE_NEG).mean(0)
AX_V = AX_V / (np.linalg.norm(AX_V) + 1e-9)

def cue_scores(tokens):
    ts = {}
    best = {"time": 0.0, "who": 0.0, "place": 0.0, "val": 0.0}
    for w in tokens:
        i = W2I.get(w)
        if i is None:
            continue
        v = V[i]
        t_ = float(v @ AX_T)
        w_ = float(v @ AX_W)
        p_ = float(v @ AX_P)
        v_ = float(v @ AX_V)
        if t_ > best["time"]:
            best["time"] = t_
        if w_ > best["who"]:
            best["who"] = w_
        if p_ > best["place"]:
            best["place"] = p_
        if abs(v_) > abs(best["val"]):
            best["val"] = v_
        ts[w] = (t_, w_, p_)
    return best, ts

def det_of(best, thr):
    cands = [("time", best["time"] - thr["time"]), ("who", best["who"] - thr["who"]),
             ("place", best["place"] - thr["place"])]
    cands.sort(key=lambda x: -x[1])
    return cands[0][0] if cands[0][1] > 0 else "none"

# 校准: 阈值取基率~15%的分位(先全量算分再定阈)
qids = list(Q.keys())
allscores = []
for qa in qids:
    best, _ = cue_scores(toks(Q[qa]["question"]))
    allscores.append(best)
import numpy as _np
thr = {}
for k in ("time", "who", "place"):
    arr = _np.array([b[k] for b in allscores])
    thr[k] = float(_np.quantile(arr, 0.85))
P("校准阈值(85分位): " + str({k: round(v, 3) for k, v in thr.items()}))

import collections
conf = collections.defaultdict(lambda: collections.defaultdict(int))
qids = list(Q.keys())
for k, qa in enumerate(qids):
    q = Q[qa]
    best, _ = cue_scores(toks(q["question"]))
    det = det_of(best, thr)
    conf[gold_type[qa]][det] += 1

P("\n金答案类型 → 线索检测(行=真, 列=测):")
types = ["time", "who", "place", "other"]
hdr = "        " + "".join("%8s" % t for t in types + ["none"])
P(hdr)
lifts = {}
for gt in types:
    row = [conf[gt][d] for d in types + ["none"]]
    tot = sum(row)
    if tot == 0:
        continue
    P("%-8s" % gt + "".join("%8d" % x for x in row))
    hit = conf[gt][gt if gt in types else "none"]
    lifts[gt] = (hit / tot, tot)
base = {d: sum(conf[g][d] for g in types) / max(1, sum(sum(conf[g].values()) for g in types)) for d in types}
P("\n解码提升率(检出率/基率):")
for gt in ("time", "who", "place"):
    if gt in lifts and lifts[gt][1] > 0:
        rate = lifts[gt][0]
        b = base.get(gt, 0)
        P("  %s: 检出=%.1f%% 基率=%.1f%% 提升=%.2fx" % (gt, 100 * rate, 100 * b, rate / max(1e-9, b)))
P("FOUNDRY7_DONE %.0fs" % (time.time() - t0))
LOG.close()
