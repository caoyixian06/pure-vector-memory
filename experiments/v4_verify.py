# -*- coding: utf-8 -*-
"""v4_verify.py — 八项离线验证打包(V1-V8, 零GLM)
"""
import io, json, os, sys, re
from datetime import date, timedelta
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
rows_all = [r for r in json.load(open(HERE + "/out_dense_all.json", encoding="utf-8"))]
rows = [r for r in rows_all if r.get("question") and r.get("answer") and (r.get("evidence_messages") or [{}])[0].get("text")]

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50, RER = z1["TOP50"], z1["RER"]
zc = np.load(HERE + "/fusion_cache.npz")
bQ = emb([r["question"] for r in rows])
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc
SBS = (l2n(bQ + u2) @ D.T).astype(np.float32)
DELTA_P = None
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# 证据与匹配
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

def r37_order(i):
    top = TOP50[i]
    inside = np.argsort(-RER[i])
    return top[inside]

def metrics(order_fn, sel=None, tag=""):
    m25 = o5 = cnt = 0
    ev1 = 0
    rsum = 0
    for i in (keys if sel is None else list(sel)):
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        order = order_fn(i)
        pos = {j: p for p, j in enumerate(order)}
        allh = list(hs) + list(tws)
        if not all(h2 in pos for h2 in allh):
            continue
        rk = min(pos[h2] for h2 in allh) + 1
        rsum += rk; cnt += 1
        if order[0] in hs or order[0] in tws:
            ev1 += 1
        if rows[i].get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %-30s 错题进25 %3d  对题进5 %3d  ev@1 %.3f  均名次%.0f (n=%d)" % (
        tag, m25, o5, ev1 / max(1, cnt), rsum / max(1, cnt), cnt), flush=True)

# V1: 组合排序(Δ+占座0.5+首提)
print("== V1 组合排序 ==", flush=True)
metrics(lambda i: r37_order(i), tag="r37对照")
def combo(i, w_d=0.6, w_occ=0.5):
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    top = np.argsort(-row_f)[:50]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in top])
    inside = np.argsort(-sc)
    newf = row_f.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    order = np.argsort(-newf)
    row = np.zeros(N, dtype=np.float32)
    row[order[:25]] = np.linspace(25, 1, 25)
    qtoks = toks(rows[i]["question"])
    COUR = re.compile(r"\b(thank|thanks|congrat|congrats|sorry|glad|awesome|great|nice|cool|wow|amazing|love|appreciate|support|proud)\b", re.I)
    for j in order[:10]:
        t = TEXTS[j]
        if t.rstrip().endswith("?") or COUR.search(t):
            row[j] -= w_occ
    return np.argsort(-row)
metrics(lambda i: combo(i), tag="V1: Δ0.6+占座0.5组合")

# V2: 簇化对多跳(重跑E1核心)
print("== V2 星形簇化(多跳双片纳入) ==", flush=True)
multi = [i for i in keys if len(targets[i] | {MID2I[REC[MID[h2]].get("raw_of")] for h2 in targets[i] if REC.get(MID[h2], {}).get("raw_of") in MID2I}) >= 2]
star2 = flat2 = 0
noise_star = noise_flat = 0
for i in multi:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    allh = hs | tws
    # 星形: 种子top6, 每种子扩邻句+簇
    order = r37_order(i)
    seeds = [j for j in order[:8]]
    used = set()
    star = []
    for s0 in seeds:
        if s0 in used:
            continue
        grp = [s0]
        if s0 + 1 < N and CONVKEY[s0 + 1] == CONVKEY[s0]:
            grp.append(s0 + 1)
        cl = [j for j in order if j not in used and j not in grp and float(D[s0] @ D[j]) >= 0.7][:4]
        grp += cl
        used |= set(grp)
        star += grp
    star = star[:25]
    if len(allh & set(star)) >= 2:
        star2 += 1
    noise_star += sum(1 for j in star if j not in allh)
    # 平铺对照
    flat = list(order[:25])
    if len(allh & set(flat)) >= 2:
        flat2 += 1
    noise_flat += sum(1 for j in flat if j not in allh)
print("  多跳题%d: 星形双片纳入%.0f%% vs 平铺%.0f%% | 噪声条数%.1f vs %.1f" % (
    len(multi), 100 * star2 / max(1, len(multi)), 100 * flat2 / max(1, len(multi)),
    noise_star / max(1, len(multi)), noise_flat / max(1, len(multi))), flush=True)

# V3: pseudofact结果读取
print("== V3 伪fact(pseudofact.log) ==", flush=True)
if os.path.exists(HERE + "/pseudofact.log"):
    for l in open(HERE + "/pseudofact.log", encoding="utf-8", errors="ignore"):
        if "覆盖" in l or "top5" in l or "top10" in l or "原子句" in l:
            print("  " + l.strip()[:120], flush=True)
else:
    print("  log缺失", flush=True)

# V4: 词覆盖(真独有词)端到端代理
print("== V4 真独有词覆盖重排(上界测试) ==", flush=True)
def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
zw = np.load(HERE + "/word_vecs.npz")
WV = l2(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c = w.strip(":").lower()
    if c and c not in W2I:
        W2I[c] = idx
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None
def toks_all(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2]
STOP2 = set("the a an is are was were be been being to of in on at for with about from by as and or but if so not no that this these those it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())
m25c = o5c = cntc = 0
for i in keys:
    r = rows[i]
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    dws = set(w for w in toks_all(str(r["answer"][0])) if w not in toks_all(r["question"]) and w not in STOP2)
    order = r37_order(i)
    row = np.linspace(50, 1, 50)
    if dws:
        cov = np.array([len(dws & set(toks_all(TEXTS[j]))) for j in order], dtype=np.float32)
        row = row + 8.0 * zs(cov)
    order2 = order[np.argsort(-row)]
    allh = list(hs) + list(tws)
    if not all(h2 in set(order2.tolist()) for h2 in allh):
        continue
    pos = {j: p for p, j in enumerate(order2)}
    rk = min(pos[h2] for h2 in allh) + 1
    cntc += 1
    if r.get("llm_score") == 1:
        o5c += rk <= 5
    else:
        m25c += rk <= 25
print("  真独有词覆盖重排: 错题进25 %d 对题进5 %d (上界, 预测强)" % (m25c, o5c), flush=True)

# V5: 门控Δ
print("== V5 门控Δ(枚举题降权) ==", flush=True)
DELTA = None
evA, noA = [], []
for i in keys[:len(keys) // 2]:
    hs = targets[i]
    tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
    top = TOP50[i]
    evA += [j for j in top if j in hs or j in tws]
    noA += [j for j in top if j not in hs and j not in tws][:8]
DELTA = D[np.array(evA)].mean(0) - D[np.array(noA)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ_D = (D @ DELTA).astype(np.float32)
def is_enum(r):
    return ("," in str(r["answer"][0])) or (" and " in str(r["answer"][0]).lower())
for gate_w, tag in ((0.6, "Δ全量0.6(无门控)"), (0.2, "Δ枚举门控0.2")):
    m25 = o5 = 0
    for i in keys[len(keys) // 2:]:
        r = rows[i]
        order = r37_order(i)
        w = gate_w if is_enum(r) else 0.6
        row = np.linspace(50, 1, 50) + w * zs(PROJ_D[order])
        order2 = order[np.argsort(-row)]
        hs = targets[i]
        tws = {MID2I[REC[MID[h2]].get("raw_of")] for h2 in hs if REC.get(MID[h2], {}).get("raw_of") in MID2I}
        allh = list(hs) + list(tws)
        if not all(h2 in set(order2.tolist()) for h2 in allh):
            continue
        pos = {j: p for p, j in enumerate(order2)}
        rk = min(pos[h2] for h2 in allh) + 1
        if r.get("llm_score") == 1:
            o5 += rk <= 5
        else:
            m25 += rk <= 25
    print("  %s: 错题进25 %d 对题进5 %d" % (tag, m25, o5), flush=True)

# V6: 路由扩池(batch2中断部分)
print("== V6 低自信路由扩池 ==", flush=True)
def np_entropy(x):
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-6, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())
scores = []
for i in keys:
    sc = RER[i]
    inside = np.argsort(-sc)
    s = sc[inside]
    scores.append(s[0] - s[1] - np_entropy(s) * 0.1)
scores = np.array(scores)
q75 = np.percentile(scores, 75)
low = [i for i in keys if scores[list(keys).index(i)] > q75]
lowset = set(low)
def rout(i):
    order = r37_order(i)
    if i not in lowset:
        return order
    row_f = zs(SBS[i]) + zs(zc["SQ"][i]) + 0.5 * zs(zc["VEX"][i])
    ext = np.argsort(-row_f)[:100]
    rer_map = {TOP50[i][k]: RER[i][k] for k in range(len(TOP50[i]))}
    sc = np.array([rer_map.get(j, -7.0) for j in ext])
    inside = np.argsort(-sc)
    row = np.zeros(N, dtype=np.float32)
    row[ext[inside[:50]]] = np.linspace(50, 1, 50)
    row[ext[inside[50:]]] = np.linspace(0.5, 0.1, max(1, len(inside) - 50))
    return np.argsort(-row)
metrics(routed := rout, tag="低自信扩池100(低自信子集)", sel=lowset)
metrics(lambda i: r37_order(i), tag="  对照(低自信子集r37)", sel=lowset)

# V8: 日历计算器离线精度
print("== V8 日历计算器(时间题错题离线模拟) ==", flush=True)
MONTHS = {m.lower(): i2 + 1 for i2, m in enumerate(
    "January February March April May June July August September October November December".split())}
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
qfile = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
WD = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
def parse_d(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        try:
            return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
        except Exception:
            return None
    return None
fixed_ok = 0
time_miss = 0
misses = [r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8")) if r.get("llm_score") != 1]
for r in misses:
    q = qfile.get(r["qa_id"])
    if not q or str(q.get("category")) != "3" and "when" not in r["question"].lower():
        continue
    evd = [parse_d(SESS_DATE.get(MID2I.get("loco-" + str(q.get("sample_id")) + "_m%02d" % (e.get("session_index", 0)) ) or "")) for e in (q.get("evidence_messages") or [])]
    evd = [d for d in evd if d]
    if not evd:
        continue
    time_miss += 1
    # 日历规则: 证据句含相对词 → 换算成绝对日期, 检查是否落在金答案字符串的日期附近
    for e in q.get("evidence_messages") or []:
        t = e.get("text") or ""
        sd = parse_d("[Session 1 — %s]" % SESS_DATE.get("loco-%s_m%02d" % (q.get("sample_id"), e.get("session_index", 1))) or "")
        if not sd:
            continue
        for m in re.finditer(r"\b(yesterday|tomorrow|last week|next week)\b", t, re.I):
            k = m.group(1).lower()
            nd = sd + timedelta(days={"yesterday": -1, "tomorrow": 1, "last week": -7, "next week": 7}[k])
            golds = " ".join(str(x) for x in r["answer"])
            mm = re.search(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)", golds)
            if mm and nd.month == MONTHS[mm.group(2).lower()] and nd.day == int(mm.group(1)):
                fixed_ok += 1
                break
print("  时间题错题中含相对时间词且证据session可定日期: %d道, 日历规则可直接命中金答案日: %d道" % (
    time_miss, fixed_ok), flush=True)
print("V4VERIFY_DONE", flush=True)
