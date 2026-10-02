# -*- coding: utf-8 -*-
"""foundry49.py — 问题结构槽拆分检索: 疑问词槽/主体槽/谓词槽/修饰槽 各自独立寻址
每槽独立查询(原子句域+双空间), topK并集入池。
靶: 全金进池率(基线89.2%@650) + 池尺寸效率。
"""
import io, json, os, re, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry49_results.txt"
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
QW = np.zeros((NR, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

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
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]

# ===== 问题拆槽(规则化, 零LLM) =====
QW_SET = {"what", "when", "where", "who", "which", "how", "why", "whom"}
TIMEW = {"when", "time", "year", "month", "week", "day", "date", "long", "ago", "many", "much", "old"}
CONV_WORDS = {"did", "do", "does", "go", "went", "start", "started", "meet", "met", "make", "made", "have", "has", "give", "gave", "get", "got", "take", "took", "buy", "bought", "plan", "plan"}
def split_slots(qa):
    q = Q[qa]
    toks_list = re.findall(r"[a-z']+", q["question"].lower())
    # 疑问词槽
    qw = [w for w in toks_list if w in QW_SET]
    # 主体槽: 专有名词(大写词, 原文保序) + 人名词表命中
    qtext = q["question"]
    caps = re.findall(r"\b[A-Z][a-z]{2,}\b", qtext)
    caps = [c for c in caps if c.lower() not in QW_SET]
    # 谓词槽: 动作词(动词表)
    verbs = [w for w in toks_list if w in CONV_WORDS]
    # 修饰槽: 时间词
    timem = [w for w in toks_list if w in TIMEW]
    return qw, caps, verbs, timem

# ===== 槽子查询 =====
def subq_scores(qa, slots):
    qi = IDX[qa]
    qw, caps, verbs, timem = slots
    scores = {}
    qv1024 = X[qi]
    qv256 = Q256[qi]
    # 主体槽: 人名宿主票(名字=词面强桥) + 双空间
    if caps:
        capset = set(norm(c) for c in caps)
        for i in range(NR):
            rn = RAWN[i]
            hit = sum(1 for c in capset if c in rn)
            if hit:
                scores.setdefault("subject", {}).setdefault(i, 0.0)
                scores["subject"][i] += 2.0 * hit
    # 疑问词+谓词槽: 原子句域 —— 会话内原子句与(疑问词+谓词)串的匹配
    target = " ".join(qw + verbs)
    if target.strip():
        tt = norm(target)
        for i in range(NR):
            if any(w in RAWN[i] for w in qw + verbs):
                scores.setdefault("pred", {}).setdefault(i, 0.0)
                scores["pred"][i] += 1.0
    # 时间槽: 256时间街区敏感度
    if timem:
        time_dims = list(range(245, 256))
        sens = np.mean(np.abs(QW[:, time_dims]), axis=1)
        top_t = np.argsort(-sens)[:120]
        scores.setdefault("time", {})
        for i in top_t:
            scores["time"][i] = float(sens[i])
    return scores

# ===== 全金在池率: 基线 vs +槽拆分 =====
import random
rng = random.Random(3)
BASE_POOL = 650
res = {"base": {"pool": 0, "size": []}, "slots": {"pool": 0, "size": []}, "base+slots": {"pool": 0, "size": []}}
n = 0
for k_i, qa in enumerate(IDS):
    G = GSETS[k_i]
    if not G:
        continue
    n += 1
    qi = IDX[qa]
    base = set(np.argsort(-FINAL[qi])[:250]) | set(np.argsort(-C0[qi])[:250])
    for i in list(np.argsort(-FINAL[qi]))[:50]:
        base.add(max(0, i - 1))
        base.add(min(NR - 1, i + 1))
    base = set(sorted(base)[:BASE_POOL])
    slots = split_slots(qa)
    slot_scores = subq_scores(qa, slots)
    # 槽并集: 每槽top60 + 主体槽top80
    slot_pool = set()
    for tag, d in slot_scores.items():
        kk = 80 if tag == "subject" else 60
        for i in sorted(d, key=d.get, reverse=True)[:kk]:
            slot_pool.add(i)
    res["base"]["size"].append(len(base))
    res["slots"]["size"].append(len(slot_pool))
    res["base+slots"]["size"].append(len(base | slot_pool))
    if G <= base:
        res["base"]["pool"] += 1
    if G <= slot_pool:
        res["slots"]["pool"] += 1
    if G <= (base | slot_pool):
        res["base+slots"]["pool"] += 1
    if n % 300 == 0:
        P("  eval %d %.0fs" % (n, time.time() - t0))

P("\n===== 槽拆分检索 (n=%d) =====" % n)
for nm in ("base", "slots", "base+slots"):
    r = res[nm]
    avg_size = np.mean(r["size"]) if r["size"] else 0
    P("%-12s 全金在池=%.1f%% 池均=%.0f" % (nm, 100.0 * r["pool"] / max(1, n), avg_size))
P("FOUNDRY49_DONE %.0fs" % (time.time() - t0))
LOG.close()
