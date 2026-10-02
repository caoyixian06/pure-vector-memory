# -*- coding: utf-8 -*-
"""foundry3.py — 双空间(1024+256)错题公式穷举: 问题数值⊗库数值→ẑ→解码→对齐金证据
解码路A(1024): argmax记录, 其原文=算出的文本; 解码路B(256): 词表最近邻=算出的词
中间尺子(零人工零LLM): ①身份(解码记录==金证据记录) ②文本Jaccard ③cos
纪律: 错题拆dev/test, 系数dev选, test一次。基线=现有FINAL第1名。
"""
import io, json, os, re, sys, time, hashlib, urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
OUT = HERE + "/foundry3_results.txt"
LOG = io.open(OUT, "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2)

t0 = time.time()
# ===== 错题表 =====
ERR, SCORE = [], {}
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m:
        s = m.group(2)
        sc = int(s) if len(s) == 1 and s in "01" else -1
        SCORE[m.group(1)] = sc
ERR = [k for k, v in SCORE.items() if v == 0]
P("错题=%d" % len(ERR))

Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q

# ===== 库(1024+256+词表+原文) =====
D = l2n(np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32))
MID = []
for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    MID.append(json.loads(l)["mid"])
MID2I = {m: i for i, m in enumerate(MID)}
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
QW = np.zeros((len(MID), 256), dtype=np.float32)
for m, i in MID2I.items():
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WORDS = list(json.load(open(HERE + "/word_hosts.json", encoding="utf-8")).keys())
P("库=%d 256记录=%d 词表=%d %.0fs" % (len(MID), QW.shape[0], len(WORDS), time.time() - t0))

# ===== 金证据记录定位(字符串) =====
def gold_rec(qa):
    q = Q[qa]
    keys = [norm(em.get("text") or "")[:60] for em in (q.get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for j, nj in enumerate(RAWN):
        if any(k in nj for k in keys):
            return j
    return None

# ===== 题目双空间向量 =====
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def bemb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def oemb(texts):
    out = []
    for s in range(0, len(texts), 32):
        chunk = texts[s:s + 32]
        for att in range(4):
            try:
                body = json.dumps({"model": "qwen3-embedding:latest", "input": chunk,
                                   "dimensions": 256}).encode()
                req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    out.append(np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32))
                break
            except Exception as e:
                print("oemb retry", s, repr(e)[:80], flush=True)
                time.sleep(3 * (att + 1))
        else:
            raise RuntimeError("oemb failed at %d" % s)
    return l2n(np.concatenate(out))

X1024 = bemb([Q[qa]["question"] for qa in ERR])
X256 = oemb([Q[qa]["question"] for qa in ERR])
P("题目双空间嵌入完成 %.0fs" % (time.time() - t0))

# ===== 每错题: 金证据记录idx / 现系统e1 / 轴 =====
GOLD = [gold_rec(qa) for qa in ERR]
keep = [i for i, g in enumerate(GOLD) if g is not None]
ERRK = [ERR[i] for i in keep]
GOLD = [GOLD[i] for i in keep]
GOLD = np.array(GOLD, dtype=np.int64)
X1024, X256 = X1024[keep], X256[keep]
P("金证据可定位=%d" % len(GOLD))

E1_1024 = D[[int(np.argmax(D @ X1024[i])) for i in range(len(ERRK))]]
E1_256 = QW[[int(np.argmax(QW @ X256[i])) for i in range(len(ERRK))]]
# 现系统基线: FINAL第1名是否==金证据
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
IDS = [str(x) for x in z_ck["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
FINAL = z_ck["FINAL"]
base_hit = 0
for i, qa in enumerate(ERRK):
    order0 = int(np.argmax(FINAL[IDX[qa]]))
    if order0 == GOLD[i]:
        base_hit += 1
P("基线(现系统FINAL第1名==金证据): %d/%d = %.1f%%" % (base_hit, len(GOLD), 100.0 * base_hit / len(GOLD)))

# 轴(256与1024各自; 用全库均值差, 不碰答案)
U2_1024 = l2n(D.mean(0, keepdims=True))[0] - l2n(X1024.mean(0, keepdims=True))[0]
U2_256 = l2n(QW.mean(0, keepdims=True))[0] - l2n(X256.mean(0, keepdims=True))[0]

# ===== dev/test 拆半(错题内) =====
half = np.array([int(hashlib.md5(qa.encode()).hexdigest(), 16) % 2 for qa in ERRK])
dev = half == 0
tes = ~dev
P("dev=%d test=%d" % (dev.sum(), tes.sum()))

def jacc(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))

def evaluate(PZ, MAT, GOLDa, mask, topk=(1, 5), wordmode=False):
    """解码: MAT内argmax; 尺: 身份hit@1/@5 + 与金证据原文的Jaccard"""
    sims = PZ[mask] @ MAT.T
    hits1 = hits5 = 0
    jac = []
    for r in range(sims.shape[0]):
        top = np.argsort(-sims[r])[:5]
        gi = GOLDa[np.where(mask)[0]][r]
        if top[0] == gi:
            hits1 += 1
        if gi in top:
            hits5 += 1
        jac.append(jacc(RAW[top[0]], RAW[gi]))
    return 100.0 * hits1 / sims.shape[0], 100.0 * hits5 / sims.shape[0], float(np.mean(jac))

def word_decode(PZ256, mask, m=10):
    """数字→词: ẑ256与词表最近邻m词, 与金证据token的覆盖率"""
    sims = PZ256[mask] @ WV.T
    prec = rec = 0.0
    for r in range(sims.shape[0]):
        gi = GOLD[np.where(mask)[0]][r]
        gt = toks(RAW[gi])
        topw = {WORDS[w].strip(":").lower() for w in np.argsort(-sims[r])[:m]}
        topw = {w for w in topw if len(w) > 3}
        if gt:
            rec += len(gt & topw) / len(gt)
        prec += len(gt & topw) / max(1, len(topw))
    n = sims.shape[0]
    return 100.0 * prec / n, 100.0 * rec / n

def run_space(name, X, E1, MAT, U2, Uax=None):
    P("\n===== %s =====" % name)
    P("基线 e1: hit@1=%.1f%% (现系统已在上方)" % evaluate(E1, MAT, GOLD, np.ones(len(GOLD), bool))[0])
    A_R = [0, 0.25, 0.5, 1, 2]
    B_R = [0, 0.25, 0.5, 1, 2]
    C_R = [0, -0.5, -0.25, 0.25, 0.5, 1]
    axes = {"none": None, "u2": U2}
    if Uax is not None:
        axes["ax"] = Uax
    best = None
    for an, ax in axes.items():
        for a in A_R:
            for b in B_R:
                if a == 0 and b == 0:
                    continue
                for c in ([0] if ax is None else C_R):
                    PZ = l2n(a * X + b * E1 + (c * ax if ax is not None else 0))
                    h1, _, jm = evaluate(PZ, MAT, GOLD, dev)
                    sc = h1 + 100 * jm
                    if best is None or sc > best[0]:
                        best = (sc, an, a, b, c)
    P("dev最优: %s" % (best,))
    _, an, a, b, c = best
    ax = axes[an]
    PZ = l2n(a * X + b * E1 + (c * ax if ax is not None else 0))
    h1, h5, jm = evaluate(PZ, MAT, GOLD, tes)
    P("TEST: 身份hit@1=%.1f%% hit@5=%.1f%% 解码文本Jaccard=%.3f" % (h1, h5, jm))
    if name.startswith("[256"):
        p_, r_ = word_decode(PZ, tes)
        P("词解码(10词): 对金证据 覆盖率=%.1f%% 命中率=%.1f%%" % (r_, p_))
    # 抽样3例(人可读)
    tidx = np.where(tes)[0][:3]
    for i in tidx:
        sims = PZ[i] @ MAT.T
        j = int(np.argmax(sims))
        P("  例[%s] 解码记录: %s" % (ERRK[i][:16], RAW[j][:100]))
        P("      金证据:    %s" % RAW[GOLD[i]][:100])
    return h1

run_space("[1024空间]", X1024, E1_1024, D, U2_1024)
run_space("[256空间]", X256, E1_256, QW, U2_256, Uax=None)
P("\nFOUNDRY3_DONE %.0fs" % (time.time() - t0))
LOG.close()
