# -*- coding: utf-8 -*-
"""formula_foundry.py — 已知x(题目) y(库向量) 求z(答案): 穷举公式+拆半+预注册基线
零API。判据: cos(ẑ, z_gold) + 答案甄别(真答案在20候选中的排位)
纪律: 系数只在train半区选, held-out只测一次; 打不过B2即伪。
"""
import io, json, os, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

# ===== 数据 =====
Q = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    Q[q["qa_id"]] = q
z_ck = np.load(HERE + "/r40bf_ckpt.npz")
FINAL, IDS = z_ck["FINAL"], [str(x) for x in z_ck["IDS"]]
print("questions:", len(IDS), flush=True)

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D = l2n(D)
NR = D.shape[0]

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

t0 = time.time()
qtexts = [Q[i]["question"] for i in IDS]
ztexts = ["; ".join(str(x) for x in (Q[i].get("answer") or [])) for i in IDS]
X = emb(qtexts)
Z = emb(ztexts)
print("embedded %.0fs" % (time.time() - t0), flush=True)

# ===== 库内金证据记录(仅用于取ê和拟合Δ, 不碰z) =====
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
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
MID = []
for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    MID.append(json.loads(l)["mid"])
RAWN = [norm(rec_raw(m)) for m in MID]
E1 = np.zeros((len(IDS), D.shape[1]), dtype=np.float32)   # 每题FINAL第一名记录
EVID = np.zeros((len(IDS), D.shape[1]), dtype=np.float32)  # 每题库内金证据记录(可定位时)
EVLBL = np.zeros((len(IDS), NR), dtype=np.float32)         # Δ拟合用标签(仅train半区读取)
ORDER = []
for qi, qa in enumerate(IDS):
    q = Q[qa]
    order = list(map(int, np.argsort(-FINAL[qi])))
    ORDER.append(order)
    E1[qi] = D[order[0]]
    keys = [norm(em.get("text") or "")[:60] for em in (q.get("evidence_messages") or [])
            if len(norm(em.get("text") or "")) >= 20]
    for r, j in enumerate(order):
        nj = RAWN[j]
        if any(k in nj for k in keys):
            if not EVID[qi].any():
                EVID[qi] = D[j]
            EVLBL[qi, j] = 1.0
    if qi % 300 == 0:
        print("scan", qi, "%.0fs" % (time.time() - t0), flush=True)
print("evidence locatable:", int(sum(1 for qi in range(len(IDS)) if EVID[qi].any())), flush=True)

# ===== 拆半 =====
half = np.array([int(hashlib.md5(qa.encode()).hexdigest(), 16) % 2 for qa in IDS])
tr = half == 0
te = ~tr
print("train=%d held-out=%d" % (tr.sum(), te.sum()), flush=True)

# ===== 轴(全部只在train半区拟合) =====
qc_all = l2n(X.mean(0, keepdims=True))[0]
stmt = l2n(D.mean(0, keepdims=True))[0]
U2 = stmt - qc_all
# Δ: train半区 top50内 证据vs噪声 逐维均值差
DELTA = np.zeros_like(U2)
cnt_e = cnt_n = 0
for qi in range(len(IDS)):
    if not tr[qi]:
        continue
    top = ORDER[qi][:50]
    lbl = EVLBL[qi, top] > 0
    if lbl.sum() == 0 or lbl.sum() == len(top):
        continue
    DELTA += D[top][lbl].mean(0) - D[top][~lbl].mean(0)
    cnt_e += 1
DELTA = DELTA / max(1, cnt_e)
DELTA /= np.linalg.norm(DELTA) + 1e-9
HEMI = l2n(Z[tr].mean(0, keepdims=True))[0] - l2n(X[tr].mean(0, keepdims=True))[0]
HEMI /= np.linalg.norm(HEMI) + 1e-9
print("axes ready (u2/Δtrain/hemi_train) %.0fs" % (time.time() - t0), flush=True)

# ===== 评估器 =====
def eval_z(PZ, mask):
    cs = np.sum(l2n(PZ[mask]) * Z[mask], axis=1)
    # 答案甄别: 每题真z + 19随机他题z, 排位
    rng = np.random.RandomState(42)
    hits = []
    idxs = np.where(mask)[0]
    for qi in idxs:
        pool = [qi] + list(rng.choice(idxs, 19, replace=False))
        sims = PZ[pool] @ Z[pool].T
        s = sims[:, [list(pool).index(qi)]] if False else None
        own = np.arange(len(pool)) == list(pool).index(qi)
        sc = (PZ[pool] * Z[qi]).sum(1)
        hits.append(int((sc >= sc[own]).sum()))
    return float(cs.mean()), float(np.mean(hits))

def win_cent(qi, k):
    return l2n(D[ORDER[qi][:k]].mean(0, keepdims=True))[0]

# ===== 预注册基线 =====
baselines = {
    "B0_z=x": X,
    "B1_z=e1(FINAL第1名)": E1,
    "B2_z=x+u2": X + U2,
    "B3_z=train答案均值": np.tile(l2n(Z[tr].mean(0, keepdims=True)), (len(IDS), 1)),
    "B4_z=金证据(作弊上界)": EVID,
}
print("\n===== 预注册基线(train | held-out) =====", flush=True)
base_ho = {}
for name, PZ in baselines.items():
    Ptr = PZ.copy()
    ctr, hr = eval_z(Ptr, tr)
    cho, ho = eval_z(Ptr, te)
    base_ho[name] = (cho, ho)
    print("%-24s cos tr=%.4f ho=%.4f | 甄别tr=%.2f ho=%.2f" % (name, ctr, cho, hr, ho), flush=True)

# ===== 穷举(系数在train半区选) =====
print("\n===== 穷举 F1: ẑ=a·x+b·ê+c·v =====", flush=True)
AXES = {"none": None, "u2": U2, "delta": DELTA, "hemi": HEMI}
A_RANGE = [0, 0.25, 0.5, 1, 2, 4]
B_RANGE = [0, 0.25, 0.5, 1, 2, 4]
C_RANGE = [0, 0.1, 0.25, 0.5, 1, 2]
best = None
for vname, v in AXES.items():
    for a in A_RANGE:
        for b in B_RANGE:
            for c in C_RANGE:
                if a == 0 and b == 0:
                    continue
                PZ = a * X + b * E1
                if v is not None and c != 0:
                    PZ = PZ + c * v
                PZ = l2n(PZ)
                ctr, hr = eval_z(PZ, tr)
                if best is None or ctr > best[0]:
                    best = (ctr, vname, a, b, c, hr)
ctr, vname, a_, b_, c_, hr = best
print("TRAIN最优: a=%s b=%s c=%s(%s) cos=%.4f" % (a_, b_, c_, vname, ctr), flush=True)
v = AXES[vname]
PZ = l2n(a_ * X + b_ * E1 + (c_ * v if (v is not None and c_ != 0) else 0))
cho, ho = eval_z(PZ, te)
print("HELD-OUT: cos=%.4f 甄别=%.2f  (对照B2 ho=%.4f)" % (cho, ho, base_ho["B2_z=x+u2"][0]), flush=True)

# ===== 穷举 F2: 窗口质心族 =====
print("\n===== 穷举 F2: ẑ=a·x+b·cent(topk) =====", flush=True)
best2 = None
for k in (3, 5, 9, 15):
    C_ = np.stack([win_cent(qi, k) for qi in range(len(IDS))])
    for a in A_RANGE:
        for b in B_RANGE:
            if a == 0 and b == 0:
                continue
            PZ = l2n(a * X + b * C_)
            ctr, hr = eval_z(PZ, tr)
            if best2 is None or ctr > best2[0]:
                best2 = (ctr, k, a, b)
ctr2, k2, a2, b2 = best2
print("TRAIN最优: k=%s a=%s b=%s cos=%.4f" % (k2, a2, b2, ctr2), flush=True)
C_ = np.stack([win_cent(qi, k2) for qi in range(len(IDS))])
cho2, ho2 = eval_z(l2n(a2 * X + b2 * C_), te)
print("HELD-OUT: cos=%.4f 甄别=%.2f" % (cho2, ho2), flush=True)
print("\nFOUNDRY_DONE %.0fs" % (time.time() - t0), flush=True)
