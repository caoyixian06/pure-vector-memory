# -*- coding: utf-8 -*-
"""lme_probe7.py — 预言6抽样验证: LME原子句粒度下话题寒暄vs内容句的方向
每题金会话抽15条记录→切原子句→抽40句→BGE嵌入→分类对比cos
判定: 原子句粒度下话题寒暄-内容句差距是否拉开(记录粒度=0.488vs0.482打平; LoCoMo记录粒度0.492vs0.443)"""
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
COURT = re.compile(r"\b(hi|hey|hello|thanks|thank you|great|awesome|cool|nice|sure|okay|ok|wow|sounds good|good to know)\b", re.I)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
# 直接读mem.jsonl(刚写完的214103条)
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
P("mem.jsonl loaded %d %.0fs" % (len(RAWS), time.time() - t0))
SID2ROWS = {}
for i, s in enumerate(SIDS):
    SID2ROWS.setdefault(s, []).append(i)

rng = np.random.RandomState(0)
# 每题: 金会话15条记录→原子句→抽40句
samples = []   # (qi, 句文本)
for qi, q in enumerate(d):
    gs = set()
    for s in (q.get("answer_session_ids") or []):
        if s in sid2h:
            gs |= set(SID2ROWS.get("lme-s" + sid2h[s][:12], []))
    gs = np.array(sorted(gs))
    if len(gs) < 5:
        continue
    recs = rng.choice(gs, size=min(15, len(gs)), replace=False)
    atoms = []
    for i in recs:
        for piece in re.split(r"[.!?]+", RAWS[i]):
            p = piece.strip()
            nw = len(p.split())
            if 4 <= nw <= 45:
                atoms.append(p)
    if not atoms:
        continue
    pick = rng.choice(len(atoms), size=min(40, len(atoms)), replace=False)
    for j in pick:
        samples.append((qi, atoms[j]))
P("样本=%d句 %.0fs" % (len(samples), time.time() - t0))

from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
texts = [s[1] for s in samples]
vecs = []
for st in range(0, len(texts), 64):
    vecs.append(np.asarray(bge.encode(texts[st:st + 64])["dense_vecs"], dtype=np.float32))
AV = l2n(np.concatenate(vecs))
P("embedded %.0fs" % (time.time() - t0))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))

res = {"topic-chat": [], "func-chat": [], "content": []}
for (qi, txt), v in zip(samples, AV):
    qstems = set(stem(w) for w in re.findall(r"[a-z']+", d[qi]["question"].lower()) if w not in QSTOP and len(w) > 2)
    nw = len(txt.split())
    isc = bool(COURT.search(txt)) or nw <= 6
    if isc:
        rsts = set(stem(w) for w in re.findall(r"[a-z']+", txt.lower()) if len(w) > 2)
        k = "topic-chat" if (rsts & qstems) else "func-chat"
    else:
        k = "content"
    res[k].append(float(v @ X[qi]))

P("\n===== 预言6验证: LME原子句粒度(金会话内, n=%s) =====" % {k: len(v) for k, v in res.items()})
for k in ("topic-chat", "content", "func-chat"):
    if res[k]:
        P("  %-11s cos=%.3f" % (k, np.mean(res[k])))
P("对照: LME记录粒度 topic 0.488 vs content 0.482 (差0.006) | LoCoMo记录粒度 0.492 vs 0.443 (差0.049)")
P("预言: 原子句粒度下 topic-content 差距应显著拉开(→0.04+)")
P("done %.0fs" % (time.time() - t0))
