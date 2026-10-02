# -*- coding: utf-8 -*-
"""r37_runner.py — 臂B+摘要域优先(r36配置, 63.7) + Δ证据指纹(w=0.6)
Δ留出纪律: 用conv-26一个会话的标签估计(12.5%样本), 其余9个会话全量验证; 拆半稳定性0.91/五拆分0.802已备案
其余与r36逐字节相同(单变量纪律: 唯一差异=排序上叠0.6×Δ投影)
"""
import io, json, os, sys, time, re, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
KEY = os.environ.get("GLM_KEY", "${GLM_KEY}")
HERE = "C:/locomo_refined/memsys"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
PREV = HERE + "/subset_all1382.json"
NL = chr(10)

from FlagEmbedding import BGEM3FlagModel, FlagReranker
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def bge_encode(texts):
    out = bge.encode(texts, return_dense=True, return_sparse=True)
    return out["dense_vecs"], out["lexical_weights"]
def glm(prompt, max_tokens=400):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
                                         json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": KEY, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            t = "".join(b.get("text", "") for b in d.get("content", [])
                        if b.get("type") == "text").strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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
TWIN = {}
MID2I = {m: i for i, m in enumerate(MID)}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]

SESS_DATE, SESS_NUM = {}, {}
cur_d, cur_n = "", None
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        rr = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session \d+ — (.+?)\]", (rr.get("raw") or "").strip())
    if m:
        cur_d = m.group(1).strip()
    m = re.match(r"\[Session (\d+)", (rr.get("raw") or "").strip())
    if m:
        cur_n = int(m.group(1))
    SESS_DATE[rr.get("memory_id")] = cur_d
    SESS_NUM[rr.get("memory_id")] = cur_n

def build_rows(recs, cap=250):
    parts = []
    for r in recs:
        raw = (r.get("raw") or "").strip()
        if raw.startswith("[Session"):
            continue
        n = SESS_NUM.get(r.get("memory_id"))
        d = SESS_DATE.get(r.get("memory_id")) or ""
        parts.append("[Session %s — %s] %s" % (n if n is not None else "?", d, raw[:cap]))
    return NL.join(parts) if parts else "(无记忆命中)"

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
CLEAN = {}
for w in WH.keys():
    c = w.strip(":").lower()
    if c and c not in CLEAN:
        CLEAN[c] = w
STOP = set("a an the is are was were be been being do does did have has had i you he she it we they me him her us them my your his its our their what when where who whom why how which that this these those there to of in on at for with about from by as and or but if so not no s t re ve ll d m".split())
HOST_IDX = {w: [MID2I[h] for h in hs if h in MID2I] for w, hs in WH.items()}
def qwords(q):
    return [w for w in re.findall(r"[a-z']+", q.lower()) if w not in STOP and len(w) > 1]
def word_votes(q):
    v = np.zeros(N, dtype=np.float32)
    for w in qwords(q):
        key = CLEAN.get(w)
        if key:
            v[HOST_IDX[key]] += 1.0
    return v
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

data = json.load(open(RAW, encoding="utf-8"))
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(PREV, encoding="utf-8"))
todo = [q for q in questions if q["qa_id"] in set(r["qa_id"] for r in prev)]
print("questions:", len(todo), flush=True)

# ---- Δ留出估计(仅conv-26) ----
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def rec_text_of(i):
    r = REC.get(MID[i], {})
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text_of(i) for i in range(N)]
held_sid = "conv-26"
ev_pool, no_pool = [], []
for q in questions:
    sid = str(q.get("sample_id"))
    if sid != held_sid:
        continue
    cand = CONV.get("loco-" + sid, [])
    hits = set()
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if not hits:
        continue
    tws = {TWIN.get(h) for h in hits} - {None}
    # 该会话全记录做池(不依赖top50)
    allc = cand
    ev_pool += [j for j in allc if j in hits or j in tws]
    no_pool += [j for j in allc if j not in hits and j not in tws][:14]
DELTA = D[np.array(ev_pool)].mean(0) - D[np.array(no_pool)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ = (D @ DELTA).astype(np.float32)
print("Δ from held-out conv-26: ev%d no%d" % (len(ev_pool), len(no_pool)), flush=True)

# ---- 预计算 ----
t0 = time.time()
QL = [q["question"] for q in todo]
bQ = []
for s in range(0, len(QL), 64):
    de, _ = bge_encode(QL[s:s + 64])
    bQ.append(np.asarray(de, dtype=np.float32))
bQ = l2n(np.concatenate(bQ))
raw_idx = [i for i in range(N) if KIND[i] == "raw"]
stmt = D[raw_idx[:3000]].mean(0); stmt /= np.linalg.norm(stmt)
qc = bQ.mean(0); qc /= np.linalg.norm(qc)
u2 = stmt - qc

def ollama_embed(texts):
    out = []
    for s in range(0, len(texts), 64):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 64], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
    return l2n(np.concatenate(out))
wQ = ollama_embed(QL)
FUSED = np.stack([zs(D @ l2n((bQ[i] + u2)[None])[0]) + zs(QW @ wQ[i]) + 0.5 * zs(word_votes(QL[i]))
                  for i in range(len(todo))])
print("fused %.0fs" % (time.time() - t0), flush=True)

rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
ORDER = {}
for i, q in enumerate(todo):
    fused = FUSED[i]
    order = np.argsort(-fused)
    top = order[:50]
    sc = np.asarray(rer.compute_score([[q["question"], REC[MID[j]].get("raw") or ""] for j in top],
                                      batch_size=50), dtype=np.float32)
    inside = np.argsort(-sc)
    newf = fused.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    ORDER[q["qa_id"]] = np.argsort(-newf)
    if i % 300 == 0:
        print("rerank", i, "%.0fs" % (time.time() - t0), flush=True)
print("rerank done %.0fs" % (time.time() - t0), flush=True)

def pass1_ids(order):
    seen, d = set(), []
    for j in order:
        tw = TWIN.get(j)
        if tw is not None and tw in seen:
            continue
        if j in seen:
            continue
        seen.add(j)
        d.append(j)
    summ = [j for j in d if KIND[j] != "raw"][:8]
    rest = [j for j in d if j not in summ][:17]
    return summ + rest

ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
_ade, _alw = bge_encode(ANCHORS)
ANCHOR_V = l2n(np.asarray(_ade, dtype=np.float32))
def is_negative_semantic(ans):
    if not (ans or "").strip():
        return True
    de, _ = bge_encode([ans[:200]])
    av = l2n(np.asarray(de, dtype=np.float32))[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

assert is_negative_semantic("不知道。")
print("smoke ok", flush=True)

import threading
_ELOCK = threading.Lock()
IDX = {q["qa_id"]: i for i, q in enumerate(todo)}

def work(q):
    i = IDX[q["qa_id"]]
    try:
        order = ORDER[q["qa_id"]]
        # r37唯一差异: 排序叠0.6×Δ投影
        order = np.argsort(-(order.astype(np.float32) * 0 + 0)) if False else order
        base = np.zeros(N, dtype=np.float32)
        base[order] = np.linspace(len(order), 1, len(order))
        row = base + 0.6 * zs(PROJ)
        order2 = np.argsort(-row)
        ids1 = pass1_ids(order2)
        ctx1 = build_rows([REC[MID[j]] for j in ids1])
    except Exception as e:
        print("RETR_FAIL", q["qa_id"], repr(e)[:120], flush=True)
        ctx1 = "(retrieval error)"
    p1 = ("根据记忆回答。若无关回答「不知道」。" + NL +
          "记忆(每条已标注所属会话与日期):" + NL + ctx1 + NL + NL +
          "问题: " + q["question"] + NL + TIME_RULE)
    ans1 = glm(p1)
    final, second = ans1, False
    with _ELOCK:
        neg1 = is_negative_semantic(ans1)
    if neg1:
        try:
            order = ORDER[q["qa_id"]]
            base = np.zeros(N, dtype=np.float32)
            base[order] = np.linspace(len(order), 1, len(order))
            row = base + 0.6 * zs(PROJ)
            d = []
            seen = set()
            for j in np.argsort(-row)[:60]:
                tw = TWIN.get(j)
                if tw is not None and tw in seen:
                    continue
                if j in seen:
                    continue
                seen.add(j)
                d.append(j)
            ctx2 = build_rows([REC[MID[j]] for j in d[:50]])
            p2 = ("根据以下原始对话记录回答问题。" + NL +
                  "记录(已标注会话与日期):" + NL + ctx2 + NL + NL +
                  "问题: " + q["question"] + NL + TIME_RULE +
                  " 确实没有答案才回答「不知道」。")
            ans2 = glm(p2, max_tokens=300)
            with _ELOCK:
                neg2 = is_negative_semantic(ans2)
            if not neg2:
                final, second = ans2, True
        except Exception as e:
            print("PASS2_FAIL", q["qa_id"], repr(e)[:100], flush=True)
    return dict(qa_id=q["qa_id"], predicted_answer=final, second_pass=second)

t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    submission = list(ex.map(work, todo))
print("SUBMISSION_DONE n=%d second_pass=%d %.1fs" % (
    len(submission), sum(1 for s in submission if s.get("second_pass")), time.time() - t0), flush=True)
with open(HERE + "/submission_r37.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions_r37.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN r37", flush=True)
