# -*- coding: utf-8 -*-
"""r36_runner.py — 回滚+单变量: 臂B原配置(r33 ARM=B) + 仅pass1摘要域优先
臂B = 融合(BGE-u2+qwen+0.5词票) → 精排重排top50 → top25上下文 → 两段+语义触发
唯一改动: pass1上下文 = 前8条限summary域(按最终序) + 后17条按最终序(去孪生)
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

def dedup(order):
    seen, outo = set(), []
    for j in order:
        tw = TWIN.get(j)
        if tw is not None and tw in seen:
            continue
        if j in seen:
            continue
        seen.add(j)
        outo.append(j)
    return outo

# 摘要域优先pass1: 前8条=最终序里的summary记录, 后17条=其余按最终序
def pass1_ids(order):
    d = dedup(order)
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

assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出,湖上日出作品很著名")
print("smoke ok", flush=True)

import threading
_ELOCK = threading.Lock()
IDX = {q["qa_id"]: i for i, q in enumerate(todo)}

def work(q):
    i = IDX[q["qa_id"]]
    try:
        order = ORDER[q["qa_id"]]
        ids1 = pass1_ids(order)
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
            d = dedup(order)
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
with open(HERE + "/submission_r36.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions_r36.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN r36", flush=True)
