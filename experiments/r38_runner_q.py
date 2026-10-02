# -*- coding: utf-8 -*-
"""r38_runner.py — V3第一件: 原文原子句域(伪fact正名: 原文衍生事实域)接入检索
架构: r37底盘(融合+精排+摘要域优先+Δ) + 原文原子句域第三路
原子句域: 从raw原文切原子事实句(规则, 零LLM), 独立嵌入, 检索时与主通道并行
测试: 100题随机(独立seed), Qwen3-14B官方判官
"""
import io, json, os, sys, re, time, random, urllib.request
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
QWEN_KEY = "${QWEN_KEY}"
QWEN_API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
HERE = "C:/locomo_refined/memsys"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
NL = chr(10)

from FlagEmbedding import BGEM3FlagModel, FlagReranker
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def bge_encode(texts):
    out = bge.encode(texts, return_dense=True, return_sparse=True)
    return out["dense_vecs"], out["lexical_weights"]
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))

# ===== 原文原子句域 =====
VERB = re.compile(r"\b(is|was|are|were|have|has|had|do|does|did|will|would|can|could|went|go|make|made|take|took|get|got|play|played|read|reads|started|finished|work|works|worked|live|lives|lived|like|likes|loves|adopted|won|signed|joined|planning|plan|bought|sold|graduated|studying|study|moving|moved|meeting|met)\b", re.I)
atoms = []       # (text, session_id, src_mid)
ATOM_IDX = {}    # memory_id -> atom rows
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") != "raw":
        continue
    s = (r.get("raw") or "").strip()
    if s.startswith("[Session"):
        continue
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if not VERB.search(p):
            continue
        mid = r.get("memory_id")
        ATOM_IDX[mid] = ATOM_IDX.get(mid, len(atoms))
        atoms.append((p, r.get("session_id") or "", mid))
print("原子句:", len(atoms), flush=True)
AVEC = emb([a[0] for a in atoms])
ASESS = [a[1] for a in atoms]
print("atom embedded %.0fs" % 0, flush=True)

# 原子句→宿主记录(mid): 用于去重和上下文映射
ATOM_HOST = [a[2] for a in atoms]
# 主库
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
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)

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
        n2 = SESS_NUM.get(r.get("memory_id"))
        d2 = SESS_DATE.get(r.get("memory_id")) or ""
        parts.append("[Session %s — %s] %s" % (n2 if n2 is not None else "?", d2, raw[:cap]))
    return NL.join(parts) if parts else "(无记忆命中)"

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
CLEAN = {}
for w in WH.keys():
    c0 = w.strip(":").lower()
    if c0 and c0 not in CLEAN:
        CLEAN[c0] = w
STOP = set("a an the is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())
HOST_IDX = {w: [MID2I[h2] for h2 in hs if h2 in MID2I] for w, hs in WH.items()}
def word_votes(q):
    v = np.zeros(N, dtype=np.float32)
    for w in re.findall(r"[a-z']+", q.lower()):
        if w in STOP or len(w) <= 1:
            continue
        key = CLEAN.get(w)
        if key:
            v[HOST_IDX[key]] += 1.0
    return v
def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def glm(prompt, max_tokens=400):
    body = {"model": "qwen3-14b", "max_tokens": max_tokens, "temperature": 0.0,
            "enable_thinking": False, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + QWEN_KEY})
            with urllib.request.urlopen(req, timeout=180) as r2:
                d = json.loads(r2.read())
            t = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", [])).strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

# ---- 测试集: 随机100(seed独立) ----
ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(HERE + "/subset_all1382.json", encoding="utf-8"))
test = prev
test_ids = {t["qa_id"] for t in test}
qmap = {q["qa_id"]: q for q in ALLQS}
todo = [qmap[qid] for qid in test_ids if qid in qmap]
print("test questions:", len(todo), flush=True)

# ---- 检索预计算 ----
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

# 原子句域检索: 每题在原子句域召回top6, 映射到宿主记录
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
FINAL = []
for i, q in enumerate(todo):
    fused = FUSED[i]
    # 原子句域召回: 该会话的原子句
    sid = "loco-" + str(q.get("sample_id"))
    idxs = [k for k, a in enumerate(atoms) if a[1] == sid]
    atom_boost = {}
    if idxs:
        sims = AVEC[idxs] @ (bQ[i])
        for oi in np.argsort(-sims)[:6]:
            host_mid = atoms[idxs[oi]][2]
            j = MID2I.get(host_mid)
            if j is not None:
                atom_boost[j] = max(atom_boost.get(j, 0), float(sims[oi]))
    top = np.argsort(-fused)[:50]
    sc = np.asarray(rer.compute_score([[q["question"], REC[MID[j]].get("raw") or ""] for j in top],
                                      batch_size=50), dtype=np.float32)
    inside = np.argsort(-sc)
    newf = fused.copy()
    newf[top[inside]] = np.linspace(50, 1, 50)
    for j, ab in atom_boost.items():
        if j < N:
            newf[j] = newf[j] + 0.5 * ab * 3.0  # 原子域命中加成
    FINAL.append(newf)
    if i % 20 == 0:
        print("rerank+atom", i, "%.0fs" % (time.time() - t0), flush=True)
print("done %.0fs" % (time.time() - t0), flush=True)

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

assert len(todo[0]["question"]) > 5
print("smoke ok", flush=True)

import threading
_ELOCK = threading.Lock()
IDX = {q["qa_id"]: i for i, q in enumerate(todo)}

def work(q):
    i = IDX[q["qa_id"]]
    order = np.argsort(-FINAL[i])
    ids1 = pass1_ids(order)
    try:
        ctx1 = build_rows([REC[MID[j]] for j in ids1])
    except Exception as e:
        print("RETR_FAIL", q["qa_id"], repr(e)[:100], flush=True)
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
            d2 = []
            seen2 = set()
            for j in order[:60]:
                tw = TWIN.get(j)
                if tw is not None and tw in seen2:
                    continue
                if j in seen2:
                    continue
                seen2.add(j)
                d2.append(j)
            ctx2 = build_rows([REC[MID[j]] for j in d2[:50]])
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
with open(HERE + "/submission_r38q.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions_r38q.jsonl", "w", encoding="utf-8") as f:
    for q in ALLQS:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN r38", flush=True)

# ---- Qwen官方判官 ----
PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""
qmap_full = {q["qa_id"]: q for q in ALLQS}
def judge(r):
    q = qmap_full.get(r["qa_id"], {})
    gold = "; ".join(str(x) for x in (r.get("answer") or []))
    body = {"model": "qwen3-14b",
            "messages": [{"role": "user", "content": PROMPT.format(
                q=q.get("question", ""), g=gold, p=str(r.get("predicted_answer") or ""))}],
            "temperature": 0.0, "max_tokens": 2048, "enable_thinking": False, "stream": False}
    for att in range(3):
        try:
            req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + QWEN_KEY})
            with urllib.request.urlopen(req, timeout=120) as r2:
                d = json.loads(r2.read())
            txt = "".join(ch.get("message", {}).get("content", "") or "" for ch in d.get("choices", []))
            m = re.search(r'"score"\s*:\s*([01])', txt)
            if m:
                return int(m.group(1))
        except Exception:
            time.sleep(2)
    return -1

scores = [judge(r) for r in submission]
ok = sum(1 for s in scores if s == 1)
valid = sum(1 for s in scores if s in (0, 1))
print("R38Q_SCORE", ok, "/", valid, "=", round(100 * ok / max(1, valid), 1), "%", flush=True)
with open(HERE + "/r38q_score.txt", "w", encoding="utf-8") as f:
    f.write("R38 %d/%d = %.1f%%\n" % (ok, valid, 100 * ok / max(1, valid)))
print("R38Q_DONE", flush=True)
