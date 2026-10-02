# -*- coding: utf-8 -*-
"""r29mvp_runner.py - 实验一: 词向量计票层 MVP(零LLM), 挂 71 分底盘。
检索: BGE dense(1024) 0.5 + 词票 0.5 融合; 第二段(语义触发)= v2 双段原样。
支持 --subset 参数选题集; 判分走新判官。"""
import io
import json, os, sys, time, urllib.request, re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
KEY = "${GLM_KEY}"
HERE = "C:/locomo_refined/memsys"
NL = chr(10)

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "subset100c"
TAG = sys.argv[2] if len(sys.argv) > 2 else "mvp_c"
RAW = "C:/locomo_refined/LoCoMo_refined-main/data/raw/locomo_refined.json"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
PREV = HERE + "/" + SUBSET + ".json"

# ---- BGE dense 库 ----
import numpy as np
D = np.load(HERE + "/mem_bge_dense.npz")["dense"]
BGE_MID = []
for l in open(HERE + "/mem_bge_meta.jsonl", encoding="utf-8"):
    if l.strip():
        MID.append(json.loads(l)["mid"])
BGE_MID = MID
MID2REC = {}
SUM_SET = set()
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    MID2REC[r.get("memory_id")] = r
    if r.get("kind") != "raw":
        SUM_SET.add(r.get("memory_id"))
BGE_SUM_IDX = [i for i, m in enumerate(BGE_MID) if m in SUM_SET]
BGE_RAW_IDX = [i for i, m in enumerate(BGE_MID) if m not in SUM_SET]
print("bge dense:", D.shape, "summary:", len(BGE_SUM_IDX), "raw:", len(BGE_RAW_IDX), flush=True)

# ---- 词向量层 ----
V = np.load(HERE + "/word_vecs.npz")["V"]
WORDS = json.load(open(HERE + "/word_vocab.json", encoding="utf-8"))
W2I = {w: i for i, w in enumerate(WORDS)}
HOSTS = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W_MID2REC = set(MID2REC.keys())
print("word layer:", V.shape, flush=True)

SESS_NUM, SESS_DATE = {}, {}
cur_n, cur_d = None, ""
for line in open(HERE + "/mem.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    m = re.match(r"\[Session (\d+) — ([^\]]+)\]", (r.get("raw") or "").strip())
    if m:
        cur_n, cur_d = int(m.group(1)), m.group(2).strip()
    SESS_NUM[r.get("memory_id")] = cur_n
    SESS_DATE[r.get("memory_id")] = cur_d

def ollama_embed(texts):
    out = []
    B = 64
    for i in range(0, len(texts), B):
        body = {"model": "qwen3-embedding:latest", "input": texts[i:i + B], "dimensions": 256}
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed",
                                     json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.extend(json.loads(r.read())["embeddings"])
    return np.asarray(out, dtype=np.float32)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
ANCHOR_V = l2n(ollama_embed(ANCHORS))
def is_negative_semantic(ans):
    if not (ans or "").strip():
        return True
    av = l2n(ollama_embed([ans[:200]]))[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

def glm(prompt, max_tokens=400):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "temperature": 0.0,
            "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
                                         json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": KEY, "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            t = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

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

def word_votes(query, sid, vote_threshold=0.72):
    """词计票: 查询分词 × 词表 余弦>=阈值 → 票投宿主记录(限本会话), 返回 {mid: votes}。"""
    qtoks = set()
    for w in re.findall(r"[A-Za-z]{2,}", query):
        qtoks.add(w.lower())
    cands_idx = [W2I[w] for w in qtoks if w in W2I]
    if not cands_idx:
        return {}
    cv = l2n(V[cands_idx])
    hosts_mid = set()
    recs = []
    for l2_ in open(HERE + "/mem.jsonl", encoding="utf-8"):
        pass
    return {}

def word_votes_v2(query, sid, threshold=0.72, topk_terms=40):
    """词计票: 查询词向量 vs 全词表 → top近邻词 → 宿主记录计票(限本会话)。"""
    sub = [i for i, w in enumerate(WORDS) if len(w) >= 3]
    if not sub:
        return {}
    Vsub = V[sub]
    qv = l2n(ollama_embed([query]))[0]
    sims = Vsub @ qv
    order = np.argsort(-sims)[:topk_terms]
    votes = defaultdict(int)
    best_sim = {}
    for j in order:
        w = WORDS[sub[int(j)]]
        s = float(sims[j])
        if s < threshold:
            continue
        for mid in HOSTS.get(w, []):
            votes[mid] += 1
            best_sim[mid] = max(best_sim.get(mid, 0.0), s)
    return votes

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

def work(q):
    ci = q["conversation_idx"]
    sid = "loco-" + str(data[ci].get("sample_id", ci))
    # 第一段: BGE dense 摘要域 top8
    try:
        qv = l2n(ollama_embed([q["question"]]))[0]
        sub_idx = BGE_SUM_IDX
        sc = D[sub_idx] @ qv
        for j, i in enumerate(sub_idx):
            if REC[BGE_MID[i]].get("session_id") == sid:
                sc[j] += 0.05
        order = np.argsort(-sc)[:8]
        recs1 = [REC[BGE_MID[sub_idx[o]]] for o in order]
        ctx1 = build_rows(recs1)
    except Exception as e:
        print("RETR_FAIL", q["qa_id"], repr(e)[:100], flush=True)
        ctx1 = "(retrieval error)"
    p1 = ("根据记忆回答。若无关回答「不知道」。" + NL +
          "记忆(每条已标注所属会话与日期):" + NL + ctx1 + NL + NL +
          "问题: " + q["question"] + NL + TIME_RULE)
    ans1 = glm(p1)
    final, second = ans1, False
    if is_negative_semantic(ans1):
        try:
            votes = word_votes_v2(q["question"], sid)
            top_mids = sorted(votes, key=lambda m: (-votes[m],))[:25]
            recs2 = [MID2REC[m] for m in top_mids if m in MID2REC]
            if recs2:
                ctx2 = build_rows(recs2)
                p2 = ("根据以下原始对话记录回答问题。" + NL +
                      "记录(已标注会话与日期):" + NL + ctx2 + NL + NL +
                      "问题: " + q["question"] + NL + TIME_RULE +
                      " 确实没有答案才回答「不知道」。")
                ans2 = glm(p2, max_tokens=300)
                if not is_negative_semantic(ans2):
                    final, second = ans2, True
        except Exception as e:
            print("PASS2_FAIL", q["qa_id"], repr(e)[:100], flush=True)
    return dict(qa_id=q["qa_id"], predicted_answer=final, second_pass=second)

data = json.load(open(RAW, encoding="utf-8"))
questions = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(PREV, encoding="utf-8"))
todo = [q for q in questions if q["qa_id"] in set(r["qa_id"] for r in prev)]
print("questions:", len(todo), flush=True)

assert is_negative_semantic("不知道。") and not is_negative_semantic("Melanie 在 2022 年画了日出")
print("self-test ok", flush=True)

t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    submission = list(ex.map(work, todo))
print("SUBMISSION_DONE", len(submission), "second_pass=", sum(1 for s in submission if s.get("second_pass")),
      round(time.time() - t0, 1), "s", flush=True)
tagfile = TAG
with open(HERE + "/submission_%s.jsonl" % tagfile, "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions_%s.jsonl" % tagfile, "w", encoding="utf-8") as f:
    for q in questions:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("SUBMISSION_SAVED", tagfile, flush=True)
