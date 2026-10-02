# -*- coding: utf-8 -*-
"""r35_runner.py — 架构V2首版随机100题测试
= r34底盘(融合+精排+hyde/槽位+邻turn+凑一套+日历)
+ Δ证据指纹融合(留出估计: Δ只用非测试题的金标签, 无泄漏)
+ 问句降权(小权重, topsep实证问句是噪声负预测)
测试集: subset_all1382中随机100题(seed=20260917)
"""
import io, json, os, sys, time, re, random, urllib.request, hashlib
from datetime import date, timedelta
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

# ---- 库 ----
D = np.load(HERE + "/mem_bge_dense.npz")["dense"].astype(np.float32)
D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
MID, KIND = [], []
for l in open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r["kind"] or "summary")
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
QW = np.zeros((N, 256), dtype=np.float32)
for i, m in enumerate(MID):
    v = REC.get(m, {}).get("vector")
    if v:
        QW[i] = v
QW = l2n(QW)
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
TWIN = {}
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

def rec_text_of(i):
    r = REC.get(MID[i], {})
    for f in ("text", "raw", "content", "summary", "raw_text", "original"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
TEXTS = [rec_text_of(i) for i in range(N)]

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

# ---- 词票/槽位/hyde/日历(同r34) ----
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

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
def norm_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(1).lower() in MONTHS:
        return "%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()])
    return ""
def q_date_keys(q):
    keys = set()
    for m in re.finditer(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})", q, re.I):
        keys.add("%s-%02d-%02d" % (m.group(3), MONTHS[m.group(1).lower()], int(m.group(2))))
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", q, re.I):
        keys.add("%s-%02d" % (m.group(2), MONTHS[m.group(1).lower()]))
    return keys
ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
ALL_NAMES = set()
for q in ALLQS:
    for k in ("speaker_a", "speaker_b"):
        if q.get(k):
            ALL_NAMES.add(q[k])
def q_names(q):
    return {n for n in ALL_NAMES if re.search(r"\b%s\b" % re.escape(n), q)}
def slot_score(qkeys, qns, rec):
    rd = norm_date(SESS_DATE.get(rec.get("memory_id")) or "")
    ds = 0.0
    if rd:
        for k in qkeys:
            if k == rd:
                ds = 1.0
                break
            if len(k) == 7 and rd.startswith(k):
                ds = max(ds, 0.5)
    ns = 0.0
    raw = rec.get("raw") or ""
    if qns:
        hit = sum(1 for n in qns if re.search(r"\b%s\b" % re.escape(n), raw))
        ns = min(1.0, hit / max(1, len(qns)))
    return 0.6 * ds + 0.4 * ns

HYDE_CACHE = HERE + "/hyde_cache.jsonl"
_hcache = {}
if os.path.exists(HYDE_CACHE):
    for l in open(HYDE_CACHE, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            _hcache[d["qh"]] = d
def cached_hyp(query):
    qh = hashlib.md5(query.encode()).hexdigest()[:16]
    if qh in _hcache:
        return _hcache[qh].get("hyp") or ""
    return ""

WD = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
def parse_sess_date(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})", s or "")
    if m and m.group(2).lower() in MONTHS:
        return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None
def calendar_lines(sel_recs):
    out = []
    for r in sel_recs[:14]:
        sd = parse_sess_date(SESS_DATE.get(r.get("memory_id")) or "")
        if not sd:
            continue
        raw = (r.get("raw") or "")
        conv = []
        if re.search(r"\byesterday\b", raw, re.I):
            conv.append(("yesterday", (sd - timedelta(days=1)).strftime("%Y年%m月%d日")))
        if re.search(r"\btomorrow\b", raw, re.I):
            conv.append(("tomorrow", (sd + timedelta(days=1)).strftime("%Y年%m月%d日")))
        for m in re.finditer(r"\b(next|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", raw, re.I):
            tgt = WD[m.group(2).lower()]
            delta = ((tgt - sd.weekday()) % 7 or 7) if m.group(1).lower() == "next" else (-((sd.weekday() - tgt) % 7 or 7))
            conv.append((m.group(0).lower(), (sd + timedelta(days=delta)).strftime("%Y年%m月%d日")))
        for m in re.finditer(r"\b(next|last)\s+(week|month|year)\b", raw, re.I):
            k = m.group(2).lower()
            try:
                if k == "week":
                    nd = sd + timedelta(days=7 if m.group(1).lower() == "next" else -7)
                elif k == "month":
                    mm = sd.month + (1 if m.group(1).lower() == "next" else -1)
                    yy = sd.year + (1 if mm == 13 else (-1 if mm == 0 else 0))
                    nd = date(yy, (mm - 1) % 12 + 1, min(sd.day, 28))
                else:
                    nd = date(sd.year + (1 if m.group(1).lower() == "next" else -1), sd.month, min(sd.day, 28))
            except Exception:
                continue
            conv.append((m.group(0).lower(), nd.strftime("%Y年%m月%d日")))
        for w, absd in conv[:2]:
            out.append("[日历换算] Session%s(%s) 的 %s = %s" % (
                SESS_NUM.get(r.get("memory_id"), "?"), sd.strftime("%Y-%m-%d"), w, absd))
        if len(out) >= 6:
            break
    return out

ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
_ade, _alw = bge_encode(ANCHORS)
ANCHOR_V = l2n(np.asarray(_ade, dtype=np.float32))
def is_negative_semantic(ans):
    if not (ans or "").strip():
        return True
    de, _ = bge_encode([ans[:200]])
    av = l2n(np.asarray(de, dtype=np.float32))[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

# ---- 测试集: 随机100 ----
prev = json.load(open(HERE + "/subset_all1382.json", encoding="utf-8"))
rng = random.Random(20260917)
test = rng.sample(prev, 100)
test_ids = {t["qa_id"] for t in test}
qmap = {q["qa_id"]: q for q in ALLQS}
todo = [qmap[qid] for qid in test_ids if qid in qmap]
print("test questions:", len(todo), flush=True)
data = json.load(open(RAW, encoding="utf-8"))

# ---- Δ留出估计(非测试题) ----
CONV = {}
for i, m in enumerate(MID):
    CONV.setdefault(re.sub(r"_(rbak\d+k|m\d+)$", "", m), []).append(i)
def normsub(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
z1 = np.load(HERE + "/rerank_stage1.npz")
TOP50C = z1["TOP50"]
ev_pool, no_pool = [], []
for qi, q in enumerate(ALLQS):
    if q["qa_id"] in test_ids:
        continue
    sid = q.get("sample_id")
    cand = CONV.get("loco-" + str(sid), [])
    hits = set()
    for e in q.get("evidence_messages") or []:
        t = normsub(e.get("text") or "")
        if len(t) < 15:
            continue
        for j in cand:
            if KIND[j] == "raw" and t[:40] in normsub(TEXTS[j]):
                hits.add(j)
                break
    if not hits or qi >= len(TOP50C):
        continue
    tws = {TWIN.get(h) for h in hits} - {None}
    top = TOP50C[qi]
    ev_pool += [j for j in top if j in hits or j in tws]
    no_pool += [j for j in top if j not in hits and j not in tws][:8]
DELTA = D[np.array(ev_pool)].mean(0) - D[np.array(no_pool)].mean(0)
DELTA /= np.linalg.norm(DELTA)
PROJ = (D @ DELTA).astype(np.float32)
print("Δ estimated from %d ev / %d no (留出测试题)" % (len(ev_pool), len(no_pool)), flush=True)

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
print("fused done %.0fs" % (time.time() - t0), flush=True)

rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
FINAL = []
for i, q in enumerate(todo):
    top = np.argsort(-FUSED[i])[:50]
    sc = np.asarray(rer.compute_score([[q["question"], REC[MID[j]].get("raw") or ""] for j in top],
                                      batch_size=50), dtype=np.float32)
    zr = zs(sc)
    pz = zs(PROJ[top])
    row = FUSED[i].copy()
    row[top] = 0.6 * zr + 0.4 * FUSED[i][top] + 0.3 * pz
    # 问句降权(小权重)
    for r0, j in enumerate(top):
        if (REC[MID[j]].get("raw") or "").rstrip().endswith("?"):
            row[j] -= 0.15
    FINAL.append(row)
print("rerank+delta done %.0fs" % (time.time() - t0), flush=True)

# hyde
hyp_texts = [cached_hyp(q["question"]) for q in todo]
hyp_idx = [i for i, h in enumerate(hyp_texts) if h]
if hyp_idx:
    hvs = []
    for s in range(0, len(hyp_idx), 64):
        de, _ = bge_encode([hyp_texts[i] for i in hyp_idx[s:s + 64]])
        hvs.append(np.asarray(de, dtype=np.float32))
    hvs = l2n(np.concatenate(hvs))
    hws = ollama_embed([hyp_texts[i] for i in hyp_idx])
    for k, i in enumerate(hyp_idx):
        FINAL[i] = 0.75 * FINAL[i] + 0.25 * (zs(D @ hvs[k]) + zs(QW @ hws[k]))
print("hyde extras:", len(hyp_idx), flush=True)

def assemble(q, i):
    qkeys, qns = q_date_keys(q["question"]), q_names(q["question"])
    row = FINAL[i]
    top = list(np.argsort(-row)[:50])
    scored = []
    for r0, j in enumerate(top[:30]):
        rec = REC[MID[j]]
        s = 0.8 * (1.0 - r0 / 30.0) + 0.3 * slot_score(qkeys, qns, rec)
        scored.append((s, j))
    scored.sort(key=lambda x: -x[0])
    sel = [j for _, j in scored]
    seen, sel_d = set(), []
    for j in sel:
        tw = TWIN.get(j)
        if tw is not None and tw in seen:
            continue
        seen.add(j)
        sel_d.append(j)
    sel = sel_d
    adj = []
    for j in sel[:5]:
        raw = (REC[MID[j]].get("raw") or "").rstrip()
        if raw.endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
            adj.append(j + 1)
    base = sel[:10] + [a for a in adj if a not in sel[:10]]
    covered = set()
    for j in base:
        covered |= set(re.findall(r"[a-z']+", (REC[MID[j]].get("raw") or "").lower()))
    qw = set(qwords(q["question"]))
    extra = []
    for j in sel[10:30]:
        if j in base:
            continue
        w = set(re.findall(r"[a-z']+", (REC[MID[j]].get("raw") or "").lower()))
        extra.append((len({x for x in w if x in qw} - covered), j))
        covered |= w
    extra.sort(key=lambda x: -x[0])
    final_ids = base + [j for _, j in extra][:max(0, 25 - len(base))]
    return [REC[MID[j]] for j in final_ids]

smoke = assemble(todo[0], 0)
assert len(smoke) >= 5, "GATE FAIL assemble"
print("smoke ok: %d recs" % len(smoke), flush=True)

import threading
_ELOCK = threading.Lock()
IDX = {q["qa_id"]: i for i, q in enumerate(todo)}

def work(q):
    i = IDX[q["qa_id"]]
    try:
        with _ELOCK:
            recs = assemble(q, i)
        ctx1 = build_rows(recs)
        cal = calendar_lines(recs)
        if cal:
            ctx1 = ctx1 + NL + NL.join(cal)
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
            with _ELOCK:
                recs2 = assemble(q, i)
            ctx2 = build_rows(recs2[:40]) + NL + NL.join(calendar_lines(recs2))
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
with open(HERE + "/submission_r35.jsonl", "w", encoding="utf-8") as f:
    for s in submission:
        f.write(json.dumps(dict(qa_id=s["qa_id"], predicted_answer=s["predicted_answer"]),
                           ensure_ascii=False) + NL)
qids = set(s["qa_id"] for s in submission)
with open(HERE + "/questions_r35.jsonl", "w", encoding="utf-8") as f:
    for q in ALLQS:
        if q["qa_id"] in qids:
            f.write(json.dumps(q, ensure_ascii=False) + NL)
print("QUESTIONS_SUBSET_WRITTEN r35", flush=True)
