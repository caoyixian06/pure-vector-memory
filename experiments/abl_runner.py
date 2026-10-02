# -*- coding: utf-8 -*-
"""abl_runner.py — r40架构消融: 500题(同r40a/b子集, seed 20260918)
答题=qwen3.8-27b(专属实例), 判官=qwen3-14b官方(并发4). 基线: r40b=76.4%.
一进程串行8臂, 每臂独立落盘(submission+score+results行), 单臂失败不挡队列.
臂: ctrl / no_atom / no_delta / no_words / no_pass2 / no_marks / marks_fixed / win10
"""
import io, json, os, sys, re, time, random, urllib.request, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
QWEN_KEY = "${QWEN_KEY}"
QWEN_API = "https://${QWEN_MAAS_HOST}/compatible-mode/v1/chat/completions"
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
NL = chr(10)
ARMS = ["ctrl", "no_atom", "no_delta", "no_words", "no_pass2", "no_marks", "marks_fixed", "win10"]

from FlagEmbedding import BGEM3FlagModel, FlagReranker
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def bge_encode(texts):
    out = bge.encode(texts, return_dense=True, return_sparse=True)
    return out['dense_vecs'], out['lexical_weights']
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
def toks_list(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s).lower()) if len(w) > 2]
COUR = re.compile(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank|cool|nice)\b", re.I)

# ===== 原子句库 v2切分 =====
atoms = []
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
    sess = r.get("session_id") or ""
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        if len(words) <= 5 and COUR.search(p):
            continue
        atoms.append((p, sess, r.get("memory_id")))
A2VEC = emb([a[0] for a in atoms])
A2TOK = [toks_set(a[0]) for a in atoms]
print("原子句:", len(atoms), flush=True)

zw = np.load(HERE + "/word_vecs.npz")
WV = l2n(zw["V"].astype(np.float32))
WH = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
W2I = {}
for idx, w in enumerate(WH.keys()):
    c0 = w.strip(":").lower()
    if c0 and c0 not in W2I:
        W2I[c0] = idx
def wv(word):
    i2 = W2I.get(word)
    return WV[i2] if i2 is not None else None
A2WV = []
for p, sess, mid in atoms:
    vs = [wv(w) for w in toks_list(p)]
    vs = [v for v in vs if v is not None]
    A2WV.append(np.stack(vs) if vs else np.zeros((1, 256), dtype=np.float32))
print("词向量预存完成", flush=True)

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

WH2 = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
CLEAN = {}
for w in WH2.keys():
    c0 = w.strip(":").lower()
    if c0 and c0 not in CLEAN:
        CLEAN[c0] = w
STOP = set("a an the is are was were be been being to of in on at for with about from by as and or but if so not no that this these those there it its i you he she they we my your his her their what when where who how why did do does s t re ve ll d m have has had will would can could should just really very much more most some any all".split())
HOST_IDX = {w: [MID2I[h2] for h2 in hs if h2 in MID2I] for w, hs in WH2.items()}
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

def ans_api(prompt, max_tokens=400):
    body = {"model": "qwen3.8-27b", "max_tokens": max_tokens, "temperature": 0.0,
            "enable_thinking": False, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}
    for a in range(3):
        try:
            req = urllib.request.Request(QWEN_API, json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + QWEN_KEY})
            with urllib.request.urlopen(req, timeout=180) as r2:
                d = json.loads(r2.read())
            t = "".join(ch.get("message", {}).get("content", "") or ""
                        for ch in d.get("choices", [])).strip()
            if t:
                return t
            time.sleep(1)
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

# ===== 测试集: r40a/b同款500题 =====
ALLQS = [json.loads(l) for l in open(QS, encoding="utf-8") if l.strip()]
prev = json.load(open(HERE + "/subset_all1382.json", encoding="utf-8"))
rng = random.Random(20260918)
test = rng.sample(prev, 500)
test_ids = {t["qa_id"] for t in test}
qmap = {q["qa_id"]: q for q in ALLQS}
todo = [qmap[qid] for qid in test_ids if qid in qmap]
IDX = {q["qa_id"]: i for i, q in enumerate(todo)}
print("test questions:", len(todo), flush=True)

# ===== 检索预计算(一次) =====
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
PROJ_D = None  # 每臂重算Δ方向(与r40runner一致的退化方向)
print("fusion inputs ready", flush=True)

sess2idx = {}
for k2, a in enumerate(atoms):
    sess2idx.setdefault(a[1], []).append(k2)
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("reranker ready", flush=True)

def pass1_ids(order, q, n_summ, n_rest, use_adj):
    seen, d = set(), []
    for j in order:
        tw = TWIN.get(j)
        if tw is not None and tw in seen:
            continue
        if j in seen:
            continue
        seen.add(j)
        d.append(j)
    summ = [j for j in d if KIND[j] != "raw"][:n_summ]
    rest = [j for j in d if j not in summ][:n_rest]
    base = summ + rest
    if not use_adj:
        return base
    adj = []
    for j in base[:5]:
        raw = (REC[MID[j]].get("raw") or "").rstrip()
        if raw.endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
            adj.append(j + 1)
    return base + [a for a in adj if a not in base]

CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
ANCHORS = ["不知道", "无法确定", "没有提到", "记忆中没有相关信息", "不确定"]
_ade, _alw = bge_encode(ANCHORS)
ANCHOR_V = l2n(np.asarray(_ade, dtype=np.float32))
def is_negative_semantic(ans):
    if not (ans or "").strip():
        return True
    de, _ = bge_encode([ans[:200]])
    av = l2n(np.asarray(de, dtype=np.float32))[0]
    return bool((ANCHOR_V @ av).max() >= 0.55)

SIG = {}
def build_rows(recs, cap=250, sig=None, use_marks=True):
    s2 = SIG if sig is None else sig
    parts = []
    for r in recs:
        raw = (r.get("raw") or "").strip()
        if raw.startswith("[Session"):
            continue
        n2 = SESS_NUM.get(r.get("memory_id"))
        d2 = SESS_DATE.get(r.get("memory_id")) or ""
        mark = ""
        if use_marks:
            sg = s2.get(r.get("memory_id"), 0)
            mark = "★" if sg >= 2 else ("◇" if sg == 1 else "")
        parts.append("[Session %s — %s]%s %s" % (n2 if n2 is not None else "?", d2, mark, raw[:cap]))
    return NL.join(parts) if parts else "(无记忆命中)"

TIME_RULE = ("只输出答案。涉及时间的问题必须输出换算后的绝对日期(如 2023年5月7日),"
             "禁止保留 yesterday/last year/上个月 等相对表述。")

PROMPT = """You are an answer grader for a conversational memory benchmark.
Question: {q}
Gold answer(s): {g}
Predicted answer: {p}

Grade the prediction by these principles: "Inclusive without contradiction, complete without overreach".
Correct ONLY IF: the prediction includes all required information, does not contradict the gold answer,
adds no unsupported details, preserves temporal granularity, and for list answers misses/adds nothing.
Minor wording differences, language (Chinese/English), or date format differences are acceptable.

Output EXACTLY one JSON object and nothing else: {{"score": 1}} or {{"score": 0}}"""
def judge(r):
    q = qmap.get(r["qa_id"], {})
    gold = "; ".join(str(x) for x in (q.get("answer") or []))
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

# ===== 真gold重判既有submission(软判分→真判分校准) =====
def rejudge(tag, path):
    if not os.path.exists(path):
        print("rejudge skip(no file):", tag, flush=True)
        return
    out = HERE + "/rejudge_%s.txt" % tag
    if os.path.exists(out):
        print("rejudge skip(done):", tag, flush=True)
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    with ThreadPoolExecutor(16) as jex:
        scores = list(jex.map(judge, rows))
    ok = sum(1 for s in scores if s == 1)
    valid = sum(1 for s in scores if s in (0, 1))
    pct = round(100 * ok / max(1, valid), 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write("%s(gold) %d/%d = %.1f%%\n" % (tag, ok, valid, pct))
    print("REJUDGE", tag, ok, "/", valid, "=", pct, "%", flush=True)

rejudge("bf_full1382", HERE + "/submission_r40bf.jsonl")
rejudge("r40b_500", HERE + "/submission_r40.jsonl")

# ===== 臂循环 =====
ELOCK = threading.Lock()
RESULTS = HERE + "/ablation_results.txt"
done_arms = set()
if os.path.exists(RESULTS):
    for ln in open(RESULTS, encoding="utf-8"):
        m = re.match(r"(\S+)\s+", ln.strip())
        if m and "= " in ln:
            done_arms.add(m.group(1))
print("already done:", sorted(done_arms), flush=True)

for arm in ARMS:
    if arm in done_arms:
        print("skip", arm, flush=True)
        continue
    t0 = time.time()
    try:
        # --- 检索层(按臂开关) ---
        use_atom = arm not in ("no_atom",)
        use_delta = arm not in ("no_delta",)
        use_words = arm not in ("no_words",)
        SIGS = {}
        FUSED = []
        for i in range(len(todo)):
            qv = bQ[i] + (u2 if True else None)  # u2属r37底盘, 所有臂保留
            c1 = zs(D @ l2n(qv[None])[0])
            c2 = zs(QW @ wQ[i]) if use_words else np.zeros(N, dtype=np.float32)
            FUSED.append(c1 + c2 + 0.5 * zs(word_votes(QL[i])))
        FINAL = []
        for i, q in enumerate(todo):
            fused = FUSED[i]
            sid = "loco-" + str(q.get("sample_id"))
            idxs = sess2idx.get(sid, [])
            atom_boost = {}
            if idxs and use_atom:
                sims = A2VEC[idxs] @ bQ[i]
                qtok = toks_set(q["question"])
                cov = np.array([len(A2TOK[idxs[k2]] & qtok) for k2 in range(len(idxs))], dtype=np.float32)
                comb = sims + 0.5 * cov
                for oi in np.argsort(-comb)[:6]:
                    j = MID2I.get(atoms[idxs[oi]][2])
                    if j is not None:
                        atom_boost[j] = max(atom_boost.get(j, 0), float(comb[oi]))
            SIG.clear()
            top = np.argsort(-fused)[:50]
            sc = np.asarray(rer.compute_score([[q["question"], REC[MID[j]].get("raw") or ""] for j in top],
                                              batch_size=50), dtype=np.float32)
            inside = np.argsort(-sc)
            newf = fused.copy()
            newf[top[inside]] = np.linspace(50, 1, 50)
            for j, ab in atom_boost.items():
                if j < N:
                    newf[j] += 1.5 * ab
            ab_med = np.median(list(atom_boost.values())) if atom_boost else 1e9
            for j, ab in atom_boost.items():
                if j >= N:
                    continue
                dp = float(D[j] @ l2n(stmt - qc)) if j < len(D) else 0
                sg = 2 if (ab >= ab_med and dp > 0) else (1 if (ab >= ab_med or dp > 0) else 0)
                SIG[MID[j]] = max(SIG.get(MID[j], 0), sg)
            if use_delta:
                order_pre = np.argsort(-newf)
                newf[order_pre] = newf[order_pre] + 0.6 * zs(np.asarray(D @ l2n(stmt - qc), dtype=np.float32)[order_pre])
            FINAL.append(newf)
            SIGS[i] = dict(SIG)
        print(arm, "retrieval done %.0fs" % (time.time() - t0), flush=True)

        # --- 答题层(按臂开关) ---
        n_summ, n_rest = (4, 6) if arm == "win10" else (6, 9)
        use_adj = True
        use_pass2 = arm != "no_pass2"
        use_marks = arm not in ("no_marks",)
        fixed_marks = arm == "marks_fixed"

        def work(q):
            i = IDX[q["qa_id"]]
            order = np.argsort(-FINAL[i])
            ids1 = pass1_ids(order, q, n_summ, n_rest, use_adj)
            sig_arg = SIGS[i] if fixed_marks else None
            try:
                ctx1 = build_rows([REC[MID[j]] for j in ids1], sig=sig_arg, use_marks=use_marks)
            except Exception as e:
                print("RETR_FAIL", q["qa_id"], repr(e)[:100], flush=True)
                ctx1 = "(retrieval error)"
            p1 = ("根据记忆回答。若无关回答「不知道」。" + NL +
                  "记忆(每条已标注所属会话与日期):" + NL + ctx1 + NL + NL +
                  "问题: " + q["question"] + NL + TIME_RULE)
            ans1 = ans_api(p1)
            final, second = ans1, False
            if use_pass2:
                with ELOCK:
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
                        ctx2 = build_rows([REC[MID[j]] for j in d2[:30]], sig=sig_arg, use_marks=use_marks)
                        p2 = ("根据以下原始对话记录回答问题。" + NL +
                              "记录(已标注会话与日期):" + NL + ctx2 + NL + NL +
                              "问题: " + q["question"] + NL + TIME_RULE +
                              " 确实没有答案才回答「不知道」。")
                        ans2 = ans_api(p2, max_tokens=300)
                        with ELOCK:
                            neg2 = is_negative_semantic(ans2)
                        if not neg2:
                            final, second = ans2, True
                    except Exception as e:
                        print("PASS2_FAIL", q["qa_id"], repr(e)[:100], flush=True)
            print("ANS", arm, q["qa_id"], flush=True)
            return dict(qa_id=q["qa_id"], predicted_answer=final, second_pass=second)

        with ThreadPoolExecutor(16) as ex:
            submission = list(ex.map(work, todo))
        with open(HERE + "/abl_%s_submission.jsonl" % arm, "w", encoding="utf-8") as f:
            for s in submission:
                f.write(json.dumps(s, ensure_ascii=False) + NL)
        print(arm, "SUBMISSION_DONE n=%d 2nd=%d %.0fs" % (
            len(submission), sum(1 for s in submission if s.get("second_pass")),
            time.time() - t0), flush=True)

        # --- 判官 ---
        _js = {"n": 0}
        def _jt(r):
            s = judge(r)
            _js["n"] += 1
            if _js["n"] % 100 == 0:
                print("judge", arm, _js["n"], flush=True)
            return s
        with ThreadPoolExecutor(16) as jex:
            scores = list(jex.map(_jt, submission))
        ok = sum(1 for s in scores if s == 1)
        valid = sum(1 for s in scores if s in (0, 1))
        pct = round(100 * ok / max(1, valid), 1)
        print("ABL", arm, ok, "/", valid, "=", pct, "%", flush=True)
        with open(HERE + "/abl_%s_score.txt" % arm, "w", encoding="utf-8") as f:
            f.write("%s %d/%d = %.1f%%\n" % (arm, ok, valid, pct))
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write("%s %d/%d = %.1f%% (%.0fs)\n" % (arm, ok, valid, pct, time.time() - t0))
    except Exception as e:
        import traceback
        print("ARM_FAIL", arm, repr(e)[:200], flush=True)
        traceback.print_exc()
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write("%s FAIL %s\n" % (arm, repr(e)[:100]))
print("ABL_ALL_DONE", flush=True)
