# -*- coding: utf-8 -*-
"""foundry122.py — 会话定位机理解剖(用户授权): 为什么max cos能找到金会话
①金max vs 陪衬max的gap分布 ②金max记录是什么(含答案词?) ③陪衬max是什么(话题相关?)"""
import io, json, re, sys, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry122_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)]
    return w
QSTOP = set("what when where who which how why did do does is are was were the a an of in on at to for and or with about from that this it its i you he she they we my your".split())
def stems_of(s):
    return set(stem(w) for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) > 2 and w not in QSTOP)

t0 = time.time()
SRC = "C:/locomo_refined/longmemeval/longmemeval_s_cleaned.json"
OUT = "C:/locomo_refined/longmemeval_mem"
d = json.load(open(SRC, encoding="utf-8"))
pairs = sum([list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])) for q in d], [])
sid2h = {}
for sid, dt, turns in pairs:
    sid2h[sid] = hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()
RAWS, SIDS = [], []
for l in io.open(OUT + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    RAWS.append(r.get("raw") or "")
    SIDS.append(r.get("session_id") or "")
D = l2n(np.load(OUT + "/mem_bge_dense.npz")["dense"].astype(np.float32))
X = l2n(np.load(OUT + "/q_bge1024.npy").astype(np.float32))
NOHDR = np.array([not r.startswith("[hdr") for r in RAWS])
P("loaded %.0fs" % (time.time() - t0))

gaps = []
gold_max_cos = []
noise_max_cos = []
gold_max_hitans = 0      # 金max记录含答案词干≥0.5
noise_max_hitans = 0     # 陪衬max含答案
gold_max_len = []
noise_max_len = []
gold_sess_nrec = []
noise_sess_nrec = []
top1_is_gold = 0
n = 0
for qi, q in enumerate(d):
    gold_h = set("lme-s" + sid2h[s][:12] for s in (q.get("answer_session_ids") or []) if s in sid2h)
    if not gold_h:
        continue
    hay_keys = set("lme-s" + sid2h[s][:12] for s in q["haystack_session_ids"] if s in sid2h)
    domain = [i for i in range(len(RAWS)) if SIDS[i] in hay_keys and NOHDR[i]]
    if not domain:
        continue
    ans_raw = q.get("answer")
    ans_txt = " ".join(str(a) for a in ans_raw) if isinstance(ans_raw, list) else str(ans_raw or "")
    ans_stems = stems_of(ans_txt)
    n += 1
    # 每会话max
    sess_max = {}
    sess_arg = {}
    for i in domain:
        c = float(D[i] @ X[qi])
        s2 = SIDS[i]
        if c > sess_max.get(s2, -9):
            sess_max[s2] = c
            sess_arg[s2] = i
    gmax = max(sess_max.get(s2, -9) for s2 in gold_h)
    nmax = max((v for k2, v in sess_max.items() if k2 not in gold_h), default=-9)
    gaps.append(gmax - nmax)
    gi = None
    for s2 in gold_h:
        if sess_max.get(s2, -9) == gmax:
            gi = sess_arg[s2]
            break
    ni = None
    for k2, v in sess_max.items():
        if k2 not in gold_h and v == nmax:
            ni = sess_arg[k2]
            break
    if gi is not None:
        gold_max_cos.append(gmax)
        gold_sess_nrec.append(sum(1 for i in domain if SIDS[i] in gold_h))
        gold_max_len.append(len(RAWS[gi].split()))
        if ans_stems and len(ans_stems & stems_of(RAWS[gi])) / len(ans_stems) >= 0.5:
            gold_max_hitans += 1
        if gmax > nmax:
            top1_is_gold += 1
    if ni is not None:
        noise_max_cos.append(nmax)
        noise_sess_nrec.append(sum(1 for i in domain if SIDS[i] not in gold_h) / max(1, len(sess_max) - len(gold_h)))
        noise_max_len.append(len(RAWS[ni].split()))
        if ans_stems and len(ans_stems & stems_of(RAWS[ni])) / len(ans_stems) >= 0.5:
            noise_max_hitans += 1
    if qi % 100 == 0:
        P("  %d %.0fs" % (qi, time.time() - t0))

gaps = np.array(gaps)
P("\n===== 会话定位机理(n=%d) =====" % n)
P("①gap(金max-陪max): 均值=%.3f 中位=%.3f  >0占比=%.1f%%  >0.02占比=%.1f%%" % (
    gaps.mean(), np.median(gaps), 100.0 * (gaps > 0).mean(), 100.0 * (gaps > 0.02).mean()))
P("  gap分位: p10=%.3f p25=%.3f p75=%.3f p90=%.3f" % tuple(np.quantile(gaps, [0.1, 0.25, 0.75, 0.9])))
P("②金会话max记录 vs 陪衬max记录:")
P("  max cos:      金=%.3f 陪衬=%.3f (差=%.3f)" % (
    np.mean(gold_max_cos), np.mean(noise_max_cos), np.mean(gold_max_cos) - np.mean(noise_max_cos)))
P("  含答案词≥50%%: 金=%.1f%% 陪衬=%.1f%%" % (
    100.0 * gold_max_hitans / n, 100.0 * noise_max_hitans / n))
P("  记录词数:     金=%.0f 陪衬=%.0f" % (np.mean(gold_max_len), np.mean(noise_max_len)))
P("  会话规模(记录数): 金=%.0f 陪衬均=%.0f" % (
    np.mean(gold_sess_nrec), np.mean(noise_sess_nrec)))
P("③金max>陪max的题(top1命中): %.1f%%" % (100.0 * top1_is_gold / n))
P("F122_DONE %.0fs" % (time.time() - t0))
