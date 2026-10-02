# -*- coding: utf-8 -*-
"""audit_why135.py — 135道"证据不在窗"错题的逐题验尸
每题输出: 证据形态(问句?代词?寒暄?长度) / 问题-证据词面重叠 / 孪生摘要是否在窗 /
         占据窗口前排的是什么 / 证据在FINAL下的全局名次
"""
import io, json, os, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
NL = chr(10)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2)

score = {}
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m:
        s = m.group(2)
        score[m.group(1)] = int(s) if len(s) == 1 and s in "01" else -1
pred = {}
for l in io.open(HERE + "/submission_r40bf.jsonl", encoding="utf-8"):
    r = json.loads(l)
    pred[r["qa_id"]] = str(r.get("predicted_answer") or "")
DK = re.compile(r"不知道|无法确定|没有相关信息|无法回答|not mentioned|no information|unclear|cannot", re.I)
QSMap = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    QSMap[q["qa_id"]] = q

MID, KIND = [], []
for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    MID.append(r["mid"]); KIND.append(r.get("kind") or "summary")
MID2I = {m: i for i, m in enumerate(MID)}
N = len(MID)
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
RAWFULL = [rec_raw(m) for m in MID]
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]

z = np.load(HERE + "/r40bf_ckpt.npz")
FINAL = z["FINAL"]
IDS = [str(x) for x in z["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}

def ev_list(q):
    out = []
    for em in (q.get("evidence_messages") or []):
        t = str(em.get("text") or "")
        if len(norm(t)) >= 20:
            out.append((t, norm(t)[:60], em.get("speaker") or "", em.get("session_index"), em.get("message_index")))
    return out

def pass1_window(order):
    seen, d = set(), []
    for j in order:
        tw = TWIN.get(j)
        if tw is not None and tw in seen or int(j) in seen:
            continue
        seen.add(int(j)); d.append(int(j))
    summ = [j for j in d if KIND[j] != "raw"][:6]
    rest = [j for j in d if j not in summ][:9]
    base = summ + rest
    adj = []
    for j in base[:5]:
        raw = (REC[MID[j]].get("raw") or "").rstrip()
        if raw.endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
            adj.append(j + 1)
    return base + [a for a in adj if a not in base], seen

errs = [i for i in IDS if score.get(i) == 0]
stats = {"n": 0, "twin_in_win": 0, "ev_is_question": 0, "ev_short": 0, "ev_greet": 0,
         "ev_long": 0, "overlap0": 0, "overlap_ge3": 0}
dump = []
PRON = re.compile(r"^(it|that|this|they|he|she|we) ", re.I)
GREET = re.compile(r"\b(haha|lol|wow|nice|cool|awesome|great|thanks|congrats|yay|same)\b", re.I)
for qa in errs:
    q = QSMap[qa]
    evs = ev_list(q)
    if not evs:
        continue
    i = IDX[qa]
    order = np.argsort(-FINAL[i])
    win1, seen = pass1_window(order)
    keys = [k for _, k, _, _, _ in evs]
    # 证据记录全局定位
    ev_recs = []
    for j in range(N):
        nj = NORMF = norm(RAWFULL[j])
        if not nj:
            continue
        for k in keys:
            if k in nj:
                ev_recs.append(j)
                break
    if not ev_recs:
        continue
    grank = min((list(map(int, order)).index(j) + 1 for j in ev_recs), default=None)
    # 只看完全不在窗的
    winset = set(win1)
    if any(j in winset for j in ev_recs):
        continue
    stats["n"] += 1
    qt = toks(q.get("question") or "")
    rec_info = []
    twin_hit = False
    for j in ev_recs[:2]:
        raw = RAWFULL[j]
        tw = TWIN.get(j)
        t_in = (tw in winset) if tw is not None else False
        twin_hit = twin_hit or t_in
        ov = len(qt & toks(raw))
        rec_info.append(dict(kind=KIND[j], len=len(raw), q=raw.rstrip().endswith("?"),
                             pron=bool(PRON.match(raw.split(": ", 1)[-1])), greet=bool(GREET.search(raw)),
                             ov=ov, twin_in_win=t_in, grank=None))
        stats["ev_is_question"] += rec_info[-1]["q"]
        stats["ev_short"] += (len(raw) < 40)
        stats["ev_greet"] += rec_info[-1]["greet"]
        stats["ev_long"] += (len(raw) > 300)
        stats["overlap0"] += (ov == 0)
        stats["overlap_ge3"] += (ov >= 3)
    stats["twin_in_win"] += twin_hit
    top5 = [RAWFULL[j][:90] for j in win1[:5]]
    dump.append(dict(qa_id=qa, grank=grank, bucket="dk" if DK.search(pred.get(qa, "")) else "other",
                     question=q.get("question"), gold=q.get("answer"),
                     ev=[e[0][:120] for e in evs[:2]], recs=rec_info, twin_in_win=twin_hit, top5=top5))
print("完全不在窗且库内可定位: %d" % stats["n"], flush=True)
print("孪生摘要在窗(形态换装失败): %d" % stats["twin_in_win"], flush=True)
print("证据是问句形态: %d | 证据<40字: %d | 证据含寒暄词: %d | 证据>300字: %d" % (
    stats["ev_is_question"], stats["ev_short"], stats["ev_greet"], stats["ev_long"]), flush=True)
print("问题-证据词面重叠=0: %d | >=3: %d" % (stats["overlap0"], stats["overlap_ge3"]), flush=True)
with io.open(HERE + "/why135.jsonl", "w", encoding="utf-8") as f:
    for r in dump:
        f.write(json.dumps(r, ensure_ascii=False) + NL)
print("\n===== 抽样8例 =====")
for r in dump[:8]:
    print("--- %s [%s] grank=%s" % (r["qa_id"], r["bucket"], r["grank"]))
    print(" Q:", r["question"][:70])
    print(" G:", "; ".join(str(x) for x in (r["gold"] or []))[:60])
    print(" E:", " || ".join(r["ev"])[:130])
    print(" rec:", json.dumps(r["recs"][0], ensure_ascii=False)[:150])
    print(" top1:", r["top5"][0][:80])
print("WHY135_DONE", flush=True)
