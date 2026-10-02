# -*- coding: utf-8 -*-
"""audit_window.py — 验证bf全量549错题: 金证据是否真的进窗+排名靠前
逐题重建真实窗口(pass1=6摘要+9混合+邻句带回, raw[:250]截断; 负答案题补pass2=30行)
三层判定: A记录进窗(身份) B文本可见(截断后) C全局名次(不在窗时)
"""
import io, json, os, re, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"
QS = "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl"
NL = chr(10)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# ===== 分数 =====
score = {}
for l in io.open(HERE + "/smoke100_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m:
        s = m.group(2)
        score[m.group(1)] = int(s) if len(s) == 1 and s in "01" else -1
print("scores:", len(score), flush=True)

# ===== 预测(负答案判定) =====
pred = {}
for l in io.open(HERE + "/submission_r40bf.jsonl", encoding="utf-8"):
    r = json.loads(l)
    pred[r["qa_id"]] = str(r.get("predicted_answer") or "")
DK = re.compile(r"不知道|无法确定|没有相关信息|无法回答|not mentioned|no information|unclear|cannot", re.I)

# ===== 题目+证据 =====
QSMap = {}
for l in io.open(QS, encoding="utf-8"):
    q = json.loads(l)
    QSMap[q["qa_id"]] = q

# ===== 库记账(与runner完全一致) =====
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
RAW250 = [x[:250] for x in RAWFULL]
NORM_FULL = [norm(x) for x in RAWFULL]
NORM_250 = [norm(x) for x in RAW250]
TWIN = {}
for i, m in enumerate(MID):
    rec = REC.get(m, {})
    tw = rec.get("raw_of") or rec.get("twin_of")
    if tw and tw in MID2I:
        TWIN[i] = MID2I[tw]
CONVKEY = [re.sub(r"_(rbak\d+k|m\d+)$", "", m) for m in MID]
print("lib:", N, flush=True)

# ===== ckpt =====
z = np.load(HERE + "/r40bf_ckpt.npz")
FINAL = z["FINAL"]
IDS = [str(x) for x in z["IDS"]]
IDX = {q: i for i, q in enumerate(IDS)}
print("ckpt:", FINAL.shape, flush=True)

def pass1_ids(order):
    seen, d = set(), []
    for j in order:
        tw = TWIN.get(j)
        if tw is not None and tw in seen:
            continue
        if j in seen:
            continue
        seen.add(j)
        d.append(int(j))
    summ = [j for j in d if KIND[j] != "raw"][:6]
    rest = [j for j in d if j not in summ][:9]
    base = summ + rest
    adj = []
    for j in base[:5]:
        raw = (REC[MID[j]].get("raw") or "").rstrip()
        if raw.endswith("?") and j + 1 < N and CONVKEY[j + 1] == CONVKEY[j]:
            adj.append(j + 1)
    return base + [a for a in adj if a not in base], d

def ev_keys(q):
    out = []
    for em in (q.get("evidence_messages") or []):
        t = norm(em.get("text") or "")
        if len(t) >= 20:
            out.append(t[:60])
    return out

RANKBUCKET = np.zeros(N, dtype=np.int64)
rank_of = np.empty(N, dtype=np.int64)
rank_of[:] = -1
rows_out = []
def analyze(qa_id):
    q = QSMap[qa_id]
    i = IDX[qa_id]
    order = np.argsort(-FINAL[i])
    win1, dedup = pass1_ids(order)
    keys = ev_keys(q)
    if not keys:
        return None
    rankbucket = {}
    pos1 = vis1 = pos2 = vis2 = None
    for pos, j in enumerate(win1, 1):
        for k in keys:
            if k in NORM_FULL[j]:
                if pos1 is None or pos < pos1:
                    pos1 = pos
                if k in NORM_250[j]:
                    if vis1 is None or pos < vis1:
                        vis1 = pos
                break
    p = pred.get(qa_id, "")
    if DK.search(p):
        seen2, d2 = set(), []
        for j in order[:60]:
            tw = TWIN.get(j)
            if tw is not None and tw in seen2 or int(j) in seen2:
                continue
            seen2.add(int(j)); d2.append(int(j))
        win2 = d2[:30]
        for pos, j in enumerate(win2, 1):
            for k in keys:
                if k in NORM_FULL[j]:
                    if pos2 is None or pos < pos2:
                        pos2 = pos
                    if k in NORM_250[j]:
                        if vis2 is None or pos < vis2:
                            vis2 = pos
                    break
    # 全局名次: 只在不在窗时算
    grank = None
    if pos1 is None and pos2 is None:
        for r, j in enumerate(order, 1):
            j = int(j)
            for k in keys:
                if k in NORM_FULL[j]:
                    grank = r
                    break
            if grank:
                break
    return dict(qa_id=qa_id, pos1=pos1, vis1=vis1, pos2=pos2, vis2=vis2, grank=grank)

errs = [i for i in IDS if score.get(i) == 0]
oks = [i for i in IDS if score.get(i) == 1]
print("errors=%d ok=%d" % (len(errs), len(oks)), flush=True)
res_e, res_o = [], []
for c, qa in enumerate(errs):
    r = analyze(qa)
    if r:
        r["bucket"] = "dk" if DK.search(pred.get(qa, "")) else "other"
        res_e.append(r)
    if c % 100 == 0:
        print("err", c, flush=True)
for qa in oks:
    r = analyze(qa)
    if r:
        r["ok"] = True
        r["bucket"] = "ok"
        res_o.append(r)

def stat(res, name):
    n = len(res)
    if not n:
        return
    inw = [r for r in res if r["pos1"] is not None]
    vis = [r for r in res if r["vis1"] is not None]
    b13 = sum(1 for r in inw if r["pos1"] <= 3)
    b49 = sum(1 for r in inw if 4 <= r["pos1"] <= 9)
    b1015 = sum(1 for r in inw if 10 <= r["pos1"] <= 15)
    dk2 = [r for r in res if r["bucket"] == "dk"]
    p2 = [r for r in dk2 if r["pos2"] is not None]
    absent = [r for r in res if r["pos1"] is None and r["pos2"] is None]
    gr = sorted(r["grank"] for r in absent if r["grank"])
    print("[%s] n=%d 记录进窗=%.1f%% 截断可见=%.1f%% | 窗内位次 1-3:%d 4-9:%d 10-15:%d" % (
        name, n, 100.0*len(inw)/n, 100.0*len(vis)/n, b13, b49, b1015))
    print("    负答案题=%d 其中pass2窗内=%d | 完全不在窗=%d 全局名次中位=%s" % (
        len(dk2), len(p2), len(absent), (gr[len(gr)//2] if gr else "-")))

print("\n===== 错题(n=%d) =====" % len(res_e))
stat(res_e, "ERR全部")
stat([r for r in res_e if r["bucket"] == "dk"], "ERR不知道型")
stat([r for r in res_e if r["bucket"] == "other"], "ERR其他")
print("===== 对题(n=%d) 校准 =====" % len(res_o))
stat(res_o, "OK全部")
with open(HERE + "/window_audit.jsonl", "w", encoding="utf-8") as f:
    for r in res_e:
        f.write(json.dumps(r, ensure_ascii=False) + NL)
    for r in res_o:
        f.write(json.dumps(r, ensure_ascii=False) + NL)
print("WINDOW_AUDIT_DONE", flush=True)
