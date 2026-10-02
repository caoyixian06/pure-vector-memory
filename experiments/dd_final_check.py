# dd_final_check: raw-kind composition of deployed pool + stacked rerank boost
import dd_common as C
import numpy as np, re, traceback
from collections import defaultdict

try:
    ISRAW = np.array([1.0 if C.KIND_OF_MID[m] == "raw" else 0.0 for m in C.MIDS])
    conv_raws = defaultdict(list)
    for m, row in C.MID2ROW.items():
        if C.KIND_OF_MID[m] == "raw":
            conv_raws[C.CONV_OF_MID[m]].append((C.TURN_OF_MID[m], row))
    for k in conv_raws:
        conv_raws[k].sort()
    MONTHS = set("january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday session".split())
    STARTER = set("i the we it but and so then my he she they you that this what when where how why did do have has was were after before well oh yeah yes no okay alright anyway wow man hey hi hello thanks thank good great nice cool really actually probably maybe listen look lets let just know think".split())
    spk = set(str(v).lower() for v in C.SPK_OF_MID.values() if v not in (None, "?", ""))
    def ents(text):
        toks = re.findall(r"[A-Za-z]{3,}", text)
        return set(t.lower() for t in toks[1:] if t[0].isupper() and t.lower() not in STARTER and t.lower() not in MONTHS and t.lower() not in spk)
    ISFIRST = np.zeros(len(C.MIDS))
    for k, lst in conv_raws.items():
        seen = set()
        for t, row in lst:
            if ents(C.RAWTEXT_OF_MID.get(C.MIDS[row], "")) - seen:
                ISFIRST[row] = 1.0
            seen |= ents(C.RAWTEXT_OF_MID.get(C.MIDS[row], ""))

    print("Z1_corpus_raw_frac", round(float(ISRAW.mean()), 4))
    fr50, fr1 = [], []
    eval_idx = [i for i, qm in enumerate(C.QMETA) if C.QROW[i] >= 0 and qm["ev_rows"]]
    for i in eval_idx:
        j = C.QROW[i]
        t50 = C.TOP50[j]
        fr50.append(float(ISRAW[t50].mean()))
        top1 = int(t50[np.argmax(C.RER[j])])
        if top1 not in set(C.QMETA[i]["ev_rows"]):
            fr1.append(ISRAW[top1])
    print("Z2_top50_raw_frac", round(float(np.mean(fr50)), 4), "rer_top1noise_raw_frac", round(float(np.mean(fr1)), 4), "n_noise_top1", len(fr1))

    # stacked CV rerank boost
    fl = C.folds_by_conv([C.QMETA[i]["conv"] for i in eval_idx], 5)
    def ev1(idx, a, b):
        hit = 0
        for i in idx:
            j = C.QROW[i]
            t50 = C.TOP50[j]
            f1 = ISRAW[t50]; f1 = (f1 - f1.mean()) / (f1.std() + 1e-9)
            f2 = ISFIRST[t50]; f2 = (f2 - f2.mean()) / (f2.std() + 1e-9)
            sc = C.RER[j] + a * f1 + b * f2
            if int(t50[np.argmax(sc)]) in set(C.QMETA[i]["ev_rows"]):
                hit += 1
        return hit / len(idx)
    grid = [(a, b) for a in [0, 0.1, 0.2, 0.3, 0.5] for b in [0, 0.05, 0.1, 0.2]]
    base, aft, picks = [], [], []
    for fi in range(5):
        te = [eval_idx[x] for x in fl[fi]]
        tr = [eval_idx[x] for x in np.concatenate([fl[x] for x in range(5) if x != fi])]
        base.append(ev1(te, 0, 0))
        best, bv = (0, 0), -1
        for a, b in grid:
            if a == 0 and b == 0:
                continue
            v = ev1(tr, a, b)
            if v > bv:
                bv, best = v, (a, b)
        aft.append(ev1(te, *best))
        picks.append(best)
    print("Z3_ev@1_base", round(float(np.mean(base)), 4), "stacked", round(float(np.mean(aft)), 4), "picks", picks)
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
