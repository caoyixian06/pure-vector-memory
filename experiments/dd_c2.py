# dd_c2: first-mention + session-head preference as rerank features
import dd_common as C
import numpy as np, re, traceback
from collections import defaultdict

try:
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
    first_of_conv, sesspos = {}, {}
    for k, lst in conv_raws.items():
        seen = set()
        fs = set()
        hs = C.HDR_TURNS.get(k, [])
        import bisect
        groups = defaultdict(list)
        for t, row in lst:
            es = ents(C.RAWTEXT_OF_MID.get(C.MIDS[row], ""))
            if es - seen:
                fs.add(row)
            seen |= es
            groups[bisect.bisect_right(hs, t) if hs else 0].append(row)
        for si, rows in groups.items():
            for j2, row in enumerate(rows):
                sesspos[row] = j2 / max(len(rows) - 1, 1)
        first_of_conv[k] = fs
    ISFIRST = np.zeros(len(C.MIDS))
    for k, fs in first_of_conv.items():
        for r in fs:
            ISFIRST[r] = 1.0

    # session-head pair AUC
    y, s = [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None:
            continue
        evp = [r for r in t50 if r in ev]
        for r in evp:
            y.append(1); s.append(sesspos.get(int(r), 0.5))
        for r in noise[:10]:
            y.append(0); s.append(sesspos.get(int(r), 0.5))
    print("C2_1_sesspos_pairAUC", round(C.auc(y, s), 4), "n", len(y))

    eval_idx = [i for i, qm in enumerate(C.QMETA) if C.QROW[i] >= 0 and qm["ev_rows"]]
    fl = C.folds_by_conv([C.QMETA[i]["conv"] for i in eval_idx], 5)

    def ev1(idx, g1, g2):
        hit = 0
        for i in idx:
            j = C.QROW[i]
            t50 = C.TOP50[j]
            f1 = ISFIRST[t50]
            f2 = np.array([sesspos.get(int(r), 0.5) for r in t50])
            f1 = (f1 - f1.mean()) / (f1.std() + 1e-9)
            f2 = (f2 - f2.mean()) / (f2.std() + 1e-9)
            sc = C.RER[j] + g1 * f1 + g2 * (-f2)
            if int(t50[np.argmax(sc)]) in set(C.QMETA[i]["ev_rows"]):
                hit += 1
        return hit / len(idx)

    base_all, best_all, pick = [], [], []
    grid = [(a, b) for a in [0, 0.05, 0.1, 0.15, 0.2, 0.3] for b in [0, 0.05, 0.1, 0.15, 0.2, 0.3]]
    for fi in range(5):
        te = [eval_idx[x] for x in fl[fi]]
        tr = [eval_idx[x] for x in np.concatenate([fl[x] for x in range(5) if x != fi])]
        b0 = ev1(te, 0, 0)
        best, bv = (0, 0), -1
        for g1, g2 in grid:
            if g1 == 0 and g2 == 0:
                continue
            v = ev1(tr, g1, g2)
            if v > bv:
                bv, best = v, (g1, g2)
        base_all.append(b0)
        best_all.append(ev1(te, *best))
        pick.append(best)
    print("C2_2_ev@1_before", round(float(np.mean(base_all)), 4), "after", round(float(np.mean(best_all)), 4),
          "picked", pick[:5])
    # feature AUCs individually on pools
    for tag in ["isfirst"]:
        y2, s2 = [], []
        for i, qm in enumerate(C.QMETA):
            j, ev, t50, noise, rer = C.pools(i)
            if j < 0 or not ev or noise is None:
                continue
            evp = [r for r in t50 if r in ev]
            for r in evp:
                y2.append(1); s2.append(ISFIRST[int(r)])
            for r in noise[:10]:
                y2.append(0); s2.append(ISFIRST[int(r)])
        print("C2_3_isfirst_pairAUC", round(C.auc(y2, s2), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
