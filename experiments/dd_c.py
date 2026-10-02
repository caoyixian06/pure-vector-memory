# dd_c: within-conversation temporal structure + first-mention axis
import dd_common as C
import numpy as np, re, traceback, bisect
from collections import defaultdict

def bisect_r(hs, t):
    return bisect.bisect_right(hs, t)

try:
    # conversation -> ordered raw record rows
    conv_raws = defaultdict(list)
    for m, row in C.MID2ROW.items():
        if C.KIND_OF_MID[m] == "raw":
            conv_raws[C.CONV_OF_MID[m]].append((C.TURN_OF_MID[m], row))
    for k in conv_raws:
        conv_raws[k].sort()
    conv_n = {k: len(v) for k, v in conv_raws.items()}
    # normalized position (0..1) of each sparse row within its conversation raw sequence
    pos_of_row = {}
    sesspos_of_row = {}
    for k, lst in conv_raws.items():
        hs = C.HDR_TURNS.get(k, [])
        sess_groups = defaultdict(list)
        for r_, (t, row) in enumerate(lst):
            pos_of_row[row] = r_ / max(len(lst) - 1, 1)
            si = bisect_r(hs, t) if hs else 0
            sess_groups[si].append(row)
        for si, rows in sess_groups.items():
            for j, row in enumerate(rows):
                sesspos_of_row[row] = j / max(len(rows) - 1, 1)

    pos_ev, pos_no, spev, spno = [], [], [], []
    lastsess_ev, lastsess_no = [], []
    pair_y, pair_s = [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None or len(noise) == 0:
            continue
        evp = [r for r in t50 if r in ev]
        if not evp:
            continue
        pe = [pos_of_row.get(r, np.nan) for r in evp]
        pn = [pos_of_row.get(int(r), np.nan) for r in noise]
        pe = [x for x in pe if not np.isnan(x)]; pn = [x for x in pn if not np.isnan(x)]
        pos_ev += pe; pos_no += pn
        spev += [sesspos_of_row.get(r, np.nan) for r in evp]
        spno += [sesspos_of_row.get(int(r), np.nan) for r in noise]
        for r in evp:
            pair_y.append(1); pair_s.append(pos_of_row.get(r, 0.5))
        for r in noise[:10]:
            pair_y.append(0); pair_s.append(pos_of_row.get(int(r), 0.5))
        # last-session indicator
        nc = len(C.HDR_TURNS.get(qm["conv"], [1]))
        for r in evp:
            lastsess_ev.append(1 if C.sess_idx_of(C.MIDS[r]) >= nc - 1 else 0)
        for r in noise[:10]:
            lastsess_no.append(1 if C.sess_idx_of(C.MIDS[int(r)]) >= nc - 1 else 0)
    spev = [x for x in spev if not np.isnan(x)]; spno = [x for x in spno if not np.isnan(x)]
    print("C1_pos_ev", round(float(np.mean(pos_ev)), 4), "pos_noise", round(float(np.mean(pos_no)), 4),
          "pairAUC_pos", round(C.auc(pair_y, pair_s), 4))
    print("C2_sesspos_ev", round(float(np.mean(spev)), 4), "sesspos_noise", round(float(np.mean(spno)), 4),
          "lastsess_frac_ev", round(float(np.mean(lastsess_ev)), 4), "noise", round(float(np.mean(lastsess_no)), 4))

    # first-mention axis
    MONTHS = set("january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday session".split())
    STARTER = set("i the we it but and so then my he she they you that this what when where how why did do have has was were after before well oh yeah yes no okay alright anyway wow man hey hi hello thanks thank good great nice cool really actually probably maybe listen look lets let just know think".split())
    speaker_names = set()
    for m in C.MIDS:
        if C.SPK_OF_MID.get(m) not in (None, "?", ""):
            speaker_names.add(C.SPK_OF_MID[m].lower())
    def ents(text):
        toks = re.findall(r"[A-Za-z]{3,}", text)
        out = set()
        for t in toks[1:]:
            if t[0].isupper() and t.lower() not in STARTER and t.lower() not in MONTHS and t.lower() not in speaker_names:
                out.add(t.lower())
        return out
    first_rows, rep_rows = [], []
    for k, lst in conv_raws.items():
        seen = set()
        for t, row in lst:
            es = ents(C.RAWTEXT_OF_MID.get(C.MIDS[row], ""))
            new = es - seen
            seen |= es
            (first_rows if new else rep_rows).append(row)
    fr, rr_ = np.array(first_rows), np.array(rep_rows)
    print("C3_first_n", len(fr), "rep_n", len(rr_))
    ax_first = C.l2n(C.V[fr].mean(0) - C.V[rr_].mean(0))
    y = np.concatenate([np.ones(len(fr)), np.zeros(len(rr_))])
    sc = np.concatenate([C.V[fr] @ ax_first, C.V[rr_] @ ax_first])
    # 5-fold by conversation
    conv_of = [C.CONV_OF_MID[C.MIDS[r]] for r in np.concatenate([fr, rr_])]
    fl = C.folds_by_conv(conv_of, 5)
    acs = []
    for fi in range(5):
        te = fl[fi]
        acs.append(C.auc(y[te], sc[te]))
    print("C4_firstaxis_cvAUC", [round(a, 3) for a in acs], "mean", round(float(np.nanmean(acs)), 4),
          "sep", round(float((C.V[fr] @ ax_first).mean() - (C.V[rr_] @ ax_first).mean()), 4))
    # enrichment: P(first|ev) vs P(first|noise) and delta projection
    fev, fno, dy, ds = [], [], [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None:
            continue
        evp = [r for r in t50 if r in ev]
        if not evp:
            continue
        fset = set(fr.tolist())
        fev.append(np.mean([1 if r in fset else 0 for r in evp]))
        fno.append(np.mean([1 if int(r) in fset else 0 for r in noise]))
        for r in evp:
            dy.append(1); ds.append(float(C.V[int(r)] @ ax_first))
        for r in noise[:10]:
            dy.append(0); ds.append(float(C.V[int(r)] @ ax_first))
    print("C5_P_first|ev", round(float(np.mean(fev)), 4), "P_first|noise", round(float(np.mean(fno)), 4),
          "evnoise_pairAUC_axisfirst", round(C.auc(dy, ds), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
