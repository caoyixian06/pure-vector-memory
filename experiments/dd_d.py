# dd_d: turn-taking / response direction
import dd_common as C
import numpy as np, traceback
from collections import defaultdict

try:
    conv_raws = defaultdict(list)
    for m, row in C.MID2ROW.items():
        if C.KIND_OF_MID[m] == "raw":
            conv_raws[C.CONV_OF_MID[m]].append((C.TURN_OF_MID[m], row, C.SPK_OF_MID.get(m, "?")))
    for k in conv_raws:
        conv_raws[k].sort()

    diffs, qmask = [], []   # consecutive different-speaker diffs
    spk_axis = {}
    for k, lst in conv_raws.items():
        for a in range(len(lst) - 1):
            (t1, r1, s1), (t2, r2, s2) = lst[a], lst[a + 1]
            if s1 != s2 and t2 == t1 + 1:
                d = C.V[r2] - C.V[r1]
                diffs.append(d)
                qmask.append(1 if "?" in C.RAWTEXT_OF_MID.get(C.MIDS[r1], "") else 0)
        # speaker axis per conv computed later
    D = np.array(diffs)
    ax_d = C.l2n(D.mean(0))
    print("D1_n_pairs", len(D))
    cosd = D @ ax_d
    print("D2_mean_cos_pairaxis", round(float(cosd.mean()), 4), "frac_pos", round(float((cosd > 0).mean()), 4),
          "rand_baseline_abs", round(1 / np.sqrt(1024), 4))

    # random-pair baseline: same conv, different speaker, non-adjacent
    rng = np.random.RandomState(0)
    rnd = []
    for k, lst in conv_raws.items():
        if len(lst) < 4:
            continue
        for _ in range(30):
            a, b = rng.choice(len(lst), 2, replace=False)
            if lst[a][2] != lst[b][2]:
                rnd.append(C.V[lst[b][1]] - C.V[lst[a][1]])
    rnd = np.array(rnd)
    print("D3_randpair_mean_cos", round(float((rnd @ ax_d).mean()), 4), "randpair_mean_norm", round(float(np.linalg.norm(rnd, axis=1).mean()), 4),
          "adjpair_mean_norm", round(float(np.linalg.norm(D, axis=1).mean()), 4))

    # question -> response alignment
    qm = np.array(qmask, dtype=bool)
    print("D4_qresp_mean_cos", round(float(cosd[qm].mean()), 4), "stmtresp_mean_cos", round(float(cosd[~qm].mean()), 4),
          "AUC_q_vs_stmt_alignment", round(C.auc(qm.astype(float), np.abs(cosd)), 4), "n_q", int(qm.sum()))

    # is response direction just the speaker axis?
    spk_ax_cos, spk_axes = [], {}
    for k, lst in conv_raws.items():
        spks = sorted(set(s for _, _, s in lst))
        if len(spks) != 2:
            continue
        va = C.V[[r for _, r, s in lst if s == spks[0]]].mean(0)
        vb = C.V[[r for _, r, s in lst if s == spks[1]]].mean(0)
        axa = C.l2n(vb - va)
        spk_axes[k] = axa
        spk_ax_cos.append(float(axa @ ax_d))
    print("D5_cos(axis_d,spk_axis)_mean", round(float(np.mean(spk_ax_cos)), 4), "per_conv", [round(x, 3) for x in spk_ax_cos])

    # evidence vs noise along axis_d
    y, s = [], []
    for i, qmeta in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None:
            continue
        evp = [r for r in t50 if r in ev]
        for r in evp:
            y.append(1); s.append(float(C.V[int(r)] @ ax_d))
        for r in noise[:10]:
            y.append(0); s.append(float(C.V[int(r)] @ ax_d))
    print("D6_evnoise_pairAUC_axisd", round(C.auc(y, s), 4), "n_pairs", len(y))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
