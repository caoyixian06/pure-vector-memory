# dd_f: twin compression axis (raw <-> summary same turn)
import dd_common as C
import numpy as np, traceback

try:
    P = C.TWIN_PAIRS  # (n,2) raw_row, sum_row
    diffs = C.V[P[:, 0]] - C.V[P[:, 1]]  # raw - summary
    ax_c = C.l2n(diffs.mean(0))
    cosd = diffs @ ax_c
    print("F1_n_pairs", len(P), "mean_cos(diff,axis)", round(float(cosd.mean()), 4),
          "frac_pos", round(float((cosd > 0).mean()), 4), "rand_baseline", round(1 / np.sqrt(1024), 4))
    rnd_norm = float(np.linalg.norm(C.V[P[:, 0]] - C.V[np.random.RandomState(1).permutation(P[:, 1])], axis=1).mean())
    print("F2_twin_diff_norm", round(float(np.linalg.norm(diffs, axis=1).mean()), 4),
          "rand_rawsum_diff_norm", round(rnd_norm, 4))
    print("F3_mean_raw_on_axis", round(float((C.V[P[:, 0]] @ ax_c).mean()), 4),
          "mean_sum_on_axis", round(float((C.V[P[:, 1]] @ ax_c).mean()), 4))

    # ev vs noise pool pairs along axis_c
    y, s = [], []
    qkind = []  # 1 if question has raw-kind evidence, 0 if summary-kind evidence
    qcos = []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None:
            continue
        evp = [r for r in t50 if r in ev]
        for r in evp:
            y.append(1); s.append(float(C.V[int(r)] @ ax_c))
        for r in noise[:10]:
            y.append(0); s.append(float(C.V[int(r)] @ ax_c))
        kinds = [C.KIND_OF_MID[m] for m in qm["ev_mids"]]
        qkind.append(1.0 if "raw" in kinds else 0.0)
        qcos.append(float(C.QEMB[i] @ ax_c))
    print("F4_evnoise_pairAUC_axisc", round(C.auc(y, s), 4))
    qkind = np.array(qkind); qcos = np.array(qcos)
    print("F5_routing mean_cos(q,ax)|ev_raw", round(float(qcos[qkind == 1].mean()), 4),
          "|ev_summary", round(float(qcos[qkind == 0].mean()), 4),
          "AUC", round(C.auc(qkind, qcos), 4), "n_raw", int(qkind.sum()), "n_sum", int((1 - qkind).sum()))

    # retrieval surgery: q' = l2n(q + b*ax_c); fast via g = V @ ax_c
    g = C.V @ ax_c  # 11753
    convs = [qm["conv"] for qm in C.QMETA]
    eval_idx = [i for i, qm in enumerate(C.QMETA)
                if C.QROW[i] >= 0 and qm["ev_rows"]]
    cv = [C.QMETA[i]["conv"] for i in eval_idx]
    fl = C.folds_by_conv(cv, 5)

    def evk(Smat, qlist, k):
        # Smat rows aligned with qlist (global question indices)
        hit = 0
        for a, i in enumerate(qlist):
            row = np.argsort(-Smat[a])[:k]
            if set(row.tolist()) & set(C.QMETA[i]["ev_rows"]):
                hit += 1
        return hit / len(qlist)

    base5 = evk(C.S, eval_idx, 5)
    print("F6_baseline ev@5_fullpool", round(base5, 4))
    gains = []
    for fi in range(5):
        te = [eval_idx[x] for x in fl[fi]]
        tr = [eval_idx[x] for x in np.concatenate([fl[x] for x in range(5) if x != fi])]
        best, bv = 0.0, -1
        for b in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
            Sp = (C.S[tr] + b * g[None, :]) / np.linalg.norm(C.QEMB[tr] + b * ax_c[None, :], axis=1, keepdims=True)
            v = evk(Sp, tr, 5)
            if v > bv:
                bv, best = v, b
        Sp = (C.S[te] + best * g[None, :]) / np.linalg.norm(C.QEMB[te] + best * ax_c[None, :], axis=1, keepdims=True)
        gains.append(evk(Sp, te, 5) - evk(C.S[te], te, 5))
        if fi == 0:
            print("F7_fold0_best_beta", best)
    print("F8_surgery_delta_ev@5_per_fold", [round(x, 4) for x in gains], "mean", round(float(np.mean(gains)), 4))

    # relation to delta
    E, Nn = [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None or len(noise) == 0:
            continue
        E.append(C.l2n(C.V[qm["ev_rows"]].mean(0)))
        Nn.append(C.l2n(C.V[noise].mean(0)))
    Dl = C.l2n(np.array(E) - np.array(Nn))
    proj = Dl @ ax_c
    r2 = float(np.var(proj) / np.var(np.linalg.norm(Dl, axis=1) ** 2) ) if False else float((proj ** 2).sum() / (Dl ** 2).sum())
    print("F9_delta_var_explained_by_axisc", round(r2, 4), "mean_cos(delta,axc)", round(float(proj.mean()), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
