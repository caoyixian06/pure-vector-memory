# dd_g: multi-hop / cross-session query signature
import dd_common as C
import numpy as np, traceback

try:
    CENT = C.l2n(C.V.mean(0))
    feats, lab_s, lab_m, idxs = [], [], [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not qm["ev_rows"]:
            continue
        s = np.sort(C.S[i][t50])[::-1]
        smax = float(C.S[i].max())
        s10 = float(s[:10].mean())
        gap12 = float(s[0] - s[1]) if len(s) > 1 else 0.0
        e = np.exp(s * 20 - (s * 20).max()); p = e / e.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        cdev = float(np.linalg.norm(C.QEMB[i] - CENT))
        vexm = float(C.VEX[j][t50][:10].mean())
        sbt = set(np.argsort(-C.SB[j])[:10].tolist())
        sqt = set(np.argsort(-C.SQ[j])[:10].tolist())
        jac = len(sbt & sqt) / max(len(sbt | sqt), 1)
        feats.append([smax, s10, gap12, ent, cdev, vexm, jac])
        lab_s.append(1 if qm["ev_nsess"] >= 2 else 0)
        lab_m.append(1 if qm["n_evmsg"] >= 2 else 0)
        idxs.append(i)
    X = np.array(feats); ys = np.array(lab_s); ym = np.array(lab_m)
    print("G1_n", len(X), "cross_session_frac", round(float(ys.mean()), 4), "multiev_frac", round(float(ym.mean()), 4))
    names = ["smax", "s10mean", "gap12", "entropy", "centroid_dev", "vex_top10", "sb_sq_jac10"]
    for k, nm in enumerate(names):
        print("G2_feat", nm, "AUC_crosssess", round(C.auc(ys, X[:, k]), 4), "AUC_multiev", round(C.auc(ym, X[:, k]), 4))
    # sign-corrected combined CV
    for tgt, tag in [(ys, "crosssess"), (ym, "multiev")]:
        Xs = X.copy()
        for k in range(X.shape[1]):
            if C.auc(tgt, X[:, k]) < 0.5:
                Xs[:, k] = -X[:, k]
        a, m_ = C.logit_auc(Xs, tgt)
        print("G3_logitcv", tag, [round(x, 3) for x in a], "mean", round(m_, 4))
    # top50 evidence coverage for mh vs single
    cov_mh, cov_sg = [], []
    for k, i in enumerate(idxs):
        j, ev, t50, noise, rer = C.pools(i)
        cov = 1.0 if set(C.QMETA[i]["ev_rows"]) <= set(t50.tolist()) else 0.0
        (cov_mh if (ys[k] == 1 or ym[k] == 1) else cov_sg).append(cov)
    print("G4_ev_in_top50 multi", round(float(np.mean(cov_mh)), 4), "single", round(float(np.mean(cov_sg)), 4))
    # quartile risk table (combined score from full fit is leaky; use entropy quartile)
    q = np.quantile(X[:, 3], [0.25, 0.5, 0.75])
    b = np.digitize(X[:, 3], q)
    print("G5_crosssess_rate_by_entropy_quartile", [round(float(ys[b == k].mean()), 4) for k in range(4)])
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
