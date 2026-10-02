# dd_h: retrieval confidence vs final correctness (llm_score)
import dd_common as C
import numpy as np, traceback

try:
    feats, wrong, cats, evrank = [], [], [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or qm["score"] is None:
            continue
        r = np.sort(rer)[::-1]
        rgap = float(r[0] - r[1]) if len(r) > 1 else 0.0
        e = np.exp(rer - rer.max()); p = e / e.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        s = np.sort(C.S[i][t50])[::-1]
        order = np.argsort(-rer)
        inv = np.empty(len(order), dtype=int)
        inv[order] = np.arange(len(order))
        pos50 = {int(r): p for p, r in enumerate(t50)}
        evp = [int(inv[pos50[k]]) for k in t50 if k in ev]
        bev = min(evp) if evp else 51
        vexm = float(C.VEX[j][t50][:10].mean())
        smax = float(C.S[i].max())
        feats.append([float(r[0]), rgap, ent, float(s[0]), float(s[0] - (s[1] if len(s) > 1 else 0)), vexm, smax, bev])
        wrong.append(1.0 - float(qm["score"]))
        cats.append(qm["cat"])
        evrank.append(bev)
    X = np.array(feats); y = np.array(wrong)
    names = ["rer_top1", "rer_gap", "rer_entropy", "sb_top1", "sb_gap", "vex_top10", "smax_full", "best_ev_rank"]
    print("H1_n", len(X), "wrong_rate", round(float(y.mean()), 4))
    for k, nm in enumerate(names):
        a = C.auc(y, X[:, k])
        print("H2_feat", nm, "AUC_wrong(sign-fixed)", round(max(a, 1 - a), 4))
    a, m_ = C.logit_auc(X, y)
    print("H3_logitcv_wrong", [round(x, 3) for x in a], "mean", round(m_, 4))
    # confidence quartiles from rer_gap + rer_top1 (simple, pipeline-available)
    conf = X[:, 0] + 2.0 * X[:, 1] - 0.02 * X[:, 7]
    q = np.quantile(conf, [0.25, 0.5, 0.75])
    b = np.digitize(conf, q)
    print("H4_wrong_rate_by_conf_quartile", [round(float(y[b == k].mean()), 4) for k in range(4)])
    # per category
    for c in sorted(set(cats)):
        msk = np.array([x == c for x in cats])
        if msk.sum() < 30:
            continue
        aa = C.auc(y[msk], conf[msk])
        print("H5_cat", c, "n", int(msk.sum()), "wrong_rate", round(float(y[msk].mean()), 4), "confAUC", round(max(aa, 1 - aa), 4))
    # headroom: among wrong, how many had evidence outside pool or ranked >1
    bev = X[:, 7]
    print("H6_among_wrong ev_not_top1_frac", round(float((y * (bev > 0)).sum() / y.sum()), 4),
          "ev_not_in_pool_frac", round(float((y * (bev > 50)).sum() / y.sum()), 4))
    # upper bound: correct-with-perfect-rerank proxy = questions with ev in pool
    print("H7_wrong_with_ev_in_pool", int((y * (bev <= 50) * (bev > 0)).sum()), "of", int(y.sum()))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
