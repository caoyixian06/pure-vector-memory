# dd_f2: is the compression-axis gain just a raw-kind prior? decompose + rerank-stage test
import dd_common as C
import numpy as np, traceback

try:
    P = C.TWIN_PAIRS
    diffs = C.V[P[:, 0]] - C.V[P[:, 1]]
    ax_c = C.l2n(diffs.mean(0))
    g = C.V @ ax_c
    ISRAW = np.array([1.0 if C.KIND_OF_MID[m] == "raw" else 0.0 for m in C.MIDS])
    eval_idx = [i for i, qm in enumerate(C.QMETA) if C.QROW[i] >= 0 and qm["ev_rows"]]

    def evk_mat(Sm, k):
        hits = 0
        for a in range(Sm.shape[0]):
            row = np.argpartition(-Sm[a], k)[:k]
            if set(row.tolist()) & set(C.QMETA[eval_idx[a]]["ev_rows"]):
                hits += 1
        return hits / Sm.shape[0]

    Se = C.S[eval_idx]
    print("F2_1_baseline ev@5", round(evk_mat(Se, 5), 4), "ev@50", round(evk_mat(Se, 50), 4))
    # beta sweep (no CV, trend only)
    for b in [0.2, 0.3, 0.5, 0.8, 1.2]:
        cn = np.linalg.norm(C.QEMB[eval_idx] + b * ax_c[None, :], axis=1, keepdims=True)
        Sp = (Se + b * g[None, :]) / cn
        print("F2_2_beta", b, "ev@5", round(evk_mat(Sp, 5), 4), "ev@50", round(evk_mat(Sp, 50), 4))
    # binary raw prior
    for d in [0.01, 0.02, 0.05]:
        Sp = Se + d * ISRAW[None, :]
        print("F2_3_rawprior", d, "ev@5", round(evk_mat(Sp, 5), 4), "ev@50", round(evk_mat(Sp, 50), 4))
    # oracle kind prior: raw-only pool
    rawrows = np.where(ISRAW > 0)[0]
    hits5 = 0
    for a, i in enumerate(eval_idx):
        row = np.argsort(-Se[a][rawrows])[:5]
        if set(rawrows[row].tolist()) & set(C.QMETA[i]["ev_rows"]):
            hits5 += 1
    print("F2_4_rawonly ev@5", round(hits5 / len(eval_idx), 4))
    # does axis_c add BEYOND kind prior? residualize: g_res = g - proj on ISRAW
    gr = g - ISRAW * (ISRAW @ g) / (ISRAW @ ISRAW)
    for b in [0.2, 0.5]:
        Sp = (Se + b * gr[None, :]) / np.linalg.norm(np.ones_like(Se) * 1.0, axis=1, keepdims=True)  # treat as additive score
        print("F2_5_axisresid_beta", b, "ev@5", round(evk_mat(Se + b * gr[None, :], 5), 4))

    # rerank-stage test: within fused top50, sc = RER + lam * z(proj or israw)
    for tag, feat in [("axc", None), ("israw", None)]:
        lams = [0.05, 0.1, 0.2, 0.3, 0.5]
        base1, after1 = [], []
        fl = C.folds_by_conv([C.QMETA[i]["conv"] for i in eval_idx], 5)
        for fi in range(5):
            te = [eval_idx[x] for x in fl[fi]]
            tr = [eval_idx[x] for x in np.concatenate([fl[x] for x in range(5) if x != fi])]
            def ev1(idx, lam):
                hit = 0
                for i in idx:
                    j = C.QROW[i]
                    t50 = C.TOP50[j]
                    f = g[t50] if tag == "axc" else ISRAW[t50]
                    f = (f - f.mean()) / (f.std() + 1e-9)
                    sc = C.RER[j] + lam * f
                    if int(t50[np.argmax(sc)]) in set(C.QMETA[i]["ev_rows"]):
                        hit += 1
                return hit / len(idx)
            bl = max(lams, key=lambda L: ev1(tr, L))
            base1.append(ev1(te, 0.0)); after1.append(ev1(te, bl))
        print("F2_6_rerankboost", tag, "ev@1_before", round(float(np.mean(base1)), 4),
              "after", round(float(np.mean(after1)), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
