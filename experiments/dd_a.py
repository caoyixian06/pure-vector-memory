# dd_a: rerank residual axis - anatomy + rescue test (v2, position/row fix)
import dd_common as C
import numpy as np, re, traceback

try:
    evalq = []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev:
            continue
        evp = np.array([r for r in t50 if r in ev], dtype=int)
        if len(evp) == 0 or len(noise) == 0:
            continue
        order = np.argsort(-rer)          # positions in pool
        inv = np.empty(len(order), dtype=int)
        inv[order] = np.arange(len(order))
        pos50 = {int(r): p for p, r in enumerate(t50)}
        best_ev_rank = min(int(inv[pos50[int(k)]]) for k in evp)
        top1_row = int(t50[int(order[0])])
        evalq.append({"i": i, "conv": qm["conv"], "t50": t50, "evp": evp, "noise": noise,
                      "rer": rer, "pos50": pos50, "best_ev_rank": best_ev_rank,
                      "top1_row": top1_row, "top1_is_ev": bool(top1_row in set(evp.tolist()))})
    n = len(evalq)
    fails = [e for e in evalq if not e["top1_is_ev"]]
    print("A1_n", n, "ev@1_rerank", round(sum(e["top1_is_ev"] for e in evalq) / n, 4), "n_fail", len(fails))
    print("A2_med_bestev_rank_all", float(np.median([e["best_ev_rank"] for e in evalq])),
          "fail_med", float(np.median([e["best_ev_rank"] for e in fails])))

    STOP = set("what where when who why how which did do does was were is are the and you your for with that this have has had about tell know".split())
    def qtoks(i):
        return set(re.findall(r"[a-z0-9]{3,}", str(C.QMETA[i]["q"]).lower())) - STOP
    def rtoks(m):
        return set(re.findall(r"[a-z0-9]{3,}", C.RAWTEXT_OF_MID.get(C.MIDS[m], "").lower()))
    ov_e, ov_n, evq_frac, noq_frac = [], [], 0, 0
    for e in fails:
        qt = qtoks(e["i"])
        p50 = e["pos50"]
        bev = e["evp"][np.argmax([e["rer"][p50[int(k)]] for k in e["evp"]])]
        tn = e["noise"][np.argmax([e["rer"][p50[int(k)]] for k in e["noise"]])]
        ov_e.append(len(qt & rtoks(bev)) / max(len(qt), 1))
        ov_n.append(len(qt & rtoks(tn)) / max(len(qt), 1))
        if C.RAWTEXT_OF_MID.get(C.MIDS[bev], "").rstrip().endswith("?"): evq_frac += 1
        if C.RAWTEXT_OF_MID.get(C.MIDS[tn], "").rstrip().endswith("?"): noq_frac += 1
    print("A3_fail_anatomy qtokov_ev", round(float(np.mean(ov_e)), 3), "qtokov_topnoise", round(float(np.mean(ov_n)), 3),
          "ev_is_question_frac", round(evq_frac / max(len(fails), 1), 3), "noise_is_question_frac", round(noq_frac / max(len(fails), 1), 3))

    convs = [e["conv"] for e in fails]
    fl = C.folds_by_conv(convs, 5)
    aucs_fail, resc_te = [], []
    for fi in range(5):
        te = [fails[x] for x in fl[fi]]
        tr = [fails[x] for x in np.concatenate([fl[x] for x in range(5) if x != fi])]
        if not tr or not te:
            continue
        D = []
        for e in tr:
            p50 = e["pos50"]
            tn = e["noise"][np.argmax([e["rer"][p50[int(k)]] for k in e["noise"]])]
            D.append(C.l2n(C.V[list(e["evp"])].mean(0)) - C.l2n(C.V[int(tn)]))
        ax = C.l2n(np.mean(D, axis=0))
        y, s = [], []
        for e in te:
            for r in e["evp"]:
                y.append(1); s.append(float(C.V[int(r)] @ ax))
            for r in e["noise"][:10]:
                y.append(0); s.append(float(C.V[int(r)] @ ax))
        aucs_fail.append(C.auc(y, s))

        def ev1(lam, pool):
            hit = 0
            for e in pool:
                sc = e["rer"] + lam * (C.V[e["t50"]] @ ax)
                if int(e["t50"][int(np.argmax(sc))]) in set(e["evp"].tolist()):
                    hit += 1
            return hit / len(pool)
        best_lam, best_v = 0.0, -1
        for lam in np.arange(0.05, 0.55, 0.05):
            v = ev1(lam, tr)
            if v > best_v:
                best_v, best_lam = v, lam
        resc_te.append((best_lam, ev1(0.0, te), ev1(best_lam, te)))
    print("A4_residaxis_auc_fail_cv", [round(a, 3) for a in aucs_fail], "mean", round(float(np.nanmean(aucs_fail)), 4))
    print("A5_rescue_test ev@1_before", round(float(np.mean([x[1] for x in resc_te])), 4),
          "after", round(float(np.mean([x[2] for x in resc_te])), 4),
          "lams", [round(x[0], 2) for x in resc_te])

    D = []
    for e in fails:
        p50 = e["pos50"]
        tn = e["noise"][np.argmax([e["rer"][p50[int(k)]] for k in e["noise"]])]
        D.append(C.l2n(C.V[list(e["evp"])].mean(0)) - C.l2n(C.V[int(tn)]))
    axg = C.l2n(np.mean(D, axis=0))
    suc = [e for e in evalq if e["top1_is_ev"]]
    y, s = [], []
    for e in suc:
        for r in e["evp"]:
            y.append(1); s.append(float(C.V[int(r)] @ axg))
        for r in e["noise"][:10]:
            y.append(0); s.append(float(C.V[int(r)] @ axg))
    print("A6_axis_on_succeeded_auc", round(C.auc(y, s), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
