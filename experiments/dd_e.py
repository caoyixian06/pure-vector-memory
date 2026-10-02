# dd_e: word-anchor axis zoo vs delta and ev-noise discrimination
import dd_common as C
import numpy as np, re, traceback

try:
    from FlagEmbedding import BGEM3FlagModel
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    def ax_of(pos, neg):
        a = np.asarray(bge.encode(pos)["dense_vecs"], dtype=np.float32).mean(0)
        b = np.asarray(bge.encode(neg)["dense_vecs"], dtype=np.float32).mean(0)
        return C.l2n(a - b)

    SETS = {
        "narrative": (["story","told","recall","remember","happened","event","trip","visit","met","mentioned","described","shared","talked","memory","past","experience","childhood","school","friend","family","journey","moved","said","years"],
                       ["hello","hi","hey","greetings","morning","goodbye","bye","thanks","thank","welcome","fine","okay","sure","right","yeah","oh"]),
        "negation": (["not","never","no","dont","didnt","cant","wont","nothing","none","isnt","wasnt","doesnt"],
                      ["always","yes","sure","definitely","certainly","absolutely","really","truly","indeed","maybe","probably","must"]),
        "interrog": (["what","where","when","who","why","how","which","whose"],
                      ["i","you","we","they","he","she","it","that","this"]),
        "emopos_neg": (["happy","love","great","awesome","excited","glad","wonderful","joy","fun","amazing"],
                        ["sad","angry","hate","worried","upset","terrible","awful","fear","cry","lonely"]),
        "time_place": (["yesterday","today","tomorrow","week","month","year","ago","last","next","soon","now"],
                        ["school","home","hospital","office","restaurant","park","church","store","room","house"]),
        "family_work": (["mother","father","sister","brother","wife","husband","son","daughter","grandma","grandpa","friend"],
                         ["job","work","meeting","boss","office","project","company","salary","interview","career"]),
        "plan_past": (["will","going","plan","want","hope","soon","future","tomorrow","dream","goal"],
                       ["was","did","had","went","ago","yesterday","last","remembered","used","before"]),
        "health": (["sick","doctor","hospital","medicine","pain","healthy","treatment","surgery","nurse","clinic"],
                    ["game","music","movie","sport","vacation","party","hobby","book","travel","concert"]),
        "money": (["money","dollar","pay","buy","cost","expensive","cheap","price","rent","budget"],
                   ["rain","sun","snow","wind","weather","hot","cold","storm","cloud","sky"]),
        "school": (["school","class","teacher","student","homework","exam","college","study","learn","test"],
                    ["trip","travel","flight","hotel","vacation","beach","tour","country","abroad","passport"]),
    }
    axes = {k: ax_of(p, n) for k, (p, n) in SETS.items()}

    # delta build
    Dl, yl, sl, convs = [], [], [], []
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None or len(noise) == 0:
            continue
        Dl.append(C.l2n(C.V[qm["ev_rows"]].mean(0)) - C.l2n(C.V[noise].mean(0)))
        convs.append(qm["conv"])
        evp = [r for r in t50 if r in ev]
        for r in evp:
            yl.append(1); sl.append(i)
        for r in noise[:10]:
            yl.append(0); sl.append(i)
    Dl = np.array(Dl)
    qidx = np.array(sl); ylab = np.array(yl)

    # residual from top-3 PCA (same recipe as dd_b)
    Dc = Dl - Dl.mean(0)
    U, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    P3 = (Dc @ Vt[:3].T) @ Vt[:3]
    Rc = Dc - P3

    print("E1_axis_table: axis | poolAUC | mean_cos_delta | corr_PC4resid | corr_PC5resid")
    fl = C.folds_by_conv(convs, 5)
    for k, ax in axes.items():
        # pool pair AUC (full, sign as-is)
        sc = C.V[qidx] @ ax
        a_full = C.auc(ylab, sc)
        # CV by conv to be safe
        acs = []
        for fi in range(5):
            te = fl[fi]
            acs.append(C.auc(ylab[te], sc[te]))
        cd = C.l2n(Dl) @ ax
        cr = (Rc @ ax)
        pc4 = Rc @ Vt[3]; pc5 = Rc @ Vt[4]
        c4 = float(np.corrcoef(cr, pc4)[0, 1])
        c5 = float(np.corrcoef(cr, pc5)[0, 1])
        print("E1", k, "auc_cv", round(float(np.nanmean(acs)), 4), "auc_full", round(a_full, 4),
              "cos_delta", round(float(cd.mean()), 4), "corrPC4", round(c4, 3), "corrPC5", round(c5, 3))
    # do axes span the residual? R2 of residual on axis subspace (10 axes)
    A = np.stack([axes[k] for k in axes]).T  # 1024 x 10
    Qm, _ = np.linalg.qr(A)
    proj = (Rc @ Qm) @ Qm.T
    print("E2_resid_energy_in_10axis_span", round(float((np.linalg.norm(proj, axis=1) ** 2).sum() / (np.linalg.norm(Rc, axis=1) ** 2).sum()), 4))
    proj_all = (Dc @ Qm) @ Qm.T
    print("E3_delta_energy_in_10axis_span", round(float((np.linalg.norm(proj_all, axis=1) ** 2).sum() / (np.linalg.norm(Dc, axis=1) ** 2).sum()), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
