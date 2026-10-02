# dd_b: per-question delta PCA, regression on known axes, residual vs lexical features
import dd_common as C
import numpy as np, re, traceback

try:
    # ---- build per-question delta
    Dl, evtexts, notexts, evrows_all, posv, fmflag = [], [], [], [], [], []
    from collections import defaultdict
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
    first_of_conv = {}
    for k, lst in conv_raws.items():
        seen = set()
        fs = set()
        for t, row in lst:
            es = ents(C.RAWTEXT_OF_MID.get(C.MIDS[row], ""))
            if es - seen:
                fs.add(row)
            seen |= es
        first_of_conv[k] = fs
    for i, qm in enumerate(C.QMETA):
        j, ev, t50, noise, rer = C.pools(i)
        if j < 0 or not ev or noise is None or len(noise) == 0:
            continue
        Dl.append(C.l2n(C.V[qm["ev_rows"]].mean(0)) - C.l2n(C.V[noise].mean(0)))
        evtexts.append(" ".join(C.RAWTEXT_OF_MID[m] for m in qm["ev_mids"]))
        notexts.append(" ".join(C.RAWTEXT_OF_MID.get(C.MIDS[int(r)], "") for r in noise[:8]))
        nconv = len(C.HDR_TURNS.get(qm["conv"], [1]))
        turns = [C.TURN_OF_MID[m] for m in qm["ev_mids"]]
        posv.append(np.mean(turns) / max(max(t for _, t in conv_raws[qm["conv"]]), 1))
        fmflag.append(np.mean([1 if r in first_of_conv[qm["conv"]] else 0 for r in qm["ev_rows"]]))
    Dl = np.array(Dl)
    print("B1_n_delta", Dl.shape)
    # ---- PCA
    mu = Dl.mean(0)
    Dc = Dl - mu
    U, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    print("B2_var_ratio_top6", [round(float(x), 4) for x in var[:6]])
    P = Dc @ Vt[:6].T  # PC scores n x 6
    # ---- known axes via bge words
    from FlagEmbedding import BGEM3FlagModel
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    def wemb(ws):
        return C.l2n(np.asarray(bge.encode(ws)["dense_vecs"], dtype=np.float32).mean(0))
    narr = wemb("story told recall remember happened event trip visit met mentioned described shared talked memory past experience childhood school friend family journey accident moved".split()) - \
           wemb("hello hi hey greetings morning goodbye bye thanks thank welcome fine okay".split())
    date_ax = wemb("yesterday today tomorrow monday tuesday wednesday thursday friday saturday sunday january february march april june july august september october november december week month year day ago last next".split()) - \
              wemb("table chair window car book street house door room water food money city picture letter box".split())
    narr = C.l2n(narr); date_ax = C.l2n(date_ax)
    cnarr = C.l2n(Dl) @ narr
    cdate = C.l2n(Dl) @ date_ax
    lenf = np.array([len(t) for t in evtexts], dtype=float) - np.array([len(t) for t in notexts], dtype=float)
    dens = []
    for t in evtexts:
        tk = re.findall(r"[a-z']{2,}", t.lower())
        dens.append(len(set(tk)) / max(len(tk), 1))
    densf = np.array(dens)
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    Xd = np.stack([z(cnarr), z(cdate), z(lenf), z(densf)], axis=1)
    print("B3_corr_knownaxes_vs_PC123")
    for pc in range(3):
        row = []
        for k in range(4):
            row.append(round(float(np.corrcoef(Xd[:, k], P[:, pc])[0, 1]), 3))
        # R2 of PC ~ Xd
        A = np.concatenate([Xd, np.ones((len(Xd), 1))], axis=1)
        beta, res, *_ = np.linalg.lstsq(A, P[:, pc], rcond=None)
        pred = A @ beta
        r2 = 1 - ((P[:, pc] - pred) ** 2).sum() / ((P[:, pc] - P[:, pc].mean()) ** 2).sum()
        print("B3_PC", pc + 1, "corrs_narr_date_len_dens", row, "R2", round(float(r2), 4))
    # ---- residual and lexical features
    recon = (P[:, :3] @ Vt[:3]) + mu
    R = Dl - recon
    # lexical contrast features
    NEG = set("not never no dont didnt cant wont nothing none isnt wasnt doesnt nor".split())
    PRON = set("i you we my me us our your his her they them".split())
    def lex(t, mode):
        tk = re.findall(r"[a-zA-Z']+", t)
        lw = [w.lower() for w in tk]
        n = max(len(lw), 1)
        if mode == "neg": return sum(w in NEG for w in lw) / n
        if mode == "pron": return sum(w in PRON for w in lw) / n
        if mode == "past": return sum(w.endswith("ed") and len(w) > 4 for w in lw) / n
        if mode == "propn":
            words = t.split()
            return sum(1 for k2, w in enumerate(words) if k2 > 0 and w[:1].isupper() and w[1:2].islower()) / n
        if mode == "num": return sum(any(ch.isdigit() for ch in w) for w in lw) / n
        if mode == "quest": return 1.0 if "?" in t else 0.0
        if mode == "excl": return 1.0 if "!" in t else 0.0
    modes = ["neg", "pron", "past", "propn", "num", "quest", "excl"]
    feats = {}
    for md in modes:
        feats[md] = np.array([lex(a, md) - lex(b, md) for a, b in zip(evtexts, notexts)])
    feats["posconv"] = np.array(posv)
    feats["firstmention"] = np.array(fmflag)
    Rc = R - R.mean(0)
    P6 = Rc @ Vt[:6].T  # residual pc scores (approx: project residual on same dirs)
    print("B4_residual_vs_lexical (corr, |r|>0.12 flagged)")
    for md, fv in feats.items():
        cs = [round(float(np.corrcoef(fv, P6[:, pc])[0, 1]), 3) for pc in range(3, 6)]
        cn = round(float(np.corrcoef(fv, np.linalg.norm(Rc, axis=1))[0, 1]), 3)
        flag = "*" if max(abs(x) for x in cs + [cn]) > 0.12 else ""
        print("B4", md, "PC4_5_6", cs, "normR", cn, flag)
    # residual energy fraction
    print("B5_resid_frac_energy", round(float((np.linalg.norm(Rc, axis=1) ** 2).sum() / (np.linalg.norm(Dc, axis=1) ** 2).sum()), 4))
    print("DONE")
except Exception:
    traceback.print_exc()
    print("FAIL")
