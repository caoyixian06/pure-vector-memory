CONFIGS = [
    ("v14_256dom_pool", 5, 500, 40),
    
    
    
]
RES = {c[0]: [0, 0] for c in CONFIGS}
FORD11 = {}
for hold in FOLDS:
    for name, trunc, nest, nnoise in CONFIGS:
        trF, trY, trG = [], [], []
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] == hold:
                continue
            pool = POOLA[k_i]
            G = set()
            for grp in GSETS[k_i]:
                G |= grp
            gold_rows = [rr for rr, c in enumerate(pool) if c in G][:8]
            gset = set(gold_rows)
            noise_rows = [rr for rr, c in enumerate(pool) if c not in G]
            noisepick = list(np.random.RandomState(k_i + 31).permutation(noise_rows)[:nnoise]) if noise_rows else []
            rows = gold_rows + noisepick
            for rr in rows:
                trF.append(FEATS[k_i][rr])
                trY.append(1 if rr in gset else 0)
            trG.append(len(rows))
        ranker = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=nest, learning_rate=0.08,
            num_leaves=63, min_child_samples=30, lambdarank_truncation_level=trunc,
            random_state=0, verbosity=-1, n_jobs=4)
        ranker.fit(np.array(trF, dtype=np.float32), np.array(trY, dtype=np.int8), group=trG)
        for k_i, qa in enumerate(IDS):
            if CONV_of[qa] != hold:
                continue
            G = GSETS[k_i]
            if not G:
                continue
            pool = POOLA[k_i]
            s = ranker.predict(FEATS[k_i])
            seq = [pool[i] for i in np.argsort(-s)]
            post = v9_post(seq, qa)
            if all(any(g in post[:15] for g in grp) for grp in G):
                RES[name][0] += 1
            cg = core_group(qa, G)
            if any(g in post[:5] for g in cg):
                RES[name][1] += 1
            FORD11[qa] = [MID[i] for i in post[:35]]
    P("  fold %s done %.0fs" % (hold[-12:], time.time() - t0))

nG = sum(1 for g in GSETS if g)
P("===== LGBMRanker grid n=%d =====" % nG)
for name, _, _, _ in CONFIGS:
    P("%-24s all15=%5.1f%%  core5=%5.1f%%" % (name, 100.0 * RES[name][0] / nG, 100.0 * RES[name][1] / nG))
P("pointwise v10: all15=78.6 core5=76.7")
json.dump(FORD11, open(HERE + "/r41_final_order_v14_256dom.json", "w", encoding="utf-8"), ensure_ascii=False)
P("v14 256dom order saved")
P("F94_DONE %.0fs" % (time.time() - t0))
LOG.close()
