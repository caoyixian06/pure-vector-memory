import dd_common as C
import numpy as np, json

print("P1_shapes", C.V.shape, C.S.shape, C.SB.shape, len(C.QMETA))
print("P2_top50_range", int(C.TOP50.min()), int(C.TOP50.max()))
uniq = len(set(C.ROW2Q))
print("P3_row2q_uniq", uniq, "of", len(C.ROW2Q), "mapped_q", sum(1 for x in C.QROW if x >= 0))
# mapping quality: recompute corr for 200 rows
rs = np.random.RandomState(0).choice(len(C.ROW2Q), 200, replace=False)
cor = []
for j in rs:
    qi = C.ROW2Q[j]
    cor.append(float(np.dot(C.SB[j], C.S[qi]) / (np.linalg.norm(C.SB[j]) * np.linalg.norm(C.S[qi]) + 1e-9)))
print("P4_mapcorr_min_med", round(min(cor), 3), round(float(np.median(cor)), 3))
# evidence recall@50
in50 = 0; has_ev = 0; evc = 0
for i, qm in enumerate(C.QMETA):
    j, ev, t50, noise, rer = C.pools(i)
    if j < 0 or not ev:
        continue
    has_ev += 1
    if ev <= set(t50.tolist()):
        in50 += 1
    evc += len(ev)
print("P5_q_with_ev", has_ev, "ev_in_top50", in50, "mean_ev_rows", round(evc / max(has_ev, 1), 2))
# twin pairs
print("P6_twins", len(C.TWIN_PAIRS), "hdr_convs", len(C.HDR_TURNS))
ns = [C.sess_idx_of(C.MIDS[0])]
print("P7_spk_known", sum(1 for m in C.MIDS if C.SPK_OF_MID.get(m, "?") not in ("?", "")), "/", len(C.MIDS))
print("DONE")
