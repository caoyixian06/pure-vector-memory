import dd_common as C
import numpy as np

print("D1_q_with_ev", sum(1 for q in C.QMETA if q["ev_rows"]))
rng = np.random.RandomState(0)
js = rng.choice(1376, 400, replace=False)
for sh in (-2, -1, 0, 1, 2):
    jj = js[(js + sh >= 0) & (js + sh < 1382)]
    cs = np.sum(C.SB[jj] * C.S[jj + sh], axis=1) / (np.linalg.norm(C.SB[jj], axis=1) * np.linalg.norm(C.S[jj + sh], axis=1) + 1e-9)
    print("D2_shift", sh, "med_cos", round(float(np.median(cs)), 4), "frac>0.99", round(float((cs > 0.99).mean()), 3))
# row norms
print("D3 norms SB", round(float(np.median(np.linalg.norm(C.SB, axis=1))), 2), "S", round(float(np.median(np.linalg.norm(C.S, axis=1))), 2))
# centered-profile match
SBc = C.SB - C.SB.mean(1, keepdims=True)
Sc = C.S - C.S.mean(1, keepdims=True)
SBc = SBc / (np.linalg.norm(SBc, axis=1, keepdims=True) + 1e-9)
Sc = Sc / (np.linalg.norm(Sc, axis=1, keepdims=True) + 1e-9)
M = SBc @ Sc.T
r2q = M.argmax(1)
vals = M[np.arange(1376), r2q]
print("D4_centered med_best", round(float(np.median(vals)), 4), "uniq_mapped", len(set(r2q.tolist())), "frac>0.99", round(float((vals > 0.99).mean()), 3))
# per-question check of identity: for q index qi, best row
qs = rng.choice(1382, 20, replace=False)
for qi in qs[:10]:
    jb = M[:, qi].argmax()
    print("D5_q", int(qi), "best_row", int(jb), "val", round(float(M[jb, qi]), 4))
print("DONE")
