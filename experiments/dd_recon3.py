import io, json, os, sys, re
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = "C:/locomo_refined/memsys"

recs = [json.loads(l) for l in open(os.path.join(base, "mem.jsonl"), encoding="utf-8")]
o = json.load(open(os.path.join(base, "out_r37.json"), encoding="utf-8"))

cf = "C:/locomo_refined/LoCoMo_refined-main/data/public/conversations.jsonl"
convs = [json.loads(l) for l in open(cf, encoding="utf-8")]
print("I1_conv_n", len(convs), "keys", ",".join(sorted(convs[0].keys())))
m0 = convs[0]["session_1"][0] if "session_1" in convs[0] else None
print("I2_sesskeys", [k for k in convs[0].keys()][:6])
if m0 is not None:
    print("I3_msgkeys", ",".join(sorted(m0.keys())))

# sample: em text vs rec raw
em0 = o[0]["evidence_messages"][0]
print("I4_em_text=", em0["text"][:80].encode("ascii", "replace").decode(), "| dia", em0.get("dia_id"), "| spk", em0.get("speaker").encode("ascii","replace").decode())
cand = [r for r in recs if r.get("kind") == "raw" and r["session_id"] == "loco-conv-26"][:200]
print("I5_recraw=", cand[3]["raw"][:80].encode("ascii", "replace").decode(), cand[3]["memory_id"])

def nrm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

def strip_spk(t):
    return re.sub(r"^[A-Za-z .']+\s*:\s*", "", t)

# index raw records by normalized stripped text
idx = {}
for r in recs:
    if r.get("kind") == "raw":
        idx.setdefault(nrm(strip_spk(r["raw"])), []).append(r["memory_id"])
tot, hit = 0, 0
misses = []
for q in o:
    for em in (q.get("evidence_messages") or []):
        tot += 1
        k = nrm(em["text"])
        if k in idx: hit += 1
        elif len(misses) < 3: misses.append(em["text"][:70])
print("I6_hit", hit, "/", tot)
for m in misses: print("I7_miss=", m.encode("ascii", "replace").decode())
print("DONE")
