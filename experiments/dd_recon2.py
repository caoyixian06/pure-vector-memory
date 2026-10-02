import io, json, os, sys, re
import numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
base = "C:/locomo_refined/memsys"

recs = [json.loads(l) for l in open(os.path.join(base, "mem.jsonl"), encoding="utf-8")]
mids = []
with open(os.path.join(base, "mem_bge_sparse.jsonl"), encoding="utf-8") as f:
    for line in f:
        mids.append(json.loads(line)["mid"])
midset = set(mids)

def suf(m):
    mm = re.search(r"_(rbak(\d+)k|m(\d+))$", m)
    if not mm: return ("none", -1)
    return ("rbak", int(mm.group(2))) if mm.group(1).startswith("rbak") else ("m", int(mm.group(3)))

print("H1_memid_unique", len(set(r["memory_id"] for r in recs)), "of", len(recs))
print("H2_mem_suf", dict(Counter(suf(r["memory_id"])[0] for r in recs)))
print("H3_sparse_suf", dict(Counter(suf(m)[0] for m in mids)))
print("H4_kind_sparse", dict(Counter((r.get("kind") or "none", r["memory_id"] in midset) for r in recs)))

raws = [r for r in recs if r.get("kind") == "raw"][:3]
for r in raws:
    print("H5_raw", r["memory_id"], "raw_of=", r.get("raw_of"), "twin=", r.get("twin_of"), "sess=", r.get("session_id"), "vlen=", len(r.get("vector") or []), "raw=", r["raw"][:40].encode("ascii", "replace").decode())

# public data dir
pdir = "C:/locomo_refined/LoCoMo_refined-main/data/public"
print("H6_pubdir", sorted(os.listdir(pdir))[:20])
qf = os.path.join(pdir, "questions.jsonl")
if os.path.exists(qf):
    qs = [json.loads(l) for l in open(qf, encoding="utf-8")]
    print("H7_pubq_n", len(qs), "keys", ",".join(sorted(qs[0].keys())))

# evidence text match
def nrm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()

o = json.load(open(os.path.join(base, "out_r37.json"), encoding="utf-8"))
em_texts = []
for q in o:
    for em in (q.get("evidence_messages") or []):
        em_texts.append(em["text"])
print("H8_n_emtexts", len(em_texts), "uniq", len(set(em_texts)))

rec_by_ntext = {}
rec_by_ntext_raw = {}
for i, r in enumerate(recs):
    if r["memory_id"] in midset:
        rec_by_ntext.setdefault(nrm(r.get("raw") or ""), []).append(i)
        if r.get("kind") == "raw":
            rec_by_ntext_raw.setdefault(nrm(r.get("raw") or ""), []).append(i)
hit_all = sum(1 for t in em_texts if nrm(t) in rec_by_ntext)
hit_raw = sum(1 for t in em_texts if nrm(t) in rec_by_ntext_raw)
print("H9_em_hit_all", hit_all, "/", len(em_texts), "hit_raw", hit_raw)

# question order alignment with public
print("H10_q0_match", nrm(o[0]["question"]) == nrm(qs[0]["question"]) if os.path.exists(qf) else "na")
print("H11_evfield", o[0].get("evidence"), "| conv_idx", o[0].get("conversation_idx"), "| qa_id", o[0].get("qa_id"), "| cat", o[0].get("category").encode("ascii","replace").decode())
print("DONE")
