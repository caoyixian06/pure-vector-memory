# -*- coding: utf-8 -*-
import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = "C:/locomo_refined/memsys"

ks = set(); n = 0; sample = None
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l); ks.update(r.keys()); n += 1
    if sample is None and r.get("kind") == "raw":
        sample = r
print("mem.jsonl total", n)
print("keys:", sorted(ks))
if sample:
    s = {k: (str(v)[:60] if not isinstance(v, list) else "list[%d]" % len(v)) for k, v in sample.items()}
    print("raw sample:", json.dumps(s, ensure_ascii=False)[:500])

for f in ("word_vecs.npz", "word_hosts.json", "mem_vecs.npz", "qwen_lib.npz", "mem_qwen.npz"):
    p = HERE + "/" + f
    print(f, "exists" if os.path.exists(p) else "MISSING", os.path.getsize(p) if os.path.exists(p) else "")

import numpy as np
if os.path.exists(HERE + "/word_vecs.npz"):
    z = np.load(HERE + "/word_vecs.npz", allow_pickle=True)
    print("word_vecs keys:", list(z.keys())[:5])
    for k in list(z.keys())[:2]:
        print("  ", k, getattr(z[k], "shape", type(z[k])))
if os.path.exists(HERE + "/word_hosts.json"):
    wh = json.load(open(HERE + "/word_hosts.json", encoding="utf-8"))
    if isinstance(wh, dict):
        it = list(wh.items())[:3]
        print("word_hosts entries:", len(wh), "sample:", [(w, str(h)[:80]) for w, h in it])
