# -*- coding: utf-8 -*-
import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = json.load(open(r"C:\locomo_refined\locomo10.json", encoding="utf-8"))
print("type:", type(d), "len:", len(d))
item = d[0] if isinstance(d, list) else list(d.values())[0]
print("item keys:", list(item.keys()))
if isinstance(d, list):
    print("sample_ids:", [x.get("sample_id") for x in d])
ann = item.get("annotation") or {}
print("annotation keys:", list(ann.keys()) if isinstance(ann, dict) else type(ann))
if isinstance(ann, dict):
    for k, v in ann.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print("ann[%s] n=%d first=%s" % (k, len(v), json.dumps(v[0], ensure_ascii=False)[:300]))
        else:
            print("ann[%s] = %s" % (k, str(v)[:100]))
conv = item.get("conversation") or {}
print("conv keys sample:", list(conv.keys())[:8])
