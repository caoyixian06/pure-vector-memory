# -*- coding: utf-8 -*-
"""foundry116.py — 上下文词向量首批导出(LoCoMo全库~31万token, fp16):
①transformer底层last_hidden_state逐token导出 ②同词不同句的向量差实测(light等)"""
import io, json, os, re, sys, time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
LOG = io.open("C:/locomo_refined/foundry116_results.txt", "w", encoding="utf-8")
def P(s):
    print(s, flush=True)
    LOG.write(str(s) + chr(10)); LOG.flush()

t0 = time.time()
HERE = "C:/locomo_refined/memsys"
MID = [json.loads(l)["mid"] for l in io.open(HERE + "/mem_bge_sparse.jsonl", encoding="utf-8") if l.strip()]
REC = {}
for l in io.open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        REC[json.loads(l).get("memory_id")] = json.loads(l)
    except Exception:
        pass
def rec_raw(m):
    r = REC.get(m, {})
    for f in ("raw", "text", "content"):
        if isinstance(r.get(f), str) and r[f].strip():
            return r[f]
    return ""
RAWS = [rec_raw(m) for m in MID]
P("records=%d %.0fs" % (len(RAWS), time.time() - t0))

import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
P("model loaded %.0fs" % (time.time() - t0))

OUTDIR = "C:/locomo_refined/ctx_token_vecs"
os.makedirs(OUTDIR, exist_ok=True)
meta_f = open(OUTDIR + "/tokens_meta.jsonl", "w", encoding="utf-8")
vec_parts = []
token_count = 0
B = 24
with torch.no_grad():
    for st in range(0, len(RAWS), B):
        batch = [r[:1500] for r in RAWS[st:st + B]]
        enc = tok(batch, padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        out = model(**enc).last_hidden_state  # (B, L, 1024)
        mask = enc["attention_mask"].bool()
        for bi in range(len(batch)):
            ids = enc["input_ids"][bi][mask[bi]]
            vecs = out[bi][mask[bi]].cpu().numpy().astype(np.float16)
            toks_ = tok.convert_ids_to_tokens(ids.tolist())
            vec_parts.append(vecs)
            token_count += len(toks_)
            for ti, tk_ in enumerate(toks_):
                meta_f.write(json.dumps({"r": st + bi, "t": tk_, "i": ti}) + "\n")
        if (st // B) % 100 == 0:
            P("  %d/%d rows, %d tokens %.0fs" % (st, len(RAWS), token_count, time.time() - t0))
meta_f.close()
ALL = np.concatenate(vec_parts)
np.save(OUTDIR + "/ctx_vecs.npy", ALL)
P("导出完成: %d tokens × %d维 (fp16, %.1fGB) %.0fs" % (
    ALL.shape[0], ALL.shape[1], ALL.nbytes / 1e9, time.time() - t0))

# ===== 同词不同句的向量差实测(用户直觉验证) =====
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
ALLn = l2n(ALL.astype(np.float32))
from collections import defaultdict
by_tok = defaultdict(list)
meta = [json.loads(l) for l in io.open(OUTDIR + "/tokens_meta.jsonl", encoding="utf-8")]
for mi, m in enumerate(meta):
    if len(by_tok[m["t"]]) < 200:
        by_tok[m["t"]].append(mi)
P("\n===== 同词不同句的向量差(上下文效应实测) =====")
for w in ("▁light", "▁bank", "▁work", "▁the", "▁Melanie"):
    idxs = by_tok.get(w, [])
    if len(idxs) < 10:
        P("%-10s 样本不足(n=%d)" % (w, len(idxs)))
        continue
    V = ALLn[idxs]
    S = V @ V.T
    iu = np.triu_indices(len(idxs), 1)
    sims = S[iu]
    P("%-10s n=%d  同词互cos: 均值=%.3f 最小=%.3f 最大=%.3f  (1.0=完全同向量)" % (
        w, len(idxs), sims.mean(), sims.min(), sims.max()))
P("解读: 互cos<1的幅度=上下文偏移量; 虚词(the)应接近1, 多义词(light/bank)应更低")
P("F116_DONE %.0fs" % (time.time() - t0))
