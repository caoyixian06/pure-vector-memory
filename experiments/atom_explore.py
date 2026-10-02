# -*- coding: utf-8 -*-
"""atom_explore.py — 原子句域双包: A文本规律迁移验证 + B过滤策略对比 (全离线零GLM)
A包: 文本空间规律在原子句空间的变形
  A1 共模(证据原子vs噪声原子的跨维相关) [对照1024的0.986]
  A2 半球(问题质心vs原子句质心夹角)      [对照117°]
  A3 有效秩                              [对照22/1024]
  A4 兄弟优势(同turn兄弟vs跨turn)        [对照0.679]
  A5 答案可达性(原子句×答案cos vs 原子句×问题cos) [对照q_a<q_ev]
B包: 过滤策略对比(F0无/F1动词表/F2寒暄锚距离/F3密度) → 桶一召回+纯度
"""
import io, json, os, sys, re, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

HERE = "C:/locomo_refined/memsys"
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
def emb(texts):
    out = []
    for s in range(0, len(texts), 64):
        out.append(np.asarray(bge.encode(texts[s:s + 64])["dense_vecs"], dtype=np.float32))
    return l2n(np.concatenate(out))
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w
def toks_set(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)
COUR = re.compile(r"\b(hi|hey|hello|wow|awesome|great|thanks|thank|cool|nice)\b", re.I)
VERB = re.compile(r"\b(is|was|are|were|have|has|had|do|does|did|will|would|can|could|went|go|make|made|take|took|get|got|play|played|read|reads|started|finished|work|works|worked|live|lives|lived|like|likes|loves|adopted|won|signed|joined|planning|plan|bought|sold|graduated|studying|study|moving|moved)\b", re.I)

# 全量原子句(含被过滤的) — 用于过滤对比
atoms_all = []   # (text, sess, has_verb)
for l in open(HERE + "/mem.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("kind") != "raw":
        continue
    s = (r.get("raw") or "").strip()
    if s.startswith("[Session"):
        continue
    sess = r.get("session_id") or ""
    for piece in re.split(r"[.!?]+", s):
        p = piece.strip()
        words = p.split()
        if len(words) < 4 or len(words) > 45:
            continue
        hv = bool(VERB.search(p))
        atoms_all.append((p, sess, hv))
print("全量切分:", len(atoms_all), flush=True)

# ===== B包: 四种过滤的桶一召回 =====
qmap = {json.loads(l)["qa_id"]: json.loads(l) for l in open(
    "C:/locomo_refined/LoCoMo_refined-main/data/public/questions.jsonl", encoding="utf-8") if l.strip()}
R37 = {r["qa_id"]: r for r in json.load(open(HERE + "/out_r37.json", encoding="utf-8"))}
def is_dk(p):
    if not p:
        return True
    p = str(p).strip().lower()
    for kw in ["not sure", "don't know", "dont know", "unknown", "no information",
               "not mentioned", "not specified", "cannot", "无法", "不知道", "未提及"]:
        if kw in p:
            return True
    return len(p) <= 2
bucket1 = [r for r in R37.values() if r.get("llm_score") != 1 and is_dk(r.get("predicted_answer"))]
print("桶一:", len(bucket1), flush=True)

def toks(s):
    return set(stem(w) for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 3)

# 四种过滤的原子句集
F = {}
F["F0无过滤"] = [(p, s) for p, s, hv in atoms_all]
F["F1动词表"] = [(p, s) for p, s, hv in atoms_all if hv]
F["F2寒暄锚"] = None  # 下面向量算完再定
F["F3密度"] = None

sess_all = {}
for k2, (p, s, hv) in enumerate(atoms_all):
    sess_all.setdefault(s, []).append(k2)

t0 = time.time()
AV = emb([a[0] for a, s, hv in atoms_all])
print("embedded %.0fs" % (time.time() - t0), flush=True)

# F2 寒暄锚: 每session前3句的均值
sess_first = {}
for k2, (p, s, hv) in enumerate(atoms_all):
    sess_first.setdefault(s, []).append(k2)
anchors = []
for s, idxs in sess_first.items():
    for k2 in idxs[:2]:
        anchors.append(AV[k2])
if anchors:
    fluff_c = l2n(np.mean(np.stack(anchors), axis=0)[None])[0]
d_f2 = np.array([float(AV[k2] @ fluff_c) for k2 in range(len(atoms_all))])
thr2 = np.percentile(d_f2, 50)  # 距寒暄锚最远的50%
F["F2寒暄锚"] = [(atoms_all[k2][0], atoms_all[k2][1]) for k2 in range(len(atoms_all)) if d_f2[k2] <= thr2]

# F3 密度: k近邻平均距离(越大越独特)
knn = 10
sample_idx = list(range(0, len(atoms_all), max(1, len(atoms_all) // 3000)))
SAMP = AV[sample_idx]
dens = []
for k2 in range(len(atoms_all)):
    dd = SAMP @ AV[k2]
    topk = np.sort(dd)[-knn-1:-1]
    dens.append(float(topk.mean()))
dens = np.array(dens)
thr3 = np.percentile(dens, 50)  # 密度低的一半=独特句
F["F3密度"] = [(atoms_all[k2][0], atoms_all[k2][1]) for k2 in range(len(atoms_all)) if dens[k2] <= thr3]

def recall_of(fset, tag):
    sess_idx = {}
    for k2, (p, s2) in enumerate(fset):
        sess_idx.setdefault(s2, []).append(k2)
    VT = emb([p for p, s2 in fset])
    hit = 0
    for r in bucket1:
        q = qmap.get(r["qa_id"])
        sid = "loco-" + str(q.get("sample_id")) if q else ""
        idxs = sess_idx.get(sid, [])
        if not idxs:
            continue
        qv = VT[idxs] @ emb([r["question"]])[0]
        order = np.argsort(-qv)[:25]
        et_set = set()
        for e in q.get("evidence_messages") or []:
            et_set |= toks(e.get("text") or "")
        found = False
        for oi in order:
            at_t = toks(fset[oi][0])
            if not at_t:
                continue
            ov = len(at_t & et_set) / max(1, min(len(at_t), len(et_set)))
            if ov >= 0.5:
                found = True
                break
        if found:
            hit += 1
    print("  %-10s 召回 %d/%d (%.0f%%)" % (tag, hit, len(bucket1), 100 * hit / len(bucket1)), flush=True)
    return hit

print("== B包 过滤策略对比(桶一147) ==", flush=True)
for tag in ("F0无过滤", "F1动词表", "F2寒暄锚", "F3密度"):
    recall_of(F[tag], tag)

# ===== A包: 规律迁移 =====
print("== A包 规律迁移(原子句空间) ==", flush=True)
# 用F1动词表集(在役版)做空间分析
FA = F["F1动词表"]
AVa = emb([p for p, s2 in FA])
A2S = [s2 for p, s2 in FA]
print("A3 有效秩:", flush=True)
U, S, Vt = np.linalg.svd(AVa - AVa.mean(0), full_matrices=False)
eff = (S ** 2).sum() / (S[0] ** 2)
print("  原子句空间有效秩=%.0f/%d [文本空间22/1024]" % (eff, min(len(FA), 1024)), flush=True)
# A1 共模: 需要 证据原子/噪声原子 分组 — 用bucket1证据附近的原子 vs 远处原子
import random
rng = random.Random(5)
ev_atoms, no_atoms = [], []
for r in bucket1:
    q = qmap.get(r["qa_id"])
    sid = "loco-" + str(q.get("sample_id")) if q else ""
    idxs = sess_idx.get(sid, [])
    if not idxs:
        continue
    et_set = set()
    for e in q.get("evidence_messages") or []:
        et_set |= toks(e.get("text") or "")
    for oi in idxs:
        at_t = toks(FA[oi][0])
        if not at_t:
            continue
        ov = len(at_t & et_set) / max(1, min(len(at_t), len(et_set)))
        if ov >= 0.4:
            ev_atoms.append(oi)
        elif ov == 0:
            no_atoms.append(oi)
ev_atoms = list(dict.fromkeys(ev_atoms))[:800]
no_atoms = list(dict.fromkeys(no_atoms))[:1600]
if ev_atoms and no_atoms:
    me = AVa[ev_atoms].mean(0) if False else AV[ev_atoms].mean(0)
    mn = AV[no_atoms].mean(0)
    print("A1 共模(证据原子vs噪声原子 跨维相关): %.3f [文本0.986]" % float(np.corrcoef(me, mn)[0, 1]), flush=True)
    # A2 半球: 问题质心 vs 陈述原子质心
    qb = emb([qmap.get(r["qa_id"], {}).get("question", "") for r in bucket1[:200]])
    print("A2 问题质心×原子质心 cos: %.3f [句空间半球117°]" % float(l2n(qb.mean(0)[None])[0] @ l2n(me[None])[0]), flush=True)
# A4 兄弟: 同session相邻原子对 vs 随机对
import itertools
sib = []
rnd_pairs = []
for s, idxs in sess_all.items():
    if s not in sess_idx:
        continue
    fidx = [k2 for k2 in idxs if (atoms_all[k2][0], atoms_all[k2][1]) in set(FA)]
    fidx = [k2 for k2 in idxs if any(FA[k3][0] == atoms_all[k2][0] for k3 in range(min(len(FA), 200)))]
    if len(fidx) >= 2:
        for a2, b2 in itertools.combinations(fidx[:20], 2):
            sib.append(float(AV[a2] @ AV[b2]))
allidx = list(range(len(FA)))
for _ in range(200):
    a2, b2 = rng.choice(allidx), rng.choice(allidx)
    if a2 != b2:
        rnd_pairs.append(float(AVa[a2] @ AVa[b2]))
if sib:
    print("A4 同session原子对cos=%.3f vs 随机对=%.3f (兄弟优势在原子域: %s)" % (
        np.mean(sib), np.mean(rnd_pairs), "存在" if np.mean(sib) > np.mean(rnd_pairs) + 0.05 else "弱"), flush=True)
# A5 答案可达性: 答案与自己的证据原子cos vs 问题与证据原子
ax_list = []
for r in bucket1[:200]:
    q = qmap.get(r["qa_id"])
    sid = "loco-" + str(q.get("sample_id")) if q else ""
    idxs = sess_idx.get(sid, [])
    if not idxs:
        continue
    qv = VT[idxs] @ emb([r["question"]])[0]
    xv = VT[idxs] @ emb([str(r["answer"][0])])[0]
    for oi in np.argsort(-qv)[:3]:
        ax_list.append((float(xv[oi]), float(qv[oi])))
if ax_list:
    xs = np.array(ax_list)
    print("A5 原子句可达性: 答案↔原子=%.3f 问题↔原子=%.3f (差%+.3f) [句空间: 答案更远]" % (
        xs[:, 0].mean(), xs[:, 1].mean(), (xs[:, 0] - xs[:, 1]).mean()), flush=True)
print("EXPLORE_DONE", flush=True)
