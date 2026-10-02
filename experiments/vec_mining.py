# -*- coding: utf-8 -*-
"""vec_mining.py — 系统挖掘qwen词向量的隐藏结构
M1 月份循环序(已发现0.902 vs 0.822) — 对照验证词面假说(January/July词面相似)
M2 星期循环序: Monday-Sunday应有同样的相邻>远隔结构
M3 数字符: one~twelve数字词的"数量序"(one-two相邻应>one-twelve)
M4 反义差向量: big-small, hot-cold, up-down等差向量的一致性(方向是否成族)
M5 比较级序: good<better<best, big<bigger<biggest 程度序
M6 实体类别簇: 动物/颜色/职业/水果 各自成簇? 簇间关系(动物-狗猫 vs 水果-苹果)
M7 情感价序: 正面词簇 vs 负面词簇, 中性词居中?
M8 集合成员: 众数词(fruit: apple/banana; color: red/blue)的簇归属准确率
每项: 构造词组→cos矩阵→结构统计
"""
import io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
import urllib.request
def emb(texts):
    out = []
    for s in range(0, len(texts), 32):
        body = json.dumps({"model": "qwen3-embedding:latest",
                           "input": texts[s:s + 32], "dimensions": 256}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r2:
            out.append(np.asarray(json.loads(r2.read())["embeddings"], dtype=np.float32))
    return l2n(np.concatenate(out))

def circ_test(words, name, cyc=True):
    E = l2n(emb(words))
    n = len(words)
    adj, far = [], []
    for a2 in range(n):
        for b2 in range(n):
            if a2 == b2:
                continue
            c = float(E[a2] @ E[b2])
            d2 = abs(a2 - b2)
            if cyc:
                d2 = min(d2, n - d2)
            if d2 == 1:
                adj.append(c)
            elif d2 >= 3:
                far.append(c)
    print("M %-28s 相邻cos=%.3f 远隔cos=%.3f 序感知=%s" % (
        name, np.mean(adj), np.mean(far), "✓" if np.mean(adj) > np.mean(far) + 0.02 else "✗"), flush=True)
    return np.mean(adj) - np.mean(far)

print("== M1 月份循环序 + 词面对照 ==", flush=True)
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
d1 = circ_test(months, "月份(真实序)")
# 词面假说对照: 乱序同词表(序打乱后"相邻"就变了) — 若词面假说成立, 乱序后相邻cos应该不变
shuf = ["March", "July", "November", "January", "September", "May", "December", "April", "October", "June", "February", "August"]
d1s = circ_test(shuf, "月份(乱序对照)", cyc=False)
print("  → 序感知差: 真序%+.3f vs 乱序%+.3f → %s" % (
    d1, d1s, "结构真实(非词面)" if d1 > d1s + 0.01 else "词面假说未排除"), flush=True)

print("== M2 星期循环序 ==", flush=True)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
circ_test(days, "星期(循环)")

print("== M3 数量序 ==", flush=True)
nums = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
circ_test(nums, "数字1-12(线性序, 不循环)", cyc=False)

print("== M4 反义差向量一致性 ==", flush=True)
ant_pairs = [("big", "small"), ("hot", "cold"), ("up", "down"), ("fast", "slow"),
             ("happy", "sad"), ("love", "hate"), ("rich", "poor"), ("strong", "weak"),
             ("bright", "dark"), ("new", "old")]
diffs = []
for a2, b2 in ant_pairs:
    E2 = l2n(emb([a2, b2]))
    diffs.append(E2[0] - E2[1])
diffs = l2n(np.stack(diffs))
n = len(diffs)
MM = diffs @ diffs.T
iu = np.triu_indices(n, 1)
print("  %d个反义差向量两两cos: 均值%.3f (随机基线≈0; >0.3=方向成族)" % (n, MM[iu].mean()), flush=True)

print("== M5 比较级程度序 ==", flush=True)
good_chain = ["bad", "good", "better", "excellent"]
big_chain = ["small", "big", "bigger", "huge"]
for chain, nm in ((good_chain, "bad→good→better→excellent"), (big_chain, "small→big→bigger→huge")):
    E3 = l2n(emb(chain))
    adjc = [float(E3[a2] @ E3[a2 + 1]) for a2 in range(len(chain) - 1)]
    farc = [float(E3[0] @ E3[-1])]
    print("  %-30s 相邻cos均值%.3f 首尾cos%.3f (序感知=%s)" % (
        nm, np.mean(adjc), farc[0], "✓" if np.mean(adjc) > farc[0] else "✗"), flush=True)

print("== M6 实体类别簇 ==", flush=True)
groups = {
    "动物": ["dog", "cat", "horse", "lion", "tiger", "rabbit"],
    "颜色": ["red", "blue", "green", "yellow", "purple", "orange"],
    "水果": ["apple", "banana", "grape", "peach", "mango", "cherry"],
    "职业": ["teacher", "doctor", "lawyer", "engineer", "nurse", "pilot"],
    "家具": ["chair", "table", "sofa", "bed", "desk", "wardrobe"],
}
Em = {}
for g, ws in groups.items():
    Em[g] = l2n(emb(ws))
gk = list(groups.keys())
print("  簇内cos:", flush=True)
for g in gk:
    MM2 = Em[g] @ Em[g].T
    iu = np.triu_indices(len(g), 1)
    print("    %-4s %.3f" % (g, MM2[iu].mean()), flush=True)
print("  簇间cos(均值):", flush=True)
for a2 in range(len(gk)):
    for b2 in range(a2 + 1, len(gk)):
        cAB = Em[gk[a2]] @ Em[gk[b2]].T
        print("    %s-%s: %.3f" % (gk[a2], gk[b2], cAB.mean()), flush=True)
# M8 簇归属准确率: 留一最近邻分类
correct = total = 0
for g in gk:
    E4 = l2n(emb(groups[g]))
    for i2 in range(len(groups[g])):
        # 最近簇心(排除自己)
        best_g, best_s = None, -9
        for g2 in gk:
            others = [j2 for j2 in range(len(groups[g2])) if not (g2 == g and j2 == i2 and g2 == g)]
            others = [j2 for j2 in range(len(groups[g2]))]
            if g2 == g and len(groups[g2]) > 1:
                others = [j2 for j2 in others if j2 != i2]
            if not others:
                continue
            c2 = l2n(Em[g2][others].mean(0)[None])[0]
            s2 = float(E4[i2] @ c2)
            if s2 > best_s:
                best_s, best_g = s2, g2
        total += 1
        if best_g == g:
            correct += 1
print("  M8 留一簇归属准确率: %d/%d = %.0f%%" % (correct, total, 100 * correct / total), flush=True)

print("== M7 情感价 ==", flush=True)
pos = ["happy", "joy", "love", "wonderful", "excellent", "delight", "cheerful", "blessed"]
neg = ["sad", "grief", "hate", "terrible", "awful", "misery", "gloomy", "cursed"]
neu = ["table", "window", "pencil", "carpet", "handle", "ceiling", "packet", "bolt"]
P, N2, U3 = l2n(emb(pos)), l2n(emb(neg)), l2n(emb(neu))
test_words = ["vacation", "funeral", "picnic", "argument", "gift", "illness", "sunset", "homework"]
print("  词        正面cos  负面cos  中性cos  预测价", flush=True)
for w in test_words:
    v = l2n(emb([w])[0])
    print("  %-10s %.3f   %.3f   %.3f   %s" % (
        w, float(v @ P.mean(0)), float(v @ N2.mean(0)), float(v @ U3.mean(0)),
        "正" if v @ P.mean(0) > v @ N2.mean(0) else "负"), flush=True)
print("MINING_DONE", flush=True)
