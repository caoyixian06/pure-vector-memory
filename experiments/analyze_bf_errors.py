# -*- coding: utf-8 -*-
"""analyze_bf_errors.py — bf全量60.3%的错因解剖(真gold, 1382题逐题)"""
import io, json, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 逐题分数
score = {}
for l in io.open("bf_full_gold_detail.txt", encoding="utf-8"):
    m = re.match(r"(\S+) ([01-]+) \|", l)
    if m:
        s = m.group(2)
        score[m.group(1)] = int(s) if s in "01" else -1
# submission预测
pred = {}
for l in io.open("submission_r40bf.jsonl", encoding="utf-8"):
    r = json.loads(l)
    pred[r["qa_id"]] = str(r.get("predicted_answer") or "")
# questions金标
gold, qtext, ansn = {}, {}, {}
for l in io.open("questions_r40bf.jsonl", encoding="utf-8"):
    q = json.loads(l)
    gold[q["qa_id"]] = [str(x) for x in (q.get("answer") or [])]
    qtext[q["qa_id"]] = q.get("question") or ""
    ansn[q["qa_id"]] = len(q.get("answer") or [])

ids = [i for i in score if i in pred and i in gold]
ok = sum(1 for i in ids if score[i] == 1)
err = [i for i in ids if score[i] == 0]
print("总=%d 对=%d(%.1f%%) 错=%d 解析失败=%d" % (
    len(ids), ok, 100.0*ok/len(ids), len(err), sum(1 for i in ids if score[i] == -1)))

DK = re.compile(r"不知道|无法确定|没有相关信息|无法回答|记忆中|not mentioned|no information|unclear|cannot determine", re.I)
DATE_ANCH = re.compile(r"2023年6月1[0-9日]|2023年6月2[0-9日]|2023年7月20日")
TIMEQ = re.compile(r"^(When|What time|Which (year|week|day)|How (long|old|many times|often))", re.I)

def toks(s):
    return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2)

cat = {"不知道型(该答不答)": [], "日期锚定病": [], "时间口径(单日vs区间/差一日)": [],
       "列举不全(沾边)": [], "列举全错": [], "答错方向": [], "疑似判官冤案": []}
for i in err:
    p, g = pred[i], " ".join(gold[i])
    gt = toks(g)
    pt = toks(p)
    if not gt:
        continue
    cov = len(gt & pt) / max(1, len(gt))
    is_time = bool(TIMEQ.match(qtext[i])) or bool(re.search(r"\d{4}|\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)", g))
    if DK.search(p):
        cat["不知道型(该答不答)"].append(i)
    elif DATE_ANCH.search(p) and is_time:
        cat["日期锚定病"].append(i)
    elif is_time:
        cat["时间口径(单日vs区间/差一日)"].append(i)
    elif len(gold[i]) > 1 or re.search(r",|/| and ", g):
        if cov >= 0.35:
            cat["列举不全(沾边)"].append(i)
        else:
            cat["列举全错"].append(i)
    elif cov >= 0.75:
        cat["疑似判官冤案"].append(i)
    else:
        cat["答错方向"].append(i)

print("\n===== 错因分布 (错题=%d) =====" % len(err))
for k, v in sorted(cat.items(), key=lambda kv: -len(kv[1])):
    print("%-28s %4d (%.0f%%)" % (k, len(v), 100.0*len(v)/max(1, len(err))))
print("\n===== 各类抽样 =====")
for k, v in cat.items():
    print("--- " + k)
    for i in v[:3]:
        print("  [%s] Q:%s" % (i, qtext[i][:56]))
        print("     G:%s" % " ; ".join(gold[i])[:70])
        print("     P:%s" % pred[i][:80])
# 列举题总分账
enum_ids = [i for i in ids if ansn[i] > 1 or re.search(r",|/| and ", " ".join(gold[i]))]
enum_ok = sum(1 for i in enum_ids if score[i] == 1)
solo_ids = [i for i in ids if i not in set(enum_ids)]
solo_ok = sum(1 for i in solo_ids if score[i] == 1)
print("\n列举题 %d对%.1f%% vs 单答案 %d对%.1f%%" % (
    len(enum_ids), 100.0*enum_ok/max(1, len(enum_ids)), len(solo_ids), 100.0*solo_ok/max(1, len(solo_ids))))
