# -*- coding: utf-8 -*-
"""清理测试查询污染。用法: python remote_clean.py [--delete]
不带 --delete 为 dry-run(只输出报告不改动)。可重复执行(幂等)。

判定标准(显式白名单, 宁可少删不可误删):
真实记录唯一来源 = ingest_profile.py 的 memory_add
  (text 固定格式 "[{blk}/{cat}#{i}] 会话... | ... | 原文: ...", request_id={blk}-{cat}-{i})
  经 migrate_global.py 迁入 profile_all(request_id 变为 {blk}-...)。
白名单:
- profile_all: 保留 kind∈{experience,experience_correction} 或
    (id 匹配 r"\\d{3}#\\d+" 且 text 以 "[{块号}/" 开头 且 request_id 以 "{块号}-" 开头)
  → 删除: 直接对 profile_all 的 ask 自写入(s#N)、default 前缀验证记录、
    迁移进来的裸问题测试查询(含块号会话id的ask写入)。
- 001..075 块空间: 保留 kind∈{experience,experience_correction} 或
    (id 匹配 r"block-{blk}#\\d+" 且 text 以 "[" 开头)
  → 删除: 各块空间里 ask 自写入的 s#N 测试查询。
- default: 仅删除 request_id ∈ {rm1, rq1, v2, v2q}(人工核对过的4条部署验证记录),
    其余保留(防御性: 万一未来有真实数据)。
写入方式: 逐 space 原子替换(archive.jsonl.new → os.replace)。
删除清单逐条(id/request_id/text前80字)写入 UTF-8 报告 clean_report.json。
"""
import json
import os
import re
import sys

DELETE = "--delete" in sys.argv
MEMDATA = r"C:\bci_eval_isolated\memory_v2\memdata"
OUT = r"C:\bci_eval_isolated\memory_v2\clean_report.json"
DEFAULT_KILL_REQ = {"rm1", "rq1", "v2", "v2q"}   # default空间测试垃圾的request_id

report = {"mode": "delete" if DELETE else "dryrun", "spaces": {}}
tot_before = tot_del = 0

for space in sorted(os.listdir(MEMDATA)):
    arc = os.path.join(MEMDATA, space, "archive.jsonl")
    if not os.path.exists(arc):
        continue
    recs = [json.loads(l) for l in open(arc, encoding="utf-8").read()
            .splitlines() if l.strip()]
    keep, kill = [], []
    for r in recs:
        rid = str(r.get("id", ""))
        text = r.get("text") or ""
        req = str(r.get("request_id", ""))
        kind = r.get("kind")
        if kind in ("experience", "experience_correction"):
            ok = True                       # 经验记录一律保留
        elif space == "profile_all":
            m = re.fullmatch(r"(\d{3})#\d+", rid)
            ok = bool(m) and text.startswith(f"[{m.group(1)}/") \
                and req.startswith(f"{m.group(1)}-")
        elif re.fullmatch(r"\d{3}", space):
            ok = re.fullmatch(rf"block-{space}#\d+", rid) is not None \
                and text.startswith("[")
        elif space == "default":
            ok = req not in DEFAULT_KILL_REQ
        else:
            ok = True                       # 未知space不碰
        (keep if ok else kill).append(r)

    entry = {"before": len(recs), "delete": len(kill), "keep": len(keep),
             "kill_list": [{"id": r.get("id"), "req": r.get("request_id"),
                            "text": (r.get("text") or "")[:80]}
                           for r in kill]}
    report["spaces"][space] = entry
    tot_before += len(recs)
    tot_del += len(kill)

    if DELETE and kill:
        tmp = arc + ".new"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, arc)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

for sp, e in report["spaces"].items():
    if e["delete"]:
        print(f"{sp}: before={e['before']} delete={e['delete']} "
              f"keep={e['keep']}")
print(f"{'CLEAN_DONE' if DELETE else 'CLEAN_DRYRUN'} "
      f"total_before={tot_before} total_delete={tot_del}")
