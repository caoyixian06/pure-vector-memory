# -*- coding: utf-8 -*-
"""patch_af_pause.py — 给af runner加逐题答题断点+跳过内置软判官(省配额), 支持暂停/续跑"""
import io, sys

src = "r40af_runner_p.py"
t = io.open(src, encoding="utf-8").read()
n0 = t

# 1) 断点载入 + todo_run, 插在 IDX 定义后
anchor = 'IDX = {q["qa_id"]: i for i, q in enumerate(todo)}'
assert t.count(anchor) == 1
ins = anchor + '''

PART = HERE + "/submission_r40af_partial.jsonl"
done_p = {}
if os.path.exists(PART):
    for _l in open(PART, encoding="utf-8"):
        if _l.strip():
            try:
                _r = json.loads(_l)
                done_p[_r["qa_id"]] = _r
            except Exception:
                pass
todo_run = [q for q in todo if q["qa_id"] not in done_p]
print("resume: partial=%d todo_run=%d" % (len(done_p), len(todo_run)), flush=True)
PLOCK = threading.Lock()'''
t = t.replace(anchor, ins)

# 2) work() 逐题落盘
anchor2 = "    return dict(qa_id=q[\"qa_id\"], predicted_answer=final, second_pass=second)"
assert t.count(anchor2) == 1
t = t.replace(anchor2,
'''    _res = dict(qa_id=q["qa_id"], predicted_answer=final, second_pass=second)
    with PLOCK:
        with open(PART, "a", encoding="utf-8") as _pf:
            _pf.write(json.dumps(_res, ensure_ascii=False) + NL)
    return _res''')

# 3) 答题用 todo_run + 合并断点
anchor3 = "with ThreadPoolExecutor(4) as ex:\n    submission = list(ex.map(work, todo))"
assert t.count(anchor3) == 1
t = t.replace(anchor3,
"with ThreadPoolExecutor(4) as ex:\n    submission = list(done_p.values()) + list(ex.map(work, todo_run))")

# 4) submission落盘后: 默认跳过内置软判官(真gold重判离线做, 省配额)
anchor4 = 'print("QUESTIONS_WRITTEN r40af", flush=True)'
assert t.count(anchor4) == 1
t = t.replace(anchor4, anchor4 + '''
if os.environ.get("AF_JUDGE", "0") != "1":
    print("AF_SUBMISSION_ONLY_DONE (soft judge skipped)", flush=True)
    sys.exit(0)''')

assert t != n0
io.open(src, "w", encoding="utf-8").write(t)
print("patched OK")
