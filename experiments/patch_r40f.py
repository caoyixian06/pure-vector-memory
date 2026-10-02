# -*- coding: utf-8 -*-
"""给 r40af/r40bf 全量runner打补丁: 断点续跑+防崩+改名+判官并发"""
import io, sys

def patch(src, trk):
    t = io.open(src, encoding="utf-8").read()
    n0 = t

    # ---- 1) 断点续跑: 载入块 + 把 检索预计算..rerankdone 整块套进 for _fresh ----
    pre = '''# ===== 断点续跑: 检索终序与答题模型无关, af的ckpt bf可复用 =====
_CKPT = HERE + "/r40%s_ckpt.npz"
_SHARE = HERE + "/r40af_ckpt.npz"
_have_ckpt = False
for _cand in (_CKPT, _SHARE):
    if os.path.exists(_cand):
        try:
            _z = np.load(_cand)
            if list(_z["IDS"]) == [q["qa_id"] for q in todo]:
                FINAL = list(_z["FINAL"])
                _have_ckpt = True
                print("ckpt loaded:", _cand, flush=True)
                break
        except Exception as _e:
            print("ckpt bad:", _cand, repr(_e)[:80], flush=True)
print("ckpt hit:", _have_ckpt, flush=True)
for _fresh in ([] if _have_ckpt else [0]):
''' % trk
    A = "# ===== 检索预计算 ====="
    B = 'print("rerank+atom+delta done %.0fs" % (time.time() - t0), flush=True)'
    ia = t.index(A)
    ib = t.index(B) + len(B)
    block = t[ia:ib]
    block = "\n".join(("    " + ln if ln.strip() else ln) for ln in block.split("\n"))
    t = t[:ia] + pre + block + "\n" + '''
if not _have_ckpt:
    np.savez_compressed(_CKPT, FINAL=np.stack(FINAL).astype(np.float32),
                        IDS=np.array([q["qa_id"] for q in todo]))
    print("ckpt saved %.0fs" % (time.time() - t0), flush=True)
''' + t[ib:]

    # ---- 2) work() 防崩 + 心跳 ----
    assert t.count("def work(q):") == 1
    t = t.replace("def work(q):", "def _work_inner(q):")
    assert t.count('    i = IDX[q["qa_id"]]\n') == 1
    t = t.replace('    i = IDX[q["qa_id"]]\n',
                  '    print("ANS", q["qa_id"], flush=True)\n    i = IDX[q["qa_id"]]\n')
    anchor = "t0 = time.time()\nwith ThreadPoolExecutor(4) as ex:"
    assert t.count(anchor) == 1
    wrap = '''def work(q):
    try:
        return _work_inner(q)
    except Exception as e:
        print("WORK_FAIL", q["qa_id"], repr(e)[:120], flush=True)
        return dict(qa_id=q["qa_id"], predicted_answer="(answering error)", second_pass=False)

'''
    t = t.replace(anchor, wrap + anchor)

    # ---- 3) 输出改名 ----
    t = t.replace('"/submission_r40.jsonl"', '"/submission_r40%s.jsonl"' % trk)
    t = t.replace('"/questions_r40.jsonl"', '"/questions_r40%s.jsonl"' % trk)
    t = t.replace('"/r40_score.txt"', '"/r40%s_score.txt"' % trk)
    t = t.replace('"QUESTIONS_WRITTEN r39"', '"QUESTIONS_WRITTEN r40%s"' % trk)
    t = t.replace('"R40_OFFICIAL_QWEN_SCORE"', '"R40%s_OFFICIAL_QWEN_SCORE"' % trk)
    t = t.replace('"R40 %d/%d = %.1f%%\\n"', '"R40' + trk.upper() + ' %d/%d = %.1f%%\\n"')
    t = t.replace('"R40_ALL_DONE"', '"R40%s_ALL_DONE"' % trk)

    # ---- 4) 判官并发4 ----
    assert t.count("scores = [judge(r) for r in submission]") == 1
    t = t.replace("scores = [judge(r) for r in submission]",
                  '''_js = {"n": 0}
def _judge_tagged(r):
    s = judge(r)
    _js["n"] += 1
    if _js["n"] % 200 == 0:
        print("judge", _js["n"], "/", len(submission), flush=True)
    return s
with ThreadPoolExecutor(4) as jex:
    scores = list(jex.map(_judge_tagged, submission))
print("JUDGE_DONE", flush=True)''')
    assert t != n0
    io.open(src.replace(".py", "_p.py"), "w", encoding="utf-8").write(t)
    print("patched", trk, "OK")

patch(sys.argv[1], sys.argv[2])
