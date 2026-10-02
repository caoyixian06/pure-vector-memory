# -*- coding: utf-8 -*-
"""备份 memdata 全部 space 数据到带时间戳目录, 生成清单(文件+大小)。只读操作。"""
import json
import os
import shutil
import time

SRC = r"C:\bci_eval_isolated\memory_v2\memdata"
ROOT = r"C:\bci_eval_isolated\memory_v2"
TS = time.strftime("%Y%m%d_%H%M%S")
DST = os.path.join(ROOT, f"_backup_{TS}")

shutil.copytree(SRC, DST)

manifest = []
for root, dirs, files in os.walk(DST):
    for f in sorted(files):
        p = os.path.join(root, f)
        manifest.append({"file": p.replace(ROOT + os.sep, ""),
                         "bytes": os.path.getsize(p)})
info = {"src": SRC, "dst": DST, "ts": TS, "n_files": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "files": manifest}
mpath = os.path.join(ROOT, f"_backup_{TS}_manifest.json")
with open(mpath, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=1)
print(f"BACKUP_OK dst={DST} files={len(manifest)} "
      f"total_bytes={info['total_bytes']}")
