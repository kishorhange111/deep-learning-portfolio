# Show runner status, the tail of the current project's log and GPU usage.
import glob
import os
import re
import subprocess


def sh(cmd):
    out = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode("utf-8", "replace")
    return re.sub(r"\x1b\[[0-9;]*m", "", out)  # strip terminal colour codes


print(sh("cat /content/dlp/run_all.log"))
logs = sorted(glob.glob("/content/dlp/logs/*.log"), key=os.path.getmtime)
if logs:
    print(f"--- {os.path.basename(logs[-1])} ---")
    print(sh(f"tail -c 2500 {logs[-1]}"))
print(sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"))
