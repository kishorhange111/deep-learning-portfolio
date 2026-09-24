# Runner summary + the error tail of any failed project.
import glob
import json
import os
import re
import subprocess

sh = lambda c: re.sub(r"\x1b\[[0-9;]*m", "", subprocess.run(c, shell=True, capture_output=True).stdout.decode("utf-8", "replace"))
print(sh("cat /content/dlp/run_all.log"))
for root in ["/content/dlp/results", "/content/dlp/results_quick"]:
    path = os.path.join(root, "run_summary.json")
    if os.path.exists(path):
        s = json.load(open(path))
        for proj, info in s.items():
            if info["exit_code"] != 0:
                print(f"### {proj} FAILED - log tail:")
                print(sh(f"grep -v 'examples/s\\|it/s\\|B/s' /content/dlp/logs/{proj}.log | tail -25"))
for m in sorted(glob.glob("/content/dlp/results*/*/metrics.json")):
    print(m.split("/content/dlp/")[1], open(m).read().replace("\n", " "))
