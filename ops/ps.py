import subprocess

sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print(sh("ps -eo pid,etimes,cmd | grep -E 'run_all|train.py' | grep -v grep"))
print(sh("cat /content/dlp/run_all.log"))
print(sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader"))
