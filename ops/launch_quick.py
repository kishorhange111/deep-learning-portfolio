# Smoke-test every pipeline on tiny data (results go to a separate folder).
import subprocess

subprocess.Popen("cd /content/dlp && QUICK=1 RESULTS_ROOT=/content/dlp/results_quick "
                 "nohup python run_all.py > run_all.log 2>&1 &", shell=True)
print("launched quick run")
