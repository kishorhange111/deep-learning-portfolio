# (runs on the VM) Bundle results + logs for download.
import subprocess

print(subprocess.run("cd /content/dlp && tar -czf /content/results.tgz results logs && ls -la /content/results.tgz",
                     shell=True, capture_output=True, text=True).stdout)
