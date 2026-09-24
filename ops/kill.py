# Stop processes on the VM by PID. (Matching by command line with `pkill -f` inside a
# shell can kill that shell itself, because its own command line contains the pattern.)
import os
import signal
import subprocess
import sys

PATTERN = os.environ.get("KILL_PATTERN", "09_gpt2_lora_finetuning/train.py")
me = os.getpid()
out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout.splitlines()[1:]
for line in out:
    pid, cmd = line.strip().split(" ", 1)
    if PATTERN in cmd and int(pid) != me and "ps -eo" not in cmd:
        os.kill(int(pid), signal.SIGKILL)
        print("killed", pid, cmd)
