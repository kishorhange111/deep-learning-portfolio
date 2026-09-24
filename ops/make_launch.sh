# Start a full training run of some projects on a session.
# Usage: bash ops/make_launch.sh <session> "<project prefixes>"   e.g.  bash ops/make_launch.sh keras2 "01 02"
set -e
S=$1
ARGS=$2
cat > /tmp/launch_$S.py <<EOF
import subprocess
subprocess.Popen("cd /content/dlp && nohup python run_all.py $ARGS > run_all.log 2>&1 &", shell=True)
print("launched on $S: $ARGS")
EOF
~/.local/bin/colab exec -s $S -f /tmp/launch_$S.py
