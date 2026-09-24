# Package the repo, upload it to a Colab session and unpack it there.
# Usage (from the repo root, inside WSL):  bash ops/push.sh <session>
set -e
S=${1:-keras}
C=~/.local/bin/colab
tar --exclude=results --exclude=results_quick --exclude=logs --exclude=.git -czf /tmp/dlp.tgz .
$C upload -s $S /tmp/dlp.tgz /content/dlp.tgz
$C exec -s $S -f ops/setup_vm.py
