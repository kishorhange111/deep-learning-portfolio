# ops — running the projects on Google Colab GPUs from a terminal

These scripts drive Colab runtimes with the [Colab CLI](https://github.com/googlecolab/google-colab-cli)
(`colab new / upload / exec / download / stop`), so training runs on cloud GPUs without opening a notebook.

| script | runs on | what it does |
|---|---|---|
| `push.sh <session>` | local | tar the repo, upload it to a session, unpack it and print library versions (`setup_vm.py`) |
| `make_launch.sh <session> "03 05"` | local | start `run_all.py` for the given projects in the background on the VM |
| `launch_quick.py` | VM (`colab exec -f`) | smoke-test every pipeline with `QUICK=1` |
| `progress.py` | VM | runner status, tail of the current project's log, GPU utilisation |
| `summary.py` | VM | exit codes, error tails of failed projects, all `metrics.json` |
| `ps.py` | VM | running training processes |
| `kill.py` | VM | stop processes by PID (`KILL_PATTERN=...`) |
| `pack_results.py` | VM | bundle `results/` + `logs/` for `colab download` |

Typical workflow (several sessions in parallel, one group of projects each):
```bash
colab new -s gpu1 --gpu L4
bash ops/push.sh gpu1
bash ops/make_launch.sh gpu1 "04 06 08"
colab exec -s gpu1 -f ops/progress.py        # check in
colab exec -s gpu1 -f ops/pack_results.py && colab download -s gpu1 /content/results.tgz .
colab stop -s gpu1                            # stop billing
```
