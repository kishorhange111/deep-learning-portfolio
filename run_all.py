"""
Run every project one after another, each in its own Python process.

    python run_all.py                 # all projects
    python run_all.py 03 07           # only projects whose folder starts with 03 or 07
    QUICK=1 python run_all.py         # tiny smoke test of every pipeline (~minutes)

A failure in one project is logged and the runner moves on to the next.
Logs go to logs/<project>.log, a summary to results/run_summary.json.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS = sorted(d for d in os.listdir(ROOT) if d[:2].isdigit() and os.path.isdir(os.path.join(ROOT, d)))


def main():
    wanted = sys.argv[1:]
    todo = [p for p in PROJECTS if not wanted or any(p.startswith(w) for w in wanted)]
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2")

    summary = {}
    for project in todo:
        start = time.time()
        print(f"=== {project} ===", flush=True)
        with open(os.path.join(ROOT, "logs", f"{project}.log"), "w") as log:
            code = subprocess.call([sys.executable, os.path.join(project, "train.py")],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        summary[project] = {"exit_code": code, "minutes": round((time.time() - start) / 60, 1)}
        print(f"    -> exit {code} in {summary[project]['minutes']} min", flush=True)
        with open(os.path.join(ROOT, "results", "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
