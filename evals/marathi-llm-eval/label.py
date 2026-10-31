"""
Collect human labels for a random sample of generated answers, to validate the LLM judges and the
automatic language check.

Items are drawn evenly across models and shuffled, and the model's name is never shown, so labels
are blind. Labels are appended to results/<run>/human_labels.jsonl after every item, so a session can
stop at any time and resume later. 100 items take about an hour for a fluent Marathi reader.

    python label.py --run main --n 100
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_eval import read_jsonl  # noqa: E402

LANG_KEYS = {"m": "marathi", "h": "hindi", "e": "english", "x": "mixed"}


def sample_items(run_dir, n, seed=0):
    """Up to n (model, id) items, balanced across models, in random order."""
    per_model = {}
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("gen_") and name.endswith(".jsonl"):
            rows = [r for r in read_jsonl(os.path.join(run_dir, name)) if not r["error"]]
            if rows:
                per_model[rows[0]["model"]] = rows
    rng = random.Random(seed)
    k = max(1, n // max(1, len(per_model)))
    items = [r for rows in per_model.values() for r in rng.sample(rows, min(k, len(rows)))]
    rng.shuffle(items)
    return items[:n]


def ask_int(prompt, read=input):
    """Ask until the answer is an integer from 1 to 5."""
    while True:
        answer = read(prompt).strip()
        if answer in {"1", "2", "3", "4", "5"}:
            return int(answer)


def ask_lang(read=input):
    """Ask until the answer is one of m/h/e/x."""
    while True:
        answer = read("language [m]arathi [h]indi [e]nglish mi[x]ed: ").strip().lower()
        if answer in LANG_KEYS:
            return LANG_KEYS[answer]


def label(run_dir, n, seed=0, read=input, show=print):
    """Interactive loop; returns the number of new labels."""
    path = os.path.join(run_dir, "human_labels.jsonl")
    done = {(r["model"], r["id"]) for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [it for it in sample_items(run_dir, n, seed) if (it["model"], it["id"]) not in done]
    show(f"{len(done)} already labelled, {len(todo)} to go. Scores: helpfulness and fluency, 1 (bad) - 5 (excellent).")
    added = 0
    for i, it in enumerate(todo, 1):
        show(f"\n===== {i}/{len(todo)} =====")
        show(f"PROMPT:\n{it['prompt']}\n\nREFERENCE:\n{it['reference']}\n\nRESPONSE:\n{it['response']}\n")
        rec = {
            "model": it["model"],
            "id": it["id"],
            "helpfulness": ask_int("helpfulness 1-5: ", read),
            "fluency": ask_int("fluency 1-5: ", read),
            "language": ask_lang(read),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        added += 1
    return added


def main(argv=None):
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="main")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    return label(os.path.join(args.out_dir, args.run), args.n, args.seed)


if __name__ == "__main__":
    main()
