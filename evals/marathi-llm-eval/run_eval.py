"""
Run the Marathi eval against any set of models.

    # 1. reading comprehension: same 900 questions in English, Marathi and romanized Marathi
    python run_eval.py mcq --models anthropic:claude-opus-5-5 openai:<model> gemini:<model> xai:<model> \
                                local:sarvamai/sarvam-m@http://localhost:8000/v1 --reverse
    # 2. open-ended answers to human-written Marathi prompts, with an automatic language check
    python run_eval.py generate --models ... --n 200
    # 3. LLM judges score those answers (use two judges from different providers to expose bias)
    python run_eval.py judge --judges anthropic:claude-opus-5-5 openai:<model>
    # 4. memorisation probe: can the model continue Belebele/FLORES passages verbatim?
    python run_eval.py contamination --models ... --n 100
    # then: python label.py (human labels) and python report.py

All requests go through a response cache (results/<run>/cache.sqlite), so reruns are free and an
interrupted run resumes where it stopped.
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import langid_mr  # noqa: E402
import marathi_data as data  # noqa: E402
from providers import CachedClient, make_client, slug  # noqa: E402

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "two or three sentences, written before the scores"},
        "helpfulness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "fluency": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "language": {"type": "string", "enum": ["marathi", "hindi", "english", "mixed"]},
    },
    "required": ["rationale", "helpfulness", "fluency", "language"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are a strict, fair evaluator of answers written for Marathi speakers.
Score the RESPONSE to the PROMPT. A human-written REFERENCE answer is given as a guide; other correct
answers are acceptable.

helpfulness (1-5): 5 = fully correct and complete, directly answers the prompt; 3 = partly correct
or incomplete; 1 = wrong, irrelevant or a refusal of a harmless request.
fluency (1-5): how natural and grammatical the Marathi is. 5 = reads like a fluent native writer;
3 = understandable but awkward or with errors; 1 = not Marathi, or barely readable. A response in Hindi
or English gets fluency 1 even if it is good Hindi or English.
language: the main language of the RESPONSE.
Write the rationale first, then the scores."""


def client_for(spec, args, run_dir):
    """Cached client; the cache key includes the settings that change outputs."""
    settings = json.dumps({"effort": args.effort, "temperature": args.temperature})
    return CachedClient(
        spec, make_client(spec, args.effort, args.temperature), os.path.join(run_dir, "cache.sqlite"), settings
    )


def run_parallel(fn, jobs, workers):
    """Map fn over jobs with a thread pool, preserving order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs))


def write_jsonl(path, rows):
    """Write rows as JSON lines (UTF-8, Devanagari kept readable)."""
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path):
    """Read JSON lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_mcq(args, run_dir):
    """Belebele in each condition (and reversed option order) for every model."""
    conds = data.build_conditions(
        data.load_belebele("eng_Latn", args.data_dir)[: args.n],
        data.load_belebele("mar_Deva", args.data_dir)[: args.n],
        romanize="mr_roman" in args.conditions,
    )
    jobs = [
        (c, item, rev)
        for c in args.conditions
        for item in conds[c]
        for rev in ([False, True] if args.reverse else [False])
    ]
    for spec in args.models:
        client = client_for(spec, args, run_dir)

        def one(job, client=client, spec=spec):
            cond, item, rev = job
            system, user, order = data.mcq_prompt(item, reverse=rev)
            r = client.complete(system, user, max_tokens=args.max_tokens)
            letter = data.parse_letter(r.text)
            pred = order[letter] if letter is not None else None
            return {
                "model": spec,
                "condition": cond,
                "reversed": rev,
                "id": item["id"],
                "pred": pred,
                "answer": item["answer"],
                "correct": int(pred == item["answer"]),
                "parsed": letter is not None,
                "refused": r.refused,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_s": r.latency_s,
                "raw": r.text[:200],
            }

        rows = run_parallel(one, jobs, args.workers)
        write_jsonl(os.path.join(run_dir, f"mcq_{slug(spec)}.jsonl"), rows)
        for c in args.conditions:
            sel = [x for x in rows if x["condition"] == c and not x["reversed"]]
            print(
                f"{spec:40s} {c:9s} accuracy {sum(x['correct'] for x in sel) / max(1, len(sel)):.3f}  "
                f"errors {sum(x['error'] is not None for x in sel)}"
            )


def cmd_generate(args, run_dir):
    """Answers to human-written Marathi prompts (sent as-is, no system prompt) + automatic language label."""
    items = data.load_aya_marathi(args.n, args.seed, args.aya_file)
    for spec in args.models:
        client = client_for(spec, args, run_dir)

        def one(item, client=client, spec=spec):
            r = client.complete(None, item["prompt"], max_tokens=args.max_tokens)
            label, details = langid_mr.classify(r.text)
            return {
                "model": spec,
                **item,
                "response": r.text,
                "lang_label": label,
                **details,
                "refused": r.refused,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_s": r.latency_s,
            }

        rows = run_parallel(one, items, args.workers)
        write_jsonl(os.path.join(run_dir, f"gen_{slug(spec)}.jsonl"), rows)
        counts = {k: sum(x["lang_label"] == k for x in rows) for k in ("marathi", "hindi", "english", "mixed", "empty")}
        print(f"{spec:40s} language labels {counts}")


def judge_prompt(row):
    """The judge sees the prompt, the human reference and the response (never the model's name)."""
    return f"PROMPT:\n{row['prompt']}\n\nREFERENCE:\n{row['reference']}\n\nRESPONSE:\n{row['response']}"


def parse_judgement(text):
    """Scores from the judge's JSON (structured outputs make this reliable; tolerate stray text anyway)."""
    start, end = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[start : end + 1])
        return {k: obj[k] for k in ("helpfulness", "fluency", "language", "rationale")}
    except (ValueError, KeyError):
        return None


def cmd_judge(args, run_dir):
    """Every judge scores every generation file in the run directory."""
    gen_files = sorted(f for f in os.listdir(run_dir) if f.startswith("gen_") and f.endswith(".jsonl"))
    for judge in args.judges:
        client = client_for(judge, args, run_dir)
        for name in gen_files:
            rows = [r for r in read_jsonl(os.path.join(run_dir, name)) if not r["error"]]

            def one(row, client=client):
                r = client.complete(
                    JUDGE_SYSTEM, judge_prompt(row), max_tokens=args.max_tokens, json_schema=JUDGE_SCHEMA
                )
                return {
                    "id": row["id"],
                    "model": row["model"],
                    "judge": client.spec,
                    **(parse_judgement(r.text) or {}),
                    "judge_error": r.error,
                    "judge_parsed": parse_judgement(r.text) is not None,
                }

            out = run_parallel(one, rows, args.workers)
            write_jsonl(os.path.join(run_dir, f"judge_{slug(judge)}__{name[4:]}"), out)
            ok = [o for o in out if o["judge_parsed"]]
            if ok:
                print(
                    f"judge {judge} on {name}: helpfulness {sum(o['helpfulness'] for o in ok) / len(ok):.2f}  "
                    f"fluency {sum(o['fluency'] for o in ok) / len(ok):.2f}"
                )


CONTAMINATION_SYSTEM = (
    "Continue the following text exactly as it appears in its original source. "
    "Output only the continuation, with no commentary."
)


def char_ngram_recall(generated, reference, n=5):
    """Share of the reference's character n-grams that also occur in the generated text (multiset overlap)."""
    ref = Counter(reference[i : i + n] for i in range(len(reference) - n + 1))
    gen = Counter(generated[i : i + n] for i in range(len(generated) - n + 1))
    return sum((ref & gen).values()) / max(1, sum(ref.values()))


def cmd_contamination(args, run_dir):
    """Prompt with the first half of each unique passage; compare the continuation with the true second half
    and, as a chance baseline, with the second half of a different passage."""
    passages = []
    for lang in ("eng_Latn", "mar_Deva"):
        unique = list(dict.fromkeys(it["passage"] for it in data.load_belebele(lang, args.data_dir)))[: args.n or 100]
        passages += [(lang, p) for p in unique]
    halves = []
    for lang, p in passages:
        words = p.split()
        halves.append((lang, " ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2 :])))
    for spec in args.models:
        client = client_for(spec, args, run_dir)

        def one(job, client=client, spec=spec):
            i, (lang, head, tail) = job
            r = client.complete(CONTAMINATION_SYSTEM, head, max_tokens=args.max_tokens)
            other = halves[(i + 1) % len(halves)][2]  # a different passage's continuation
            gen = r.text[: 2 * len(tail)]
            return {
                "model": spec,
                "lang": lang,
                "overlap": char_ngram_recall(gen, tail),
                "baseline": char_ngram_recall(gen, other),
                "refused": r.refused,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }

        rows = run_parallel(one, list(enumerate(halves)), args.workers)
        write_jsonl(os.path.join(run_dir, f"contam_{slug(spec)}.jsonl"), rows)
        ok = [x for x in rows if not x["error"]]
        print(
            f"{spec:40s} continuation overlap {sum(x['overlap'] for x in ok) / max(1, len(ok)):.3f}  "
            f"(chance {sum(x['baseline'] for x in ok) / max(1, len(ok)):.3f})"
        )


def main(argv=None):
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["mcq", "generate", "judge", "contamination"])
    p.add_argument("--models", nargs="*", default=["anthropic:claude-opus-5-5"])
    p.add_argument("--judges", nargs="*", default=["anthropic:claude-opus-5-5"])
    p.add_argument("--run", default="main", help="results/<run>/")
    p.add_argument("--n", type=int, default=None, help="limit items (default: all 900 MCQ / 200 prompts)")
    p.add_argument("--conditions", nargs="*", default=["en", "mr", "mr_roman"])
    p.add_argument("--reverse", action="store_true", help="also ask with the options in reversed order")
    p.add_argument("--data_dir", default=None, help="offline Belebele jsonl files")
    p.add_argument("--aya_file", default=None, help="offline Aya jsonl (inputs/targets/language fields)")
    p.add_argument("--effort", default="medium", help="Claude output_config.effort")
    p.add_argument("--temperature", type=float, default=None, help="only sent to OpenAI-compatible providers")
    p.add_argument("--max_tokens", type=int, default=16000, help="includes thinking tokens for reasoning models")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    if args.cmd == "generate" and args.n is None:
        args.n = 200
    run_dir = os.path.join(args.out_dir, args.run)
    os.makedirs(run_dir, exist_ok=True)
    {"mcq": cmd_mcq, "generate": cmd_generate, "judge": cmd_judge, "contamination": cmd_contamination}[args.cmd](
        args, run_dir
    )
    return run_dir


if __name__ == "__main__":
    main()
