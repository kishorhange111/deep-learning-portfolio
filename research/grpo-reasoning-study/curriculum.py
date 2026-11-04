"""
H2 - curriculum: estimate how often the *starting* model solves each training question.

GRPO only learns from questions the model gets right sometimes but not always. A question solved
8/8 times or 0/8 times gives every completion the same reward, so it produces no gradient. This script
samples k completions per GSM8K training question and writes the pass rates. train.py --curriculum
then keeps only questions with 0 < pass rate < 1.

The estimate is noisy: a question that went 0/8 may still have a true pass rate of a few percent.
k=8 matches the group size used in training, so "0/8 here" means "likely a zero-gradient group there".

    python curriculum.py --k 8          # ~10 min with vLLM for all 7,473 questions on an A100
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grpo  # noqa: E402
import train  # noqa: E402


def estimate_pass_rates(sampler, items, k, seed):
    outs = sampler.generate([it["prompt_ids"] for it in items], n=k, temperature=1.0, seed=seed)
    return [
        float(np.mean([grpo.extract_answer(c.text) == it["gold"] for c in group])) for it, group in zip(items, outs)
    ]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm" if torch.cuda.is_available() else "hf")
    p.add_argument("--vllm_mem", type=float, default=0.85)
    p.add_argument("--out", default=os.path.join(HERE, "results", "pass_rates.json"))
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.backend == "vllm":
        sampler = train.VLLMSampler(args.model, tokenizer, args.max_new_tokens, args.vllm_mem, seed=args.seed)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        sampler = train.HFSampler(model, tokenizer, args.max_new_tokens)

    items = train.add_prompt_ids(train.load_split(args, "train"), tokenizer)[: args.limit]
    rates = estimate_pass_rates(sampler, items, args.k, args.seed)
    rates_arr = np.array(rates)
    summary = {
        "model": args.model,
        "k": args.k,
        "questions": len(rates),
        "always_solved": float(np.mean(rates_arr == 1)),
        "never_solved": float(np.mean(rates_arr == 0)),
        "mixed": float(np.mean((rates_arr > 0) & (rates_arr < 1))),
        "mean_pass_rate": float(rates_arr.mean()),
    }
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "pass_rates": rates}, f)


if __name__ == "__main__":
    main()
