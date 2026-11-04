"""
Why didn't GRPO improve accuracy? A controlled study of where the RL signal goes.

Background: in generative-ai-engineering/reinforcement-learning/grpo-math-reasoning, 307 steps of TRL's
GRPOTrainer on Qwen2.5-1.5B-Instruct took GSM8K format compliance from 55 % to 98 % but left accuracy
at 67.6 %. TRL reported that 10-30 % of groups had all 8 rewards identical. Those groups get zero
advantage, so they produce no gradient.

This script is a hand-written GRPO (see grpo.py) with switches for four interventions, each a
hypothesis about where the signal is lost:

  H1 dynamic sampling   --dynamic_sampling  drop zero-variance groups and refill the batch (DAPO)
  H2 curriculum         --curriculum FILE   train only on questions the base model sometimes solves
  H3 group size         --group_size 16     more samples per question -> fewer all-same groups
  H4 format reward      --format_weight 0   the cheap format reward may dominate the correctness signal

sweep.py runs every config with 3 seeds, and analyze.py compares them to the baseline with paired
bootstrap confidence intervals over test questions. Dynamic sampling generates more completions per
step, so the comparison is also made per generated sample, not just per step.

    python train.py --run_name baseline --seed 0                 # one run
    python sweep.py                                              # the full study (resumable)
    QUICK=1 python train.py --run_name smoke                     # 2-step pipeline check
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grpo  # noqa: E402
from sampling import HFSampler, VLLMSampler  # noqa: E402


def parse_args(argv=None):
    """Command-line options (defaults = the baseline config); QUICK=1 shrinks everything for a smoke test."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--run_name", default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results", "runs"))
    p.add_argument("--data_dir", default=None, help="local train.jsonl/test.jsonl instead of HF GSM8K")
    # GRPO
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--prompts_per_step", type=int, default=8)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--format_weight", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty; 0 skips the reference model")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--ppo_epochs", type=int, default=1, help="optimizer steps per sampled batch")
    p.add_argument("--eps_low", type=float, default=0.2)
    p.add_argument("--eps_high", type=float, default=0.2, help="DAPO 'clip-higher' uses 0.28")
    p.add_argument("--aggregation", choices=["token", "sequence"], default="token")
    p.add_argument("--no_std_scaling", action="store_true", help="Dr. GRPO advantages")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # interventions
    p.add_argument("--dynamic_sampling", action="store_true")
    p.add_argument("--oversample", type=float, default=2.0, help="prompts drawn per missing group")
    p.add_argument("--max_sample_rounds", type=int, default=4)
    p.add_argument("--curriculum", default=None, help="pass-rate JSON written by curriculum.py")
    p.add_argument("--pass_min", type=float, default=0.0, help="keep questions with pass_min < p < pass_max")
    p.add_argument("--pass_max", type=float, default=1.0)
    # sampling / evaluation
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm" if torch.cuda.is_available() else "hf")
    p.add_argument("--vllm_mem", type=float, default=0.3)
    p.add_argument("--eval_n", type=int, default=1319, help="GSM8K test questions (1319 = all)")
    p.add_argument("--eval_before", action="store_true", help="also evaluate the starting model")
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--passk_n", type=int, default=0, help="samples per question for pass@k (0 = skip)")
    p.add_argument("--passk_questions", type=int, default=300)
    p.add_argument("--time_budget_min", type=float, default=120)
    p.add_argument("--save_model", action="store_true")
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.steps, args.prompts_per_step, args.group_size = 2, 2, 4
        args.max_new_tokens, args.eval_n, args.passk_questions = 16, 4, 2
    return args


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def load_split(args, split):
    """[{'question', 'gold'}] from GSM8K (or local jsonl files with GSM8K's question/answer fields)."""
    if args.data_dir:
        with open(os.path.join(args.data_dir, f"{split}.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "gold": grpo.gsm8k_gold(r["answer"])} for r in rows]


def add_prompt_ids(items, tokenizer):
    """Attach chat-formatted prompt token ids (system prompt + question) to every item."""
    for it in items:
        messages = [{"role": "system", "content": grpo.SYSTEM_PROMPT}, {"role": "user", "content": it["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        it["prompt_ids"] = tokenizer(text, add_special_tokens=False)["input_ids"]
    return items


def apply_curriculum(items, args):
    """H2: keep only questions whose estimated pass rate lies strictly between pass_min and pass_max."""
    with open(args.curriculum) as f:
        pass_rates = json.load(f)["pass_rates"]
    kept = [it for i, it in enumerate(items) if args.pass_min < pass_rates[i] < args.pass_max]
    print(f"[curriculum] {len(kept)}/{len(items)} questions with {args.pass_min} < pass rate < {args.pass_max}")
    return kept


def endless(items, seed):
    """Yield items forever, reshuffled every epoch with a seeded RNG."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            yield items[i]


def collate(prompts, completions, pad_id):
    """[left-padded prompt | right-padded completion] -> input_ids, attention_mask, completion_mask."""
    P, C = max(map(len, prompts)), max(map(len, completions))
    ids = torch.full((len(prompts), P + C), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    comp_mask = torch.zeros((len(prompts), C), dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        ids[i, P - len(p) : P] = torch.tensor(p)
        ids[i, P : P + len(c)] = torch.tensor(c)
        attn[i, P - len(p) : P + len(c)] = 1
        comp_mask[i, : len(c)] = 1
    return ids, attn, comp_mask, C


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------
def load_models(args, device):
    """Policy (fp32, trainable), frozen reference model (only if beta > 0) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 master weights + bf16 autocast. With lr = 1e-6, an update is ~1e-6 of a weight's size, far
    # below bf16's resolution (8 mantissa bits, ~4e-3 relative): pure-bf16 weights would not move.
    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    policy.config.use_cache = False
    if device.type == "cuda":
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref = None
    if args.beta > 0:
        ref_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=ref_dtype).to(device).eval()
        ref.requires_grad_(False)
    return policy, ref, tokenizer


def make_sampler(args, policy, tokenizer):
    """vLLM engine (fast, own weight copy) or the policy itself via Hugging Face generate()."""
    if args.backend == "vllm":
        return VLLMSampler(args.model, tokenizer, args.max_new_tokens, args.vllm_mem, seed=args.seed)
    return HFSampler(policy, tokenizer, args.max_new_tokens)


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------
def evaluate(sampler, items, n_keep_samples=20):
    """Greedy decoding: answer accuracy and strict-format compliance, per question (for paired tests)."""
    outs = sampler.generate([it["prompt_ids"] for it in items], n=1, greedy=True)
    correct, fmt, samples = [], [], []
    for it, (c,) in zip(items, outs):
        correct.append(int(grpo.extract_answer(c.text) == it["gold"]))
        fmt.append(int(grpo.format_ok(c.text)))
        if len(samples) < n_keep_samples:
            samples.append({"question": it["question"], "gold": it["gold"], "completion": c.text})
    return {
        "accuracy": float(np.mean(correct)),
        "format": float(np.mean(fmt)),
        "n": len(items),
        "correct": correct,
        "format_ok": fmt,
        "samples": samples,
    }


def evaluate_pass_at_k(sampler, items, n, seed, ks=(1, 2, 4, 8, 16, 32, 64)):
    """pass@k with n samples at temperature 1. Does RL teach new solutions, or only make the model pick
    solutions it could already sample more reliably? If RL only sharpens, pass@1 rises while large-k
    pass@k stays flat or falls (Yue et al., 2025)."""
    outs = sampler.generate([it["prompt_ids"] for it in items], n=n, temperature=1.0, seed=seed)
    counts = [sum(grpo.extract_answer(c.text) == it["gold"] for c in group) for it, group in zip(items, outs)]
    return {
        "n": n,
        "questions": len(items),
        "correct_counts": counts,
        "pass_at_k": {k: float(np.mean([grpo.pass_at_k(n, c, k) for c in counts])) for k in ks if k <= n},
    }


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------
def sample_groups(sampler, stream, args, step):
    """Sample prompts_per_step groups. With dynamic sampling, groups whose rewards are all equal are
    dropped and replaced by fresh prompts, up to max_sample_rounds rounds."""
    kept, log = [], defaultdict(float)
    all_rewards, all_correct, all_fmt, lengths, truncated = [], [], [], [], []
    for round_ in range(args.max_sample_rounds if args.dynamic_sampling else 1):
        need = args.prompts_per_step - len(kept)
        if need <= 0:
            break
        n_draw = math.ceil(need * args.oversample) if args.dynamic_sampling else need
        items = [next(stream) for _ in range(n_draw)]
        outs = sampler.generate(
            [it["prompt_ids"] for it in items],
            n=args.group_size,
            temperature=args.temperature,
            seed=args.seed * 1_000_003 + step * 101 + round_,
        )
        for it, comps in zip(items, outs):
            rewards, correct, fmt = grpo.compute_rewards([c.text for c in comps], it["gold"], args.format_weight)
            all_rewards += rewards
            all_correct += correct
            all_fmt += fmt
            lengths += [len(c.token_ids) for c in comps]
            truncated += [c.truncated for c in comps]
            informative = max(rewards) > min(rewards)
            log["groups_generated"] += 1
            log["zero_std_groups"] += not informative
            log["all_correct_groups"] += min(correct) == 1
            log["all_wrong_groups"] += max(correct) == 0
            if args.dynamic_sampling and not informative:
                continue
            kept.append({"item": it, "completions": comps, "rewards": rewards})
        log["sample_rounds"] = round_ + 1
    kept = kept[: args.prompts_per_step]
    g = max(log["groups_generated"], 1)
    log.update(
        {
            "reward": float(np.mean(all_rewards)),
            "correct_reward": float(np.mean(all_correct)),
            "format_reward": float(np.mean(all_fmt)),
            "completion_len": float(np.mean(lengths)),
            "truncated_frac": float(np.mean(truncated)),
            "zero_std_frac": log["zero_std_groups"] / g,
            "all_correct_frac": log["all_correct_groups"] / g,
            "all_wrong_frac": log["all_wrong_groups"] / g,
            "samples_generated": log["groups_generated"] * args.group_size,
            "groups_trained": len(kept),
        }
    )
    return kept, dict(log)


def policy_update(policy, ref, optimizer, scheduler, groups, args, pad_id, device):
    """One GRPO update on the kept groups: advantages, reference/old log-probs, accumulated gradient step."""
    prompts, completions, advantages = [], [], []
    for g in groups:
        adv = grpo.group_advantages(torch.tensor([g["rewards"]]), scale_by_std=not args.no_std_scaling)[0]
        for c, a in zip(g["completions"], adv.tolist()):
            prompts.append(g["item"]["prompt_ids"])
            completions.append(c.token_ids if c.token_ids else [pad_id])
            advantages.append(a)
    total_tokens = sum(len(c) for c in completions)
    total_seqs = len(completions)

    def amp():
        """bf16 autocast on GPU, a no-op on CPU."""
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    micro = []
    for s in range(0, total_seqs, args.micro_batch_size):
        ids, attn, cmask, C = collate(
            prompts[s : s + args.micro_batch_size], completions[s : s + args.micro_batch_size], pad_id
        )
        mb = {
            "ids": ids.to(device),
            "attn": attn.to(device),
            "cmask": cmask.to(device),
            "C": C,
            "adv": torch.tensor(advantages[s : s + args.micro_batch_size], device=device),
        }
        with torch.no_grad(), amp():
            if ref is not None:
                mb["ref_logp"] = grpo.completion_logprobs(ref, mb["ids"], mb["attn"], C)
            if args.ppo_epochs > 1:
                mb["old_logp"] = grpo.completion_logprobs(policy, mb["ids"], mb["attn"], C)
        micro.append(mb)

    stats = defaultdict(float)
    policy.train()
    for _ in range(args.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for mb in micro:
            with amp():
                logp = grpo.completion_logprobs(policy, mb["ids"], mb["attn"], mb["C"])
            old = mb.get("old_logp", logp.detach())
            loss, st = grpo.grpo_loss(
                logp,
                old,
                mb["adv"],
                mb["cmask"],
                mb.get("ref_logp"),
                args.beta,
                args.eps_low,
                args.eps_high,
                args.aggregation,
            )
            # rescale micro-batch means so that gradient accumulation equals one big batch
            n_tok, n_seq = mb["cmask"].sum().item(), mb["ids"].shape[0]
            weight = n_tok / total_tokens if args.aggregation == "token" else n_seq / total_seqs
            (loss * weight).backward()
            stats["loss"] += loss.item() * weight / args.ppo_epochs
            for k, v in st.items():
                stats[k] += v * weight / args.ppo_epochs
        stats["grad_norm"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip).item()
        optimizer.step()
    scheduler.step()
    return dict(stats)


def set_seed(seed):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path):
    """Write obj as indented JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def train(args):
    """Full run: optional base eval -> GRPO steps (logged per step) -> final eval (+ pass@k) -> results."""
    set_seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.run_name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, ref, tokenizer = load_models(args, device)
    train_items = add_prompt_ids(load_split(args, "train"), tokenizer)
    test_items = add_prompt_ids(load_split(args, "test"), tokenizer)[: args.eval_n]
    if args.curriculum:
        train_items = apply_curriculum(train_items, args)
    stream = endless(train_items, args.seed)
    sampler = make_sampler(args, policy, tokenizer)

    if args.eval_before:
        base = evaluate(sampler, test_items)
        print(f"[eval before] accuracy {base['accuracy']:.3f}  format {base['format']:.3f}")
        save_json(base, os.path.join(run_dir, "eval_before.json"))
        if args.passk_n:
            save_json(
                evaluate_pass_at_k(sampler, test_items[: args.passk_questions], args.passk_n, args.seed),
                os.path.join(run_dir, "passk_before.json"),
            )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start, samples_total, step = time.time(), 0, 0
    log_path = os.path.join(run_dir, "train_log.jsonl")
    with open(log_path, "w") as log_file:
        for step in range(1, args.steps + 1):
            t0 = time.time()
            groups, log = sample_groups(sampler, stream, args, step)
            t_gen = time.time() - t0
            samples_total += log["samples_generated"]
            if groups:
                log.update(
                    policy_update(policy, ref, optimizer, scheduler, groups, args, tokenizer.pad_token_id, device)
                )
                sampler.sync(policy)
            log.update(
                {
                    "step": step,
                    "samples_total": samples_total,
                    "gen_time": t_gen,
                    "step_time": time.time() - t0,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            if args.eval_every and step % args.eval_every == 0:
                mid = evaluate(sampler, test_items[:200])
                log.update({"eval_accuracy": mid["accuracy"], "eval_format": mid["format"]})
            log_file.write(json.dumps(log) + "\n")
            log_file.flush()
            if step % 10 == 0 or step == 1:
                print(
                    f"step {step:4d}  reward {log['reward']:.3f}  correct {log['correct_reward']:.3f}  "
                    f"format {log['format_reward']:.3f}  zero-std {log['zero_std_frac']:.2f}  "
                    f"len {log['completion_len']:.0f}  {log['step_time']:.1f}s"
                )
            if time.time() - start > args.time_budget_min * 60:
                print(f"[time budget] {args.time_budget_min} min reached at step {step}")
                break

    final = evaluate(sampler, test_items)
    final.update({"steps": step, "samples_total": samples_total, "train_minutes": (time.time() - start) / 60})
    print(f"[eval after] accuracy {final['accuracy']:.3f}  format {final['format']:.3f}")
    save_json(final, os.path.join(run_dir, "eval_final.json"))
    if args.passk_n:
        save_json(
            evaluate_pass_at_k(sampler, test_items[: args.passk_questions], args.passk_n, args.seed),
            os.path.join(run_dir, "passk_final.json"),
        )
    if args.save_model:
        policy.save_pretrained(os.path.join(run_dir, "model"))
        tokenizer.save_pretrained(os.path.join(run_dir, "model"))
    return final


def main(argv=None):
    """Entry point: parse arguments and run."""
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
