"""
Pretrain a GPT on FineWeb-Edu on one GPU or many, with DDP or FSDP, and measure how well the
hardware is used.

One run trains one model for a fixed budget: compute (--flops), tokens (--tokens) or tokens per
parameter (--tokens_per_param, Chinchilla's ~20). scaling.py runs a grid of these to fit a scaling law.

Parallelism:
  none  single process
  ddp   DistributedDataParallel: every GPU holds the full model, gradients are all-reduced
  fsdp  FSDP2 (fully_shard): parameters, gradients and optimizer state are sharded across GPUs and
        gathered layer by layer. Memory per GPU falls with the number of GPUs, at the cost of
        extra communication.

Reported per step: loss, tokens/s and **MFU** (model FLOPs utilisation): the training FLOPs the
model mathematically needs (gpt.flops_per_token x tokens/s) divided by the GPUs' peak bf16 FLOPs.
Recomputation and padding are not credited.

    python train.py --preset 25M --tokens_per_param 20
    torchrun --nproc_per_node 8 train.py --preset 302M --parallel fsdp --flops 1e18
    QUICK=1 python train.py --synthetic        # CPU pipeline check, no data needed
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpt import GPT, PRESETS, config_for, configure_optimizer, flops_per_token, non_embedding_params  # noqa: E402

# Peak dense bf16 (fp16 for T4) tensor-core FLOP/s per GPU, from NVIDIA datasheets.
PEAK_FLOPS = {"H100": 989e12, "A100": 312e12, "L4": 121e12, "T4": 65e12}
# Peak learning rate per model size. These follow the downward trend of the GPT-3 paper's table and
# were NOT tuned per size here. Run `scaling.py lr-sweep` before trusting fitted exponents.
LR = {"3M": 2e-3, "11M": 1.5e-3, "25M": 1.2e-3, "49M": 1e-3, "85M": 8e-4, "201M": 5e-4, "302M": 4e-4}


def parse_args(argv=None):
    """Command-line options; QUICK=1 shrinks the run to a few CPU-sized steps."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="25M", choices=list(PRESETS))
    p.add_argument("--flops", type=float, default=None, help="training compute budget (FLOPs)")
    p.add_argument("--tokens", type=float, default=None, help="training token budget")
    p.add_argument("--tokens_per_param", type=float, default=20.0, help="used if neither budget is given")
    p.add_argument("--batch_tokens", type=int, default=2**18, help="global batch size in tokens")
    p.add_argument("--micro_batch", type=int, default=16, help="sequences per GPU per forward pass")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=None, help="peak LR (default: per-preset table)")
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--parallel", choices=["none", "ddp", "fsdp"], default="none")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data_dir", default=os.path.join(HERE, "data", "fineweb_edu"))
    p.add_argument("--synthetic", action="store_true", help="random-pattern data (tests, throughput checks)")
    p.add_argument("--max_steps", type=int, default=None, help="cap steps (throughput benchmarks)")
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=40)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--peak_tflops", type=float, default=None, help="override the GPU peak for MFU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results", "runs"))
    p.add_argument("--save_checkpoint", action="store_true")
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.seq_len, args.micro_batch, args.batch_tokens = 64, 4, 512
        args.max_steps = args.max_steps or 6
        args.eval_interval, args.eval_batches, args.log_interval, args.warmup_steps = 3, 2, 1, 2
    return args


# --------------------------------------------------------------------------------------------------
# Distributed setup
# --------------------------------------------------------------------------------------------------
def setup_distributed():
    """Initialise torch.distributed when launched by torchrun; returns (rank, local_rank, world, device)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, local_rank, world, device


def all_reduce_mean(value, world, device):
    """Mean of a Python float over all ranks."""
    if world == 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return t.item() / world


def peak_flops_per_gpu(device, override_tflops=None):
    """Datasheet peak for the detected GPU, or None (MFU not reported) on CPU/unknown GPUs."""
    if override_tflops:
        return override_tflops * 1e12
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in PEAK_FLOPS.items() if k in name), None)


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
class TokenWindows:
    """Random (seq_len + 1)-token windows from memory-mapped uint16 shards; x = w[:-1], y = w[1:].

    Each rank uses its own seeded RNG, so ranks see different data without any coordination. All runs
    in the sweep train on well under one epoch, so sampling windows with replacement behaves like a
    single pass."""

    def __init__(self, data_dir, split, seq_len, rank, seed):
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        self.shards = [np.memmap(os.path.join(data_dir, s["path"]), dtype=np.uint16, mode="r") for s in meta[split]]
        sizes = np.array([len(s) - seq_len - 1 for s in self.shards], dtype=np.float64)
        self.probs = sizes / sizes.sum()
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed * 1009 + rank)

    def batch(self, batch_size):
        """(x, y) int64 tensors of shape (batch_size, seq_len)."""
        windows = []
        for shard_idx in self.rng.choice(len(self.shards), size=batch_size, p=self.probs):
            shard = self.shards[shard_idx]
            start = self.rng.integers(0, len(shard) - self.seq_len - 1)
            windows.append(np.asarray(shard[start : start + self.seq_len + 1], dtype=np.int64))
        w = torch.from_numpy(np.stack(windows))
        return w[:, :-1], w[:, 1:]

    def fixed_batches(self, n_batches, batch_size, rank, world):
        """Deterministic validation windows (evenly spaced), split across ranks."""
        shard = self.shards[0]
        total = n_batches * batch_size * world
        starts = np.linspace(0, len(shard) - self.seq_len - 2, total).astype(np.int64)[rank::world]
        for i in range(n_batches):
            w = np.stack(
                [
                    np.asarray(shard[s : s + self.seq_len + 1], dtype=np.int64)
                    for s in starts[i * batch_size : (i + 1) * batch_size]
                ]
            )
            w = torch.from_numpy(w)
            yield w[:, :-1], w[:, 1:]


class SyntheticWindows:
    """Learnable synthetic data: sequences of a short repeating random pattern (for tests and benchmarks)."""

    def __init__(self, vocab_size, seq_len, rank, seed, period=8):
        self.vocab_size, self.seq_len, self.period = vocab_size, seq_len, period
        self.gen = torch.Generator().manual_seed(seed * 1009 + rank)

    def batch(self, batch_size):
        """Each row repeats its own random `period`-token pattern, so the model can learn to copy."""
        pattern = torch.randint(0, self.vocab_size, (batch_size, self.period), generator=self.gen)
        w = pattern.repeat(1, (self.seq_len + 1) // self.period + 1)[:, : self.seq_len + 1]
        return w[:, :-1], w[:, 1:]

    def fixed_batches(self, n_batches, batch_size, rank, world):
        """Validation batches from a separate fixed seed."""
        val = SyntheticWindows(self.vocab_size, self.seq_len, rank, seed=10_000, period=self.period)
        for _ in range(n_batches):
            yield val.batch(batch_size)


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------
def lr_at(step, max_lr, warmup, total, min_frac):
    """Linear warm-up, then cosine decay to min_frac * max_lr at the end of *this* run's budget.

    Matching the schedule length to the token budget matters: Chinchilla's revision of the Kaplan
    exponents came largely from this."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def wrap_model(model, args, device, world):
    """Apply DDP or FSDP2 (per-block sharding, bf16 compute / fp32 gradient reduction)."""
    if args.parallel == "ddp" and world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if args.parallel == "fsdp" and world > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh(device.type, (world,))
        mp = (
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if device.type == "cuda"
            else MixedPrecisionPolicy()
        )
        for block in model.blocks:  # shard block by block so only one block is ever fully gathered
            fully_shard(block, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def grad_sync_context(model, args, last_micro_step):
    """Skip the gradient all-reduce / reduce-scatter on all but the last micro-step of accumulation."""
    if args.parallel == "ddp" and hasattr(model, "no_sync") and not last_micro_step:
        return model.no_sync()
    if args.parallel == "fsdp" and hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(last_micro_step)
    return contextlib.nullcontext()


def full_state_dict(model, args, world):
    """Unsharded state dict on every rank (collective under FSDP)."""
    if args.parallel == "fsdp" and world > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    inner = model.module if hasattr(model, "module") else model
    return {k: v.detach().cpu() for k, v in inner.state_dict().items()}


@torch.no_grad()
def evaluate(model, data, args, rank, world, device, autocast):
    """Mean validation loss over fixed windows, averaged across ranks."""
    model.eval()
    losses = []
    for x, y in data.fixed_batches(args.eval_batches, args.micro_batch, rank, world):
        with autocast():
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return all_reduce_mean(float(np.mean(losses)), world, device)


def budget_tokens(args, cfg):
    """Training tokens implied by the budget flags."""
    if args.flops:
        return args.flops / flops_per_token(cfg)
    if args.tokens:
        return args.tokens
    return args.tokens_per_param * non_embedding_params(cfg)


def train(args):
    """One pretraining run; writes train_log.jsonl and metrics.json (and ckpt.pt if asked)."""
    rank, _, world, device = setup_distributed()
    torch.manual_seed(args.seed)  # identical initial weights on every rank
    vocab = 512 if args.synthetic and os.environ.get("QUICK") else 50304
    cfg = config_for(args.preset, seq_len=args.seq_len, vocab_size=vocab)
    model = GPT(cfg).to(device)

    per_step = args.micro_batch * args.seq_len * world
    accum = max(1, round(args.batch_tokens / per_step))
    batch_tokens = accum * per_step  # actual global batch after rounding to whole micro-batches
    tokens = budget_tokens(args, cfg)
    steps = max(1, math.ceil(tokens / batch_tokens))
    if args.max_steps:
        steps = min(steps, args.max_steps)
    max_lr = args.lr or LR[args.preset]
    warmup = min(args.warmup_steps, max(1, steps // 10))

    wrapped = wrap_model(model, args, device, world)  # keep for no_sync / state dicts
    model = torch.compile(wrapped) if args.compile else wrapped  # compile after wrapping (DDP/FSDP docs)
    fused = device.type == "cuda" and args.parallel != "fsdp"
    optimizer = configure_optimizer(wrapped, max_lr, args.weight_decay, fused=fused)
    use_amp = device.type == "cuda" and args.parallel != "fsdp"  # FSDP's policy already computes in bf16

    def autocast():
        """bf16 autocast for none/DDP on GPU."""
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp)

    if args.synthetic:
        train_data = SyntheticWindows(cfg.vocab_size, cfg.seq_len, rank, args.seed)
        val_data = train_data
    else:
        train_data = TokenWindows(args.data_dir, "train", cfg.seq_len, rank, args.seed)
        val_data = TokenWindows(args.data_dir, "val", cfg.seq_len, rank, args.seed)

    run_name = args.run_name or f"{args.preset}_{args.parallel}_w{world}"
    run_dir = os.path.join(args.out_dir, run_name)
    peak = peak_flops_per_gpu(device, args.peak_tflops)
    fpt = flops_per_token(cfg)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        print(
            f"{args.preset}: {non_embedding_params(cfg) / 1e6:.1f}M non-embedding params, {steps} steps x "
            f"{batch_tokens:,} tokens = {steps * batch_tokens / 1e6:.1f}M tokens, {fpt * steps * batch_tokens:.2e} FLOPs, "
            f"world {world} ({args.parallel}), grad accum {accum}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_file = open(os.path.join(run_dir, "train_log.jsonl"), "w") if rank == 0 else None
    t_start, tok_s_hist, mfu_hist, val_loss = time.time(), [], [], None
    model.train()
    for step in range(steps):
        t0 = time.time()
        lr = lr_at(step, max_lr, warmup, steps, args.min_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for micro in range(accum):
            x, y = train_data.batch(args.micro_batch)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with grad_sync_context(wrapped, args, micro == accum - 1), autocast():
                _, loss = model(x, y)
                (loss / accum).backward()
            loss_sum += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if hasattr(norm, "full_tensor"):  # DTensor under FSDP
            norm = norm.full_tensor()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        tok_s = batch_tokens / dt
        if step >= min(5, steps - 1):  # skip warm-up/compile steps in throughput averages
            tok_s_hist.append(tok_s)
            if peak:
                mfu_hist.append(fpt * tok_s / (peak * world))

        is_last = step == steps - 1
        if step % args.log_interval == 0 or is_last:
            train_loss = all_reduce_mean((loss_sum / accum).item(), world, device)
            rec = {
                "step": step,
                "tokens": (step + 1) * batch_tokens,
                "loss": train_loss,
                "lr": lr,
                "grad_norm": float(norm),
                "tok_per_s": tok_s,
                "mfu": fpt * tok_s / (peak * world) if peak else None,
            }
            if (step > 0 and step % args.eval_interval == 0) or is_last:
                val_loss = rec["val_loss"] = evaluate(model, val_data, args, rank, world, device, autocast)
            if rank == 0:
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                extra = f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else ""
                mfu = f"  MFU {100 * rec['mfu']:.1f}%" if rec["mfu"] else ""
                print(f"step {step:5d}/{steps}  loss {train_loss:.4f}{extra}  {tok_s / 1e3:.1f}k tok/s{mfu}")

    state = full_state_dict(wrapped, args, world) if args.save_checkpoint else None
    metrics = {
        "preset": args.preset,
        "config": cfg.to_dict(),
        "non_embedding_params": non_embedding_params(cfg),
        "total_params": non_embedding_params(cfg) + cfg.vocab_size * cfg.d_model,
        "flops_per_token": fpt,
        "tokens": steps * batch_tokens,
        "flops": fpt * steps * batch_tokens,
        "steps": steps,
        "batch_tokens": batch_tokens,
        "peak_lr": max_lr,
        "final_val_loss": val_loss,
        "parallel": args.parallel,
        "world_size": world,
        "compile": args.compile,
        "synthetic": args.synthetic,
        "tok_per_s": float(np.mean(tok_s_hist)) if tok_s_hist else None,
        "tok_per_s_per_gpu": float(np.mean(tok_s_hist)) / world if tok_s_hist else None,
        "mfu": float(np.mean(mfu_hist)) if mfu_hist else None,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "wall_minutes": (time.time() - t_start) / 60,
    }
    if rank == 0:
        log_file.close()
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
        if state is not None:
            torch.save({"config": cfg.to_dict(), "model": state, "metrics": metrics}, os.path.join(run_dir, "ckpt.pt"))
        print(f"final val loss {val_loss:.4f}  ({metrics['wall_minutes']:.1f} min)")
    if world > 1:
        dist.destroy_process_group()
    return metrics


def main(argv=None):
    """Entry point."""
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
