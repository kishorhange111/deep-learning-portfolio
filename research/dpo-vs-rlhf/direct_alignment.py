"""
Direct alignment (DPO and IPO) for the exact-KL policy of the reward-model overoptimization study.

DPO (Rafailov et al., 2023) skips the reward model: it trains the policy directly on preference
pairs with
    loss = -log sigmoid(beta * h),   h = [log pi(y_w) - log ref(y_w)] - [log pi(y_l) - log ref(y_l)]
where y_w is the preferred and y_l the rejected sequence. Its "implicit reward" is
beta * log(pi / ref). IPO (Azar et al., 2023) replaces the log-sigmoid with a squared loss,
(h - 1 / (2 tau))^2, which stops pushing once the margin reaches 1 / (2 tau) instead of forever.

The policy, reference, gold reward and preference data come from the overoptimization study
(research/reward-model-overoptimization/goodhart.py), so every number here is directly comparable
with RLHF on the same data: KL(policy || reference) is exact, and the gold reward is measured on
fresh samples.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reward-model-overoptimization"))
import goodhart as gh  # noqa: E402,F401  (re-exported for the training script and tests)


def as_pairs(data):
    """(first, second, label) from goodhart.preference_data -> (chosen, rejected)."""
    a, b, label = data
    first = label.bool()
    chosen = torch.where(first[:, None], a, b)
    rejected = torch.where(first[:, None], b, a)
    return chosen, rejected


def log_ratio(policy, reference, y):
    """log pi(y) - log ref(y) per sequence (the reference is frozen)."""
    with torch.no_grad():
        ref = reference.log_prob(y)
    return policy.log_prob(y) - ref


def margin(policy, reference, chosen, rejected):
    """h = log-ratio of the chosen sequence minus log-ratio of the rejected one."""
    return log_ratio(policy, reference, chosen) - log_ratio(policy, reference, rejected)


def dpo_loss(policy, reference, chosen, rejected, beta):
    """Mean DPO loss -log sigmoid(beta * h)."""
    return -F.logsigmoid(beta * margin(policy, reference, chosen, rejected)).mean()


def ipo_loss(policy, reference, chosen, rejected, tau):
    """Mean IPO loss (h - 1 / (2 tau))^2."""
    return (margin(policy, reference, chosen, rejected) - 1 / (2 * tau)).pow(2).mean()


@torch.no_grad()
def evaluate_policy(policy, reference, gold, generator, n=4096, held_out=None, train_pairs=None):
    """Exact KL, gold reward of fresh samples, and diagnostics of the implicit reward:
    * implicit_accuracy: how often log(pi/ref) orders held-out gold-labelled pairs like the gold reward;
    * chosen / rejected log-ratio: mean log pi - log ref on the training pairs' preferred and rejected
      sequences (DPO is known to push *both* down)."""
    y = policy.sample(n, generator)
    rec = {"kl": float(gh.exact_kl(policy, reference)), "gold": float(gold(y).mean())}
    if hasattr(gold, "parts"):
        rec["diminishing"] = float(gold.parts(y)[1].mean())
    if held_out is not None:
        a, b = held_out
        implicit = (log_ratio(policy, reference, a) > log_ratio(policy, reference, b)).float()
        truth = (gold(a) > gold(b)).float()
        rec["implicit_accuracy"] = float((implicit == truth).float().mean())
    if train_pairs is not None:
        chosen, rejected = train_pairs
        rec["chosen_logratio"] = float(log_ratio(policy, reference, chosen).mean())
        rec["rejected_logratio"] = float(log_ratio(policy, reference, rejected).mean())
    return rec


def train_direct(
    reference, gold, pairs, method, strength, steps, lr=0.03, batch=512, seed=0, held_out=None, n_evals=40
):
    """Train a policy (initialized at the reference) with DPO (strength = beta) or IPO (strength = tau)
    for `steps` optimizer steps of `batch` pairs, evaluating `n_evals` times. Returns the log."""
    chosen, rejected = pairs
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    policy = gh.MarkovPolicy(reference.length, reference.vocab, reference.logits.detach())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(chosen)
    steps_per_epoch = max(1, -(-n // batch))
    eval_every = max(1, steps // n_evals)
    probe = (chosen[:4096], rejected[:4096])
    log = [
        {"epoch": 0.0, "step": 0, **evaluate_policy(policy, reference, gold, g, held_out=held_out, train_pairs=probe)}
    ]
    step = 0
    while step < steps:
        order = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            if step >= steps:
                break
            idx = order[i : i + batch]
            if method == "dpo":
                loss = dpo_loss(policy, reference, chosen[idx], rejected[idx], strength)
            elif method == "ipo":
                loss = ipo_loss(policy, reference, chosen[idx], rejected[idx], strength)
            else:
                raise ValueError(method)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                rec = evaluate_policy(policy, reference, gold, g, held_out=held_out, train_pairs=probe)
                log.append({"epoch": step / steps_per_epoch, "step": step, "loss": loss.item(), **rec})
    return log


class BigramReward(nn.Module):
    """A reward model in exactly the function class of DPO's implicit reward for this policy:
    r(y) = sum_t W[t, y_(t-1), y_t] (position 0 reads row 0), i.e. a sum of per-position bigram scores.

    beta * log(pi / ref) of a Markov policy has this form, so comparing RLHF with this reward model
    against RLHF with the MLP reward model separates "DPO vs RL" from "which reward function class"."""

    def __init__(self, length, vocab):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(length, vocab, vocab))
        self.shift, self.scale = 0.0, 1.0

    def forward(self, y):
        """(n, L) tokens -> (n,) rewards (after `normalize`)."""
        prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        t = torch.arange(y.shape[1]).expand_as(y)
        return (self.table[t, prev, y].sum(1) - self.shift) / self.scale

    @torch.no_grad()
    def normalize(self, samples):
        """Mean 0, std 1 on reference samples."""
        self.shift, self.scale = 0.0, 1.0
        r = self(samples)
        self.shift, self.scale = float(r.mean()), float(r.std())


def train_bigram_proxy(pairs, length, vocab, seed, epochs=200, lr=0.03, val_frac=0.1, patience=20):
    """Bradley-Terry training of a BigramReward on (chosen, rejected) pairs, early-stopped on held-out pairs."""
    chosen, rejected = pairs
    n = len(chosen)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]
    model = BigramReward(length, vocab)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), 512):
            idx = order[i : i + 512]
            loss = -F.logsigmoid(model(chosen[idx]) - model(rejected[idx])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = float(-F.logsigmoid(model(chosen[va]) - model(rejected[va])).mean())
        if val < best:
            best, best_epoch, best_state = val, epoch, model.table.detach().clone()
        elif epoch - best_epoch >= patience:
            break
    with torch.no_grad():
        model.table.copy_(best_state)
    model.table.requires_grad_(False)
    return model, {"val_loss": best, "best_epoch": best_epoch}


def endpoint(log):
    """Final and peak gold of a training log, with the KL at each."""
    best = max(log, key=lambda r: r["gold"])
    return {
        "final_gold": log[-1]["gold"],
        "final_kl": log[-1]["kl"],
        "peak_gold": best["gold"],
        "kl_at_peak": best["kl"],
        "epoch_at_peak": best.get("epoch", best.get("step")),
    }


def frontier(points, grid):
    """Upper envelope of (kl, gold) points: the best gold reached at KL <= each grid value."""
    pts = np.array(points, dtype=float)
    return [float(pts[pts[:, 0] <= k, 1].max()) if (pts[:, 0] <= k).any() else float("nan") for k in grid]
