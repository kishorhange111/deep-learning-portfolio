"""
GRPO, written out by hand: rewards, group-relative advantages, token log-probabilities and the clipped
policy-gradient loss. No TRL. Every function here is pure and unit-tested (test_grpo.py).

The objective for one group of G completions o_1..o_G sampled for question q:

    A_i      = (r_i - mean(r)) / (std(r) + eps)                       group-relative advantage, no critic
    ratio_it = pi_theta(o_it | q, o_<t) / pi_old(o_it | q, o_<t)        per-token importance ratio
    L        = - min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A)  +  beta * KL_k3(pi_theta || pi_ref)

averaged over completion tokens. If every completion in a group gets the same reward, std(r) = 0 and
A_i = 0 for the whole group: those samples cost generation time but contribute no gradient. How often
that happens - and what to do about it - is the subject of this study.
"""

import math
import re

import torch

SYSTEM_PROMPT = (
    "You are a helpful assistant. Think through the problem step by step inside <think> </think> tags, "
    "then give only the final numerical answer inside <answer> </answer> tags."
)

# --------------------------------------------------------------------------------------------------
# Answers and rewards
# --------------------------------------------------------------------------------------------------
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STRICT_FORMAT = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)


def gsm8k_gold(answer_field):
    """GSM8K stores the worked solution followed by '#### <final answer>'."""
    return normalize_number(answer_field.split("####")[-1])


def normalize_number(text):
    """'$1,234.50' -> '1234.5'; returns None if the text holds no number."""
    if text is None:
        return None
    matches = _NUMBER.findall(text.replace("$", "").replace("%", ""))
    if not matches:
        return None
    raw = matches[-1].replace(",", "").rstrip(".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isfinite(value) and value == int(value):
        return str(int(value))
    return repr(value)


def extract_answer(completion):
    """Final answer from <answer>..</answer> if present, otherwise the last number in the text.

    The lenient fallback means *accuracy* measures whether the model can solve the problem, independent
    of whether it learned the tag format - the two are scored separately (see `format_ok`).
    """
    tagged = _ANSWER_TAG.findall(completion)
    return normalize_number(tagged[-1] if tagged else completion)


def format_ok(completion):
    return bool(_STRICT_FORMAT.match(completion))


def compute_rewards(completions, gold, format_weight):
    """Rule-based (verifiable) rewards: 1.0 for a correct answer, + format_weight for the strict format."""
    correct = [float(extract_answer(c) == gold and gold is not None) for c in completions]
    fmt = [float(format_ok(c)) for c in completions]
    total = [c + format_weight * f for c, f in zip(correct, fmt)]
    return total, correct, fmt


# --------------------------------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------------------------------
def group_advantages(rewards, scale_by_std=True, eps=1e-4):
    """rewards: (num_groups, G) -> advantages of the same shape.

    scale_by_std=False is the 'Dr. GRPO' variant (Liu et al., 2025), which argues the std division
    over-weights groups whose rewards barely differ (very easy or very hard questions).
    """
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if not scale_by_std:
        return centred
    return centred / (rewards.std(dim=1, keepdim=True, unbiased=False) + eps)


def informative_groups(rewards):
    """Boolean mask of groups whose rewards are not all identical (std > 0) - i.e. groups that produce
    a non-zero gradient. DAPO's 'dynamic sampling' keeps only these."""
    return rewards.max(dim=1).values > rewards.min(dim=1).values


# --------------------------------------------------------------------------------------------------
# Log-probabilities
# --------------------------------------------------------------------------------------------------
def completion_logprobs(model, input_ids, attention_mask, completion_len):
    """Per-token log pi(token | prefix) for the last `completion_len` positions of each sequence.

    Sequences are laid out [left-padded prompt | right-padded completion], so every completion starts
    at the same index T - completion_len. Asking the model only for the last completion_len + 1 logits
    (`logits_to_keep`) avoids materialising a (batch, T, vocab) tensor - with Qwen's 151k-token vocabulary
    the prompt logits alone would dominate GPU memory.
    """
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)  # positions ignore the left padding
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits_to_keep=completion_len + 1,
        use_cache=False,
    ).logits
    logits = logits[:, :-1].float()  # position t predicts token t + 1
    targets = input_ids[:, -completion_len:]
    token_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logits - torch.logsumexp(logits, dim=-1)


# --------------------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------------------
def kl_k3(logp, ref_logp):
    """Unbiased, always non-negative estimator of KL(pi || pi_ref) per token (Schulman, 2020):
    exp(ref - logp) - (ref - logp) - 1."""
    delta = ref_logp - logp
    return torch.exp(delta) - delta - 1


def grpo_loss(
    logp, old_logp, advantages, mask, ref_logp=None, beta=0.0, eps_low=0.2, eps_high=0.2, aggregation="token"
):
    """Clipped GRPO surrogate.

    logp, old_logp, ref_logp, mask: (B, C) over completion tokens; advantages: (B,).
    aggregation='token'    - average over all completion tokens in the batch (DAPO): long answers
                             are not down-weighted per token.
    aggregation='sequence' - average within each completion, then over completions (original GRPO).
    With one optimisation step per batch, old_logp == logp.detach(), the ratio is exactly 1 and the
    gradient reduces to the REINFORCE-style -A * grad log pi.
    Returns (loss, stats dict).
    """
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    per_token = -torch.min(unclipped, clipped)
    kl = None
    if ref_logp is not None and beta > 0:
        kl = kl_k3(logp, ref_logp)
        per_token = per_token + beta * kl

    mask = mask.to(per_token.dtype)
    if aggregation == "token":
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    elif aggregation == "sequence":
        loss = ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    with torch.no_grad():
        n = mask.sum().clamp(min=1)
        was_clipped = ((ratio < 1 - eps_low) & (adv < 0)) | ((ratio > 1 + eps_high) & (adv > 0))
        stats = {"clip_frac": ((was_clipped.float() * mask).sum() / n).item()}
        if kl is not None:
            stats["kl"] = ((kl * mask).sum() / n).item()
    return loss, stats


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
