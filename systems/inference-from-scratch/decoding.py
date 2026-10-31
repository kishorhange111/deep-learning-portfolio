"""
Decoding algorithms: naive (no cache), KV-cached, and speculative decoding.

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023): a small draft model proposes k
tokens autoregressively (cheap), and the large target model scores all k in one forward pass (the
same cost as generating one token, since decoding is memory-bound). Each draft token x is accepted
with probability min(1, p(x) / q(x)). At the first rejection, a replacement is sampled from
norm(max(0, p - q)). If all k are accepted, one bonus token is sampled from the target. The output
distribution is then *exactly* the target's: speculative decoding changes the speed, never the samples'
distribution. Greedy decoding is the special case where p and q are one-hot, so the output is
token-for-token identical to the target's greedy output (tested).

Expected tokens per target forward pass, if each draft token is accepted with probability alpha:
(1 - alpha^(k+1)) / (1 - alpha).
"""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class DecodeStats:
    """Counters for benchmarking."""

    new_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    proposed: int = 0
    accepted: int = 0
    seconds: float = 0.0
    accepted_per_round: list = field(default_factory=list)

    @property
    def acceptance_rate(self):
        """Fraction of proposed draft tokens the target accepted."""
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def tokens_per_target_forward(self):
        """New tokens produced per (expensive) target forward pass."""
        return self.new_tokens / self.target_forwards if self.target_forwards else float("nan")


def probs_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Logits (..., V) -> the sampling distribution. temperature == 0 gives a one-hot argmax (greedy).

    Draft and target probabilities must come from the same transformation for speculative decoding
    to reproduce the target's (transformed) distribution exactly."""
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p  # smallest prefix with mass >= top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample(probs, generator=None):
    """One token per row from probability rows (B, V) -> (B,)."""
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_naive(model, prompt, n, temperature=0.0, top_k=None, top_p=None, generator=None):
    """Re-run the whole sequence for every new token (the baseline the KV cache removes). prompt: (B, T)."""
    tokens, stats, t0 = prompt, DecodeStats(), time.perf_counter()
    for _ in range(n):
        logits = model(tokens)[:, -1]
        nxt = sample(probs_from_logits(logits, temperature, top_k, top_p), generator)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        stats.target_forwards += 1
    stats.new_tokens, stats.seconds = n * prompt.shape[0], time.perf_counter() - t0
    return tokens, stats


@torch.no_grad()
def generate_kv(model, prompt, n, temperature=0.0, top_k=None, top_p=None, generator=None):
    """Prefill the prompt once, then one single-position forward per new token. prompt: (B, T)."""
    t0 = time.perf_counter()
    cache = model.new_cache(prompt.shape[0], prompt.shape[1] + n)
    logits = model(prompt, cache)[:, -1]  # prefill: the whole prompt in one parallel pass
    out = [prompt]
    for i in range(n):
        nxt = sample(probs_from_logits(logits, temperature, top_k, top_p), generator)
        out.append(nxt[:, None])
        if i < n - 1:
            logits = model(nxt[:, None], cache)[:, -1]  # decode: one position, attends to the cache
    stats = DecodeStats(new_tokens=n * prompt.shape[0], target_forwards=n, seconds=time.perf_counter() - t0)
    return torch.cat(out, dim=1), stats


def verify_draft(p, q, draft, generator=None):
    """The speculative-sampling acceptance rule for one round.

    p: (k+1, V) target distributions at the k draft positions plus one bonus position.
    q: (k, V) draft distributions the draft tokens were sampled from. draft: (k,) tokens.
    Returns (number of accepted draft tokens, the next token: a replacement or the bonus).
    """
    k = draft.shape[0]
    u = torch.rand(k, generator=generator, device=p.device)
    for i in range(k):
        x = draft[i]
        if u[i] < torch.clamp(p[i, x] / q[i, x], max=1.0):  # accept with prob min(1, p/q)
            continue
        residual = torch.clamp(p[i] - q[i], min=0.0)  # rejected: resample from the leftover mass
        total = residual.sum()
        residual = residual / total if total > 0 else p[i]
        return i, sample(residual[None], generator)[0]
    return k, sample(p[k][None], generator)[0]  # all accepted: free bonus token from the target


@torch.no_grad()
def generate_speculative(target, draft, prompt, n, k=4, temperature=0.0, top_k=None, top_p=None, generator=None):
    """Speculative decoding for one sequence. prompt: (1, T). Returns (tokens (1, T + n), stats).

    Invariant between rounds: both caches hold every token except the last one (`pending`)."""
    assert prompt.shape[0] == 1, "speculative decoding here is single-sequence (the latency-bound case)"
    t0 = time.perf_counter()
    max_len = prompt.shape[1] + n + k + 1
    t_cache, d_cache = target.new_cache(1, max_len), draft.new_cache(1, max_len)
    if prompt.shape[1] > 1:
        target(prompt[:, :-1], t_cache)
        draft(prompt[:, :-1], d_cache)
    tokens, pending = prompt[0].tolist(), prompt[:, -1:]
    stats = DecodeStats()  # counts decoding rounds only (the prompt prefill is shared by every method)
    produced = 0
    while produced < n:
        start = t_cache.length
        # 1. draft proposes k tokens, one cheap forward each
        q_rows, proposal, x = [], [], pending
        for _ in range(k):
            q = probs_from_logits(draft(x, d_cache)[0, -1], temperature, top_k, top_p)
            x = sample(q[None], generator)[None]
            q_rows.append(q)
            proposal.append(x)
        stats.draft_forwards += k
        proposal_t = torch.cat(proposal, dim=1)  # (1, k)
        # 2. target scores pending + all k proposals in ONE forward pass
        p = probs_from_logits(target(torch.cat([pending, proposal_t], dim=1), t_cache)[0], temperature, top_k, top_p)
        stats.target_forwards += 1
        # 3. accept/reject
        n_acc, nxt = verify_draft(p, torch.stack(q_rows), proposal_t[0], generator)
        stats.proposed += k
        stats.accepted += n_acc
        stats.accepted_per_round.append(n_acc)
        # 4. roll both caches back to "everything up to the last accepted draft token"
        t_cache.truncate(start + 1 + n_acc)
        if n_acc == k:  # the draft never fed its own last proposal; do it so the caches stay in step
            draft(proposal_t[:, -1:], d_cache)
            stats.draft_forwards += 1
        d_cache.truncate(start + 1 + n_acc)
        new = proposal_t[0, :n_acc].tolist() + [int(nxt)]
        tokens += new
        produced += len(new)
        pending = torch.tensor([[int(nxt)]], device=prompt.device)
    out = torch.tensor([tokens[: prompt.shape[1] + n]], device=prompt.device)
    stats.new_tokens, stats.seconds = n, time.perf_counter() - t0
    return out, stats


def expected_tokens_per_round(alpha, k):
    """Leviathan et al.'s formula for the expected tokens per target call at acceptance rate alpha."""
    return k + 1 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)
