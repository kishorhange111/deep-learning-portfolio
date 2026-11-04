"""
A mixture-of-experts feed-forward layer written from scratch, as in Switch Transformer (Fedus et al.,
2021) and Mixtral (Jiang et al., 2024): a learned router sends each token to its top-k of E expert
MLPs and mixes their outputs with the router's renormalized probabilities.

Pieces:
  * router: linear d_model -> E logits; top-k selection; softmax over the chosen k (Mixtral style)
    or the full softmax probability of each chosen expert (Switch style, used when k = 1). With
    k = 1 that probability starts near 1/E, so expert outputs start E times smaller than a dense
    MLP's; `unit_gate=True` divides the gate by its own detached value, which keeps the forward
    value at 1 while the router still receives gradient (an ablation of that down-scaling);
  * load-balancing loss (Switch): E * sum_e f_e * P_e, where f_e is the fraction of routed slots
    that went to expert e and P_e the mean router probability for e. It equals 1 when routing is
    uniform and grows as routing concentrates; without it, routers tend to collapse onto a few
    experts;
  * router z-loss (ST-MoE, Zoph et al., 2022): mean(logsumexp(logits)^2), which keeps router logits
    small and training stable;
  * capacity factor: each expert processes at most ceil(capacity_factor * k * tokens / E) tokens
    per batch; overflow tokens skip the layer (their residual stream passes through unchanged).
    capacity_factor = None means no limit (no tokens dropped).

The layer records routing statistics on every forward pass (`last_stats`) so the training script
can measure expert utilization and specialization.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """One expert: d_model -> d_ff -> GELU -> d_model (the same shape as the dense model's MLP)."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """(n, d_model) -> (n, d_model)."""
        return self.proj(F.gelu(self.fc(x)))


class MoELayer(nn.Module):
    """Top-k routed mixture of expert MLPs with load-balancing and z losses."""

    def __init__(
        self, d_model, d_ff, n_experts, top_k=1, balance_coef=0.01, z_coef=1e-3, capacity_factor=None, unit_gate=False
    ):
        super().__init__()
        self.n_experts, self.top_k, self.unit_gate = n_experts, top_k, unit_gate
        self.balance_coef, self.z_coef, self.capacity_factor = balance_coef, z_coef, capacity_factor
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.aux_loss, self.last_stats = None, None

    def route(self, h):
        """h: (n, d) -> (expert index (n, k), gate weight (n, k), router probabilities (n, E), logits)."""
        logits = self.router(h).float()
        probs = logits.softmax(-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        gates = top_p if self.top_k == 1 else top_p / top_p.sum(-1, keepdim=True)
        if self.unit_gate:  # forward value 1 (no down-scaling), gradient still reaches the router
            gates = gates / gates.detach()
        return top_i, gates, probs, logits

    def capacity(self, n_tokens):
        """Max tokens per expert (None = unlimited)."""
        if self.capacity_factor is None:
            return None
        return math.ceil(self.capacity_factor * self.top_k * n_tokens / self.n_experts)

    def forward(self, x):
        """(..., d) -> (..., d). Sets self.aux_loss (weighted balance + z losses) and self.last_stats."""
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        n = h.shape[0]
        top_i, gates, probs, logits = self.route(h)

        # Slot order: all first choices, then all second choices, so first choices win capacity.
        flat_e = top_i.T.reshape(-1)  # (k * n,)
        flat_tok = torch.arange(n, device=h.device).repeat(self.top_k)
        flat_gate = gates.T.reshape(-1)
        one_hot = F.one_hot(flat_e, self.n_experts)
        position = (one_hot.cumsum(0) - 1)[torch.arange(len(flat_e)), flat_e]  # rank within the expert's queue
        cap = self.capacity(n)
        kept = torch.ones_like(flat_e, dtype=torch.bool) if cap is None else position < cap

        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            sel = (flat_e == e) & kept
            if sel.any():
                tok = flat_tok[sel]
                out.index_add_(0, tok, expert(h[tok]) * flat_gate[sel, None].to(h.dtype))

        # Switch load-balancing loss over first choices: E * sum_e f_e * P_e.
        f = F.one_hot(top_i[:, 0], self.n_experts).float().mean(0)
        p_mean = probs.mean(0)
        balance = self.n_experts * (f * p_mean).sum()
        z = torch.logsumexp(logits, -1).pow(2).mean()
        self.aux_loss = self.balance_coef * balance + self.z_coef * z
        with torch.no_grad():
            load = torch.bincount(flat_e[kept], minlength=self.n_experts).float()
            self.last_stats = {
                "first_choice": top_i[:, 0].detach(),
                "load": load / max(1.0, float(load.sum())),
                "dropped_frac": float(1 - kept.float().mean()),
                "balance_loss": float(balance),
                "z_loss": float(z),
            }
        return out.reshape(shape)


def active_and_total_mlp_params(d_model, d_ff, n_experts, top_k):
    """(parameters used per token, parameters stored) of one MoE layer's experts plus its router."""
    per_expert = 2 * d_model * d_ff
    router = d_model * n_experts
    return top_k * per_expert + router, n_experts * per_expert + router
