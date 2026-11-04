"""
The model, data and Fourier tools for the grokking study.

Task: the input is the three tokens [a, b, =] with a, b in 0..p-1, and the label is (a + b) mod p.
Model: the 1-layer transformer of Nanda et al. (2023), "Progress measures for grokking via mechanistic
interpretability": learned token and position embeddings, 4 attention heads, a ReLU MLP, no LayerNorm.

Only the logits at the "=" position are trained, and with a single layer the other positions cannot
influence them, so the forward pass computes the query, the MLP and the unembedding for that position
only. This is exactly equivalent to running the full causal model and reading its last position, at
about a third of the cost.

The Fourier tools follow the mechanism that paper reverse-engineered. The trained model represents
a and b as cos(w a), sin(w a) for a few key frequencies w = 2 pi k / p, combines them with
trigonometric identities into cos(w (a + b)) and sin(w (a + b)), and reads out
logit(c) ~ sum_k alpha_k cos(w_k (a + b - c)). For a fixed output c, that is a function of (a, b)
inside the span of cos(w_k (a + b)) and sin(w_k (a + b)), which is what the restricted and excluded
losses below project onto.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GrokConfig:
    """Model shape. The defaults are the paper's: p = 113, d_model 128, 4 heads, d_mlp 512."""

    p: int = 113
    d_model: int = 128
    n_head: int = 4
    d_mlp: int = 512


def make_dataset(p, train_frac, seed):
    """All p^2 pairs as [a, b, =] token rows, labels (a + b) mod p, and a random train / test split."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)  # token p is "="
    y = (a + b) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    n_train = int(round(train_frac * p * p))
    return x, y, perm[:n_train], perm[n_train:]


class ModAddTransformer(nn.Module):
    """1-layer transformer that returns logits over the p answers at the '=' position."""

    def __init__(self, cfg: GrokConfig):
        super().__init__()
        self.cfg = cfg
        d, h, dh, m, p = cfg.d_model, cfg.n_head, cfg.d_model // cfg.n_head, cfg.d_mlp, cfg.p

        def init(*shape, fan_in):
            return nn.Parameter(torch.randn(*shape) / math.sqrt(fan_in))

        self.W_E = init(p + 1, d, fan_in=d)
        self.W_pos = init(3, d, fan_in=d)
        self.W_Q, self.W_K, self.W_V = (init(h, d, dh, fan_in=d) for _ in range(3))
        self.W_O = init(h, dh, d, fan_in=d)
        self.W_in = init(d, m, fan_in=d)
        self.b_in = nn.Parameter(torch.zeros(m))
        self.W_out = init(m, d, fan_in=m)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = init(d, p, fan_in=d)

    def forward(self, x, cache=None):
        """x: (B, 3) tokens -> (B, p) logits. Pass a dict as `cache` to receive attention and MLP activations."""
        resid = self.W_E[x] + self.W_pos  # (B, 3, d)
        q = torch.einsum("bd,hde->bhe", resid[:, -1], self.W_Q)
        k = torch.einsum("bpd,hde->bhpe", resid, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", resid, self.W_V)
        attn = torch.softmax(torch.einsum("bhe,bhpe->bhp", q, k) / math.sqrt(q.shape[-1]), dim=-1)
        z = torch.einsum("bhp,bhpe->bhe", attn, v)
        mid = resid[:, -1] + torch.einsum("bhe,hed->bd", z, self.W_O)
        post = torch.relu(mid @ self.W_in + self.b_in)
        out = mid + post @ self.W_out + self.b_out
        if cache is not None:
            cache["attn"], cache["mlp_post"] = attn, post
        return out @ self.W_U


def loss_fn(logits, y):
    """Cross-entropy in float64. Train loss reaches ~1e-7 here, where float32 log-softmax loses precision."""
    return F.cross_entropy(logits.double(), y)


def accuracy(logits, y):
    """Fraction of rows whose arg-max logit is the label."""
    return (logits.argmax(-1) == y).float().mean().item()


def weight_norm(model):
    """Sum of squared parameters: what weight decay pushes down."""
    return float(sum((w.detach().double() ** 2).sum() for w in model.parameters()))


# ---------------------------------------------------------------- Fourier tools
def fourier_basis(p):
    """(p, p) orthonormal rows: constant, then cos(w_k x), sin(w_k x) for k = 1..(p-1)/2."""
    x = np.arange(p)
    rows = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        for f in (np.cos, np.sin):
            r = f(2 * np.pi * k * x / p)
            rows.append(r / np.linalg.norm(r))
    return np.stack(rows)


def embedding_frequency_norms(model):
    """Share of the token embedding's squared norm on each frequency k = 1..(p-1)/2 (sums to <= 1).

    The generalizing algorithm uses only a few frequencies, so this distribution becomes sparse."""
    p = model.cfg.p
    w_e = model.W_E.detach().double().numpy()[:p]  # drop the '=' token
    coef = fourier_basis(p) @ w_e  # (p, d)
    energy = (coef**2).sum(1)
    per_freq = energy[1::2] + energy[2::2]
    return per_freq / energy.sum()


def gini(x):
    """Gini coefficient of non-negative values: 0 = spread evenly, -> 1 = concentrated on one entry."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


@torch.no_grad()
def logit_grid(model, x):
    """Logits for all p^2 inputs (x from make_dataset, in a-major order), as float64 (p, p, p) [a, b, c]."""
    p = model.cfg.p
    return model(x).double().reshape(p, p, p)


def sum_frequency_directions(p, freqs):
    """(2 * len(freqs), p * p) orthonormal directions cos(w_k (a + b)), sin(w_k (a + b)) over the (a, b) grid."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    s = (a + b).flatten()
    dirs = []
    for k in freqs:
        for f in (np.cos, np.sin):
            d = f(2 * np.pi * k * s / p)
            dirs.append(d / np.linalg.norm(d))
    return np.stack(dirs) if dirs else np.zeros((0, p * p))


def key_frequency_projection(logits, freqs):
    """Component of the (p, p, p) logits that lies in the key-frequency (a + b) directions, per output c."""
    p = logits.shape[0]
    flat = logits.reshape(p * p, p).numpy() if torch.is_tensor(logits) else logits.reshape(p * p, p)
    dirs = sum_frequency_directions(p, freqs)
    return torch.from_numpy((dirs.T @ (dirs @ flat)).reshape(p, p, p))


def frequency_shares_of_logits(logits):
    """Share of the logits' non-constant variance (over (a, b), summed over c) in each frequency's
    (a + b) directions."""
    p = logits.shape[0]
    flat = logits.reshape(p * p, p)
    flat = flat - flat.mean(0, keepdim=True)
    total = float((flat**2).sum())
    dirs = torch.from_numpy(sum_frequency_directions(p, range(1, (p - 1) // 2 + 1)))
    energy = ((dirs @ flat) ** 2).sum(1)
    return (energy[0::2] + energy[1::2]).numpy() / total


def key_frequencies(logits, min_share=0.02):
    """Frequencies k whose (a + b) directions carry at least `min_share` of the final logits' variance."""
    shares = frequency_shares_of_logits(logits)
    return [int(k) + 1 for k in np.argsort(-shares) if shares[k] >= min_share]


def restricted_and_excluded_loss(logits, y_grid, freqs, train_mask):
    """The two progress measures of Nanda et al. (2023), on the (p, p, p) logit grid.

    * restricted loss: keep only the key-frequency components (plus each output's mean) and evaluate on
      every pair. Low restricted loss means the Fourier algorithm alone already solves the task.
    * excluded loss: delete the key-frequency components and evaluate on the training pairs. Rising
      excluded loss while train loss stays ~0 means the model's fit to the training set increasingly
      runs through the generalizing algorithm rather than through memorization."""
    p = logits.shape[0]
    proj = key_frequency_projection(logits, freqs)
    mean = logits.reshape(p * p, p).mean(0)
    restricted = (proj + mean).reshape(p * p, p)
    excluded = (logits - proj).reshape(p * p, p)
    y = y_grid.flatten()
    r = float(F.cross_entropy(restricted, y))
    e = float(F.cross_entropy(excluded[train_mask], y[train_mask]))
    return r, e


@torch.no_grad()
def neuron_frequency_purity(model, x):
    """For each MLP neuron: the share of its activation's non-constant variance over the (a, b) grid that
    lies in the 2D Fourier terms of its single best frequency (cos / sin of w a, of w b, and their products).

    Returns (best frequency per neuron, purity per neuron; NaN for neurons that never fire). Neurons
    of the generalizing algorithm each read one frequency, so purity near 1 is the signature of the
    mechanism."""
    p = model.cfg.p
    cache = {}
    model(x, cache=cache)
    act = cache["mlp_post"].double().numpy().reshape(p, p, -1)
    basis = fourier_basis(p)
    coef = np.einsum("ia,abn,jb->ijn", basis, act, basis)  # 2D Fourier coefficients per neuron
    energy = coef**2
    energy[0, 0] = 0.0  # drop the constant term
    total = energy.sum((0, 1))
    n_freq = (p - 1) // 2
    per_freq = np.zeros((n_freq, energy.shape[2]))
    for k in range(n_freq):
        idx = [2 * k + 1, 2 * k + 2]
        per_freq[k] = (
            energy[np.ix_([0], idx)].sum((0, 1))  # cos / sin of w b
            + energy[np.ix_(idx, [0])].sum((0, 1))  # cos / sin of w a
            + energy[np.ix_(idx, idx)].sum((0, 1))  # products
        )
    best = per_freq.argmax(0)
    alive = total > 1e-12 * max(float(total.max()), 1e-300)
    purity = np.where(alive, per_freq.max(0) / np.where(alive, total, 1.0), np.nan)  # NaN for dead neurons
    return best + 1, purity
