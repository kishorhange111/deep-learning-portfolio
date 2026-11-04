"""
A small byte-level GPT shared by the optimizer, mixture-of-experts, quantization, sparse-autoencoder
and long-context projects.

Pre-LayerNorm decoder blocks, learned positions (or rotary embeddings, `pos="rope"`, with the
context-extension methods in `rope_frequencies`), causal attention through PyTorch's
scaled_dot_product_attention, a GELU MLP and an untied output head. Every weight matrix is an
nn.Linear (biases off), so optimizers can treat "matrix" parameters separately and quantizers can
find every layer by type. The MLP can be swapped for any module with `mlp_factory`, which is how
the mixture-of-experts project plugs in its expert layer; a module may set `aux_loss` during its
forward pass, and the model adds those up for the trainer.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ByteGPTConfig:
    """Model shape. vocab is 256 bytes."""

    seq_len: int = 128
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 512
    vocab: int = 256
    pos: str = "learned"  # "learned" (absolute embeddings) or "rope" (rotary)
    rope_base: float = 10000.0


ROPE_METHODS = ("none", "pi", "ntk", "yarn")


def rope_frequencies(head_dim, base=10000.0, method="none", factor=1.0, train_len=None, alpha=1.0, beta=32.0):
    """Rotary inverse frequencies for a head dimension, adapted for running at `factor` times the
    training context. Returns (inv_freq (head_dim / 2,), attention logit scale).

    * none: plain RoPE; positions beyond the training length are extrapolated.
    * pi:   position interpolation (Chen et al., 2023): every frequency divided by `factor`, so the
            longer context is squeezed into the trained position range.
    * ntk:  "NTK-aware" scaling (bloc97, 2023): the base is raised to base * factor^(d / (d - 2)), so
            high frequencies (local order) barely change while low frequencies are interpolated.
    * yarn: YaRN (Peng et al., 2023): per-frequency blend. Frequencies that complete many rotations
            inside the training window (more than `beta`) are kept, those with fewer than `alpha`
            are interpolated like PI, a linear ramp in between; attention logits are multiplied by
            (0.1 ln factor + 1)^2 to undo the entropy increase of longer contexts."""
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    scale = 1.0
    if method == "none" or factor <= 1.0:
        pass
    elif method == "pi":
        inv = inv / factor
    elif method == "ntk":
        inv = (base * factor ** (head_dim / (head_dim - 2))) ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        )
    elif method == "yarn":
        if train_len is None:
            raise ValueError("yarn needs the training context length")
        rotations = train_len * inv / (2 * math.pi)  # full turns each frequency makes in the training window
        keep = ((rotations - alpha) / (beta - alpha)).clamp(0.0, 1.0)  # 1 = keep, 0 = interpolate
        inv = inv / factor * (1 - keep) + inv * keep
        scale = (0.1 * math.log(factor) + 1.0) ** 2
    else:
        raise ValueError(f"unknown RoPE method {method!r}; expected one of {ROPE_METHODS}")
    return inv.float(), scale


def apply_rope(x, cos, sin):
    """Rotate pairs (x_i, x_(i + d/2)) of x (..., T, d) by the angles behind cos / sin (T, d / 2)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MLP(nn.Module):
    """d_model -> d_ff -> GELU -> d_model."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """(..., d_model) -> (..., d_model)."""
        return self.proj(F.gelu(self.fc(x)))


class Attention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, cfg, rope=None):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = rope  # shared dict with the model's RoPE settings, or None for learned positions

    def forward(self, x):
        """(B, T, d) -> (B, T, d)."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, d // self.n_head).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            inv, scale = rope_frequencies(d // self.n_head, **self.rope)
            angles = torch.arange(t, dtype=torch.float32, device=x.device)[:, None] * inv.to(x.device)[None, :]
            cos, sin = angles.cos(), angles.sin()
            q, k = apply_rope(q, cos, sin) * scale, apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Pre-LN residual block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg, mlp, rope=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg, rope), mlp

    def forward(self, x):
        """(B, T, d) -> (B, T, d)."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    """Byte-level decoder-only transformer."""

    def __init__(self, cfg, mlp_factory=None):
        super().__init__()
        self.cfg = cfg
        make_mlp = mlp_factory or (lambda layer: MLP(cfg.d_model, cfg.d_ff))
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.rope = None
        if cfg.pos == "learned":
            self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        elif cfg.pos == "rope":
            self.rope = {"base": cfg.rope_base, "method": "none", "factor": 1.0, "train_len": cfg.seq_len}
        else:
            raise ValueError(f"unknown position scheme {cfg.pos!r}")
        self.blocks = nn.ModuleList([Block(cfg, make_mlp(i), self.rope) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual-branch outputs by depth
            if name.endswith("out.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        """Normal(0, 0.02) for linear and embedding weights."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        """x: (B, T) bytes -> (logits, loss or None). The loss includes any aux_loss set by submodules."""
        h = self.tok(x)
        if self.rope is None:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def set_rope_scaling(self, method="none", factor=1.0):
        """Choose how a RoPE model handles contexts longer than it was trained on (see rope_frequencies)."""
        if self.rope is None:
            raise ValueError("set_rope_scaling needs a model built with pos='rope'")
        if method not in ROPE_METHODS:
            raise ValueError(f"unknown RoPE method {method!r}; expected one of {ROPE_METHODS}")
        self.rope.update(method=method, factor=float(factor))

    def aux_loss(self):
        """Sum of `aux_loss` attributes set by submodules during the last forward pass (0 if none)."""
        total = 0.0
        for m in self.modules():
            if m is not self and getattr(m, "aux_loss", None) is not None and torch.is_tensor(m.aux_loss):
                total = total + m.aux_loss
        return total

    def num_params(self):
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())


def lr_schedule(step, total, warmup, min_ratio=0.1):
    """Linear warmup, then cosine decay to min_ratio of the peak. Returns a multiplier."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_bpb(model, x, y, batch=64):
    """Mean cross-entropy over fixed windows, in bits per byte."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb = torch.from_numpy(np.asarray(x[i : i + batch])), torch.from_numpy(np.asarray(y[i : i + batch]))
        logits, _ = model(xb)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="sum").item()
        count += yb.numel()
    model.train(was_training)
    return total / count / math.log(2)
