"""
A small byte-level GPT shared by the optimizer, mixture-of-experts and quantization projects.

Pre-LayerNorm decoder blocks, learned positions, causal attention through PyTorch's
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

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        """(B, T, d) -> (B, T, d)."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, d // self.n_head).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Pre-LN residual block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg, mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), mlp

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
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, make_mlp(i)) for i in range(cfg.n_layer)])
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
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

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
