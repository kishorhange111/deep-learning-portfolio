"""
A compact decoder-only Transformer for pretraining: pre-norm RMSNorm, rotary position embeddings,
GELU MLP, tied input/output embeddings, PyTorch's fused scaled_dot_product_attention.

It also counts the parameters and FLOPs that scaling laws are written in terms of:

    N_nonemb  = 12 * n_layer * d_model^2              (attention 4d^2 + MLP 8d^2 per layer, Kaplan et al.)
    FLOPs/tok = 6 * (N_nonemb + V * d_model)          (forward + backward of every matmul, incl. the LM head)
              + 12 * n_layer * d_model * seq_len      (attention scores and weighted sum, PaLM appendix B)

Parameter names are shared with systems/inference-from-scratch, which loads these checkpoints.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # GPT-2's 50,257 rounded up to a multiple of 64 for faster matmuls
    seq_len: int = 1024
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)


# Model family with d_model ~ 64 x n_layer and 64-dim heads; names are non-embedding parameter counts.
PRESETS = {
    "3M": dict(n_layer=4, d_model=256, n_head=4),
    "11M": dict(n_layer=6, d_model=384, n_head=6),
    "25M": dict(n_layer=8, d_model=512, n_head=8),
    "49M": dict(n_layer=10, d_model=640, n_head=10),
    "85M": dict(n_layer=12, d_model=768, n_head=12),
    "201M": dict(n_layer=16, d_model=1024, n_head=16),
    "302M": dict(n_layer=24, d_model=1024, n_head=16),
}


class RMSNorm(nn.Module):
    """x / rms(x) * g - LayerNorm without mean-centring or bias (cheaper, same quality in practice)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalise in fp32 for stability, return in the input dtype."""
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(seq_len, head_dim, base, device=None):
    """cos/sin tables of shape (seq_len, head_dim / 2) for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs of channels of x (B, H, T, D) by position-dependent angles; cos/sin are (T, D/2).

    Dot products between rotated queries and keys then depend only on the relative position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    """Causal multi-head self-attention with one fused qkv projection."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        """(B, T, d) -> (B, T, d)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash / memory-efficient kernels
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    """d -> 4d -> d with GELU."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        """Position-wise feed-forward."""
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.mlp_norm, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin):
        """One Transformer layer."""
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """Token embedding -> n_layer blocks -> RMSNorm -> tied LM head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        cos, sin = rope_tables(cfg.seq_len, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual-branch outputs by 1/sqrt(2L)
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        """N(0, 0.02) for linear and embedding weights."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx (B, T) -> logits (B, T, V), and mean cross-entropy if targets are given."""
        T = idx.shape[1]
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, non_embedding=True):
        """Non-embedding parameters exclude the (tied) token embedding matrix."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n


def non_embedding_params(cfg):
    """12 L d^2 (+ the tiny RMSNorm gains)."""
    return 12 * cfg.n_layer * cfg.d_model**2 + (2 * cfg.n_layer + 1) * cfg.d_model


def flops_per_token(cfg):
    """Training FLOPs per token (forward + backward), counting the LM head and attention."""
    return 6 * (non_embedding_params(cfg) + cfg.vocab_size * cfg.d_model) + 12 * cfg.n_layer * cfg.d_model * cfg.seq_len


def config_for(preset, **overrides):
    """GPTConfig for a named preset, with optional field overrides (e.g. seq_len, vocab_size)."""
    return GPTConfig(**{**PRESETS[preset], **overrides})


def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95), fused=None):
    """AdamW with weight decay on matrices only (not on norms gains)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
