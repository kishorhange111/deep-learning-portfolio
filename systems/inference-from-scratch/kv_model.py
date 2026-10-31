"""
The pretraining project's GPT, extended with a KV cache for fast autoregressive decoding.

Without a cache, generating token t means re-running the model over all t previous tokens: O(t)
work per token and O(T^2) for a sequence. Keys and values of past tokens never change (causal
attention), so they can be stored and each new token costs one forward pass over one position.
Decoding then becomes memory-bound: every step streams all weights and the cache from HBM to compute
a handful of FLOPs. That is why batching (several sequences per weight read) and speculative decoding
(several tokens per weight read) speed it up.

Parameter names match systems/gpt-pretraining-scaling-laws/gpt.py, so checkpoints from train.py
--save_checkpoint load with `GPT.from_checkpoint`.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    seq_len: int = 1024
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    rope_base: float = 10000.0


class KVCache:
    """Pre-allocated keys/values for every layer: (n_layer, batch, heads, max_len, head_dim).

    `length` is how many positions are valid. Speculative decoding rolls the cache back after
    rejected draft tokens just by lowering `length`: stale entries beyond it are overwritten later."""

    def __init__(self, cfg, batch, max_len, device, dtype):
        shape = (cfg.n_layer, batch, cfg.n_head, max_len, cfg.d_model // cfg.n_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length, self.max_len = 0, max_len

    def truncate(self, length):
        """Forget every position >= length."""
        assert 0 <= length <= self.length
        self.length = length

    def nbytes(self):
        """Memory held by the cache (2 * layers * batch * heads * max_len * head_dim * bytes)."""
        return self.k.numel() * self.k.element_size() * 2


class RMSNorm(nn.Module):
    """Same as the pretraining model."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """x / rms(x) * g, computed in fp32."""
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(seq_len, head_dim, base, device=None):
    """cos/sin tables (seq_len, head_dim / 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate channel pairs of x (B, H, T, D) by the angles of each token's absolute position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    """Causal self-attention that can append to and read from a KV cache."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None, layer=0, start=0):
        """x holds T new tokens at absolute positions start..start+T-1; cos/sin are already sliced."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            cache.k[layer, :, :, start : start + T] = k
            cache.v[layer, :, :, start : start + T] = v
            k, v = cache.k[layer, :, :, : start + T], cache.v[layer, :, :, : start + T]
            if T == 1:  # a single new query may attend to everything in the cache
                y = F.scaled_dot_product_attention(q, k, v)
            else:  # query i (absolute position start + i) sees keys 0..start + i
                mask = torch.ones(T, start + T, dtype=torch.bool, device=x.device).tril(diagonal=start)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    """d -> 4d -> d with tanh-GELU."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        """Position-wise feed-forward."""
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.mlp_norm, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0, start=0):
        """One layer, optionally cached."""
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer, start)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """Decoder-only Transformer with an optional KV cache."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg.seq_len, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a checkpoint written by the pretraining project's train.py --save_checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        return model.to(device).eval()

    def new_cache(self, batch, max_len, device=None, dtype=None):
        """Empty KV cache sized for `max_len` positions."""
        p = self.tok_emb.weight
        return KVCache(self.cfg, batch, max_len, device or p.device, dtype or p.dtype)

    def forward(self, idx, cache=None):
        """idx (B, T) -> logits (B, T, V). With a cache, idx continues from cache.length and the cache grows by T."""
        T = idx.shape[1]
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.seq_len:
            raise ValueError(f"sequence of {start + T} tokens exceeds the trained context of {self.cfg.seq_len}")
        cos, sin = self.rope_cos[start : start + T], self.rope_sin[start : start + T]
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i, start)
        if cache is not None:
            cache.length = start + T
        return self.lm_head(self.norm(x))
