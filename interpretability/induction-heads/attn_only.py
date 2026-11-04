"""
An attention-only transformer whose internals are easy to read, plus the measurements used to
find induction heads (Elhage et al., 2021; Olsson et al., 2022).

An induction head completes patterns: having seen "... A B ... A", it predicts B. It needs two heads
in different layers:
  1. a previous-token head (layer 0) writes "the token before me was A" into each position;
  2. an induction head (layer 1) whose keys read that information (K-composition), so a query at
     the second A attends to the position right after the first A, the B, and copies it.
A 1-layer model has no earlier head to compose with, so it cannot form induction heads.

Attention is computed explicitly (no fused kernel) so that patterns can be returned and single heads
ablated.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head causal attention with access to patterns and per-head ablation."""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, ablate_heads=()):
        """x (B, T, d) -> (output (B, T, d), pattern (B, H, T, T)). Ablated heads output zero."""
        B, T, _ = x.shape

        def split(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        pattern = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        z = pattern @ v  # (B, H, T, d_head)
        if ablate_heads:
            keep = torch.ones(self.n_head, device=x.device)
            keep[list(ablate_heads)] = 0.0
            z = z * keep[None, :, None, None]
        return self.W_O(z.transpose(1, 2).reshape(B, T, -1)), pattern

    def head_matrices(self):
        """Per-head W_QK (d x d, score = x_q W_QK x_k^T) and W_OV (d x d, output = x W_OV), row-vector convention."""
        d, H, dh = self.W_Q.weight.shape[1], self.n_head, self.d_head
        wq = self.W_Q.weight.T.view(d, H, dh)  # nn.Linear stores (out, in)
        wk = self.W_K.weight.T.view(d, H, dh)
        wv = self.W_V.weight.T.view(d, H, dh)
        wo = self.W_O.weight.T.view(H, dh, d)
        qk = torch.einsum("ahe,bhe->hab", wq, wk) / math.sqrt(dh)
        ov = torch.einsum("ahe,heb->hab", wv, wo)
        return qk, ov


class AttnOnlyTransformer(nn.Module):
    """Token + learned position embeddings -> n_layer attention-only residual blocks -> LayerNorm -> unembed."""

    def __init__(self, vocab_size, n_layer=2, n_head=8, d_model=256, seq_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList(Attention(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layer, self.n_head, self.seq_len = n_layer, n_head, seq_len
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx, ablate=(), return_patterns=False):
        """idx (B, T) -> logits; `ablate` is a set of (layer, head) pairs to zero out."""
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        patterns = []
        for i, layer in enumerate(self.layers):
            out, pattern = layer(x, [h for (lyr, h) in ablate if lyr == i])
            x = x + out
            patterns.append(pattern)
        logits = self.unembed(self.ln_f(x))
        return (logits, patterns) if return_patterns else logits


# --------------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------------
def per_position_loss(model, idx, ablate=()):
    """Cross-entropy at every position, averaged over the batch -> (T - 1,)."""
    logits = model(idx[:, :-1], ablate=ablate)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1), reduction="none")
    return loss.view(idx.shape[0], -1).mean(0)


def in_context_learning_score(pos_loss):
    """Loss late in the context minus loss early on. Olsson et al. use token 500 vs. 50 of 1024; here
    a window around 10 % of the context vs. its last ~4 % (positions ~21-31 vs. ~245-255 of 256).
    More negative means the model uses its context more."""
    T = len(pos_loss)
    w = max(2, T // 50)
    centre = max(w, round(0.1 * T))
    early, late = pos_loss[centre - w : centre + w], pos_loss[-2 * w :]
    return (late.mean() - early.mean()).item()


def repeated_random_tokens(batch, length, vocab_size, generator=None, device="cpu"):
    """[r_1 .. r_L, r_1 .. r_L] with uniformly random r: the second half is predictable only by copying."""
    r = torch.randint(100, vocab_size, (batch, length), generator=generator).to(device)
    return torch.cat([r, r], dim=1)


def head_scores(patterns, length):
    """Induction and previous-token scores for every head from patterns on a repeated random sequence.

    induction: mean attention from position i (second half) to i - L + 1, the token after the
               previous occurrence of the current token;
    previous-token: mean attention from i to i - 1.
    Returns two (n_layer, n_head) tensors."""
    induction, previous = [], []
    for p in patterns:  # (B, H, 2L, 2L)
        i = torch.arange(length, 2 * length)
        induction.append(p[:, :, i, i - length + 1].mean(dim=(0, 2)))
        j = torch.arange(1, 2 * length)
        previous.append(p[:, :, j, j - 1].mean(dim=(0, 2)))
    return torch.stack(induction), torch.stack(previous)


def repeated_sequence_loss(model, tokens, length, ablate=()):
    """Mean loss on the second (copyable) half of repeated random sequences."""
    return per_position_loss(model, tokens, ablate)[length:].mean().item()


def composition_scores(model, kind="K"):
    """Elhage et al.'s composition score between every layer-0 head (rows) and layer-1 head (columns):
    ||M_1 M_0|| / (||M_1|| ||M_0||) for the relevant pair of circuit matrices, minus the mean score of
    random Gaussian matrices of the same shape (so ~0 means "no more than chance")."""
    qk1, ov1 = model.layers[1].head_matrices()
    _, ov0 = model.layers[0].head_matrices()

    def score(a, b):
        return (a @ b).norm() / (a.norm() * b.norm())

    def pair(h1, h0):
        if kind == "K":  # layer-1 keys read layer-0 output: x_q W_QK (x W_OV0)^T
            return score(qk1[h1], ov0[h0].T)
        if kind == "Q":
            return score(ov0[h0], qk1[h1])
        return score(ov0[h0], ov1[h1])  # V-composition

    raw = torch.tensor([[pair(h1, h0).item() for h1 in range(qk1.shape[0])] for h0 in range(ov0.shape[0])])
    gen = torch.Generator().manual_seed(0)
    d, dh = qk1.shape[-1], model.layers[0].d_head

    def low_rank():  # per-head circuit matrices have rank d_head, so the chance level must too
        return torch.randn(d, dh, generator=gen) @ torch.randn(dh, d, generator=gen)

    baseline = torch.stack([score(low_rank(), low_rank()) for _ in range(20)])
    return raw - baseline.mean()
