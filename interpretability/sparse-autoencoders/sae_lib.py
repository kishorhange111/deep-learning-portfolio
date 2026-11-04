"""
Sparse autoencoders (SAEs) for dictionary learning on a language model's residual stream.

An SAE rewrites each activation x (d dims) as a sparse, non-negative combination of many more
"latent" directions (here 16x d): z = sparsify(W_enc (x - b_dec) + b_enc), x_hat = W_dec z + b_dec.
If it works, each latent is a human-interpretable feature of the input (Bricken et al., 2023,
"Towards Monosemanticity"; Templeton et al., 2024). Three ways to make z sparse are compared:
  * relu:      ReLU, plus an L1 penalty on z (the original recipe). The penalty also shrinks the
               activations it keeps ("shrinkage"), trading reconstruction for sparsity.
  * topk:      keep the k largest pre-activations per token, zero the rest (Gao et al., 2024). L0 is
               exactly k; no penalty, so no shrinkage.
  * batchtopk: keep the k * batch largest across the whole batch (Bussmann et al., 2024), so tokens
               that need more latents can use more. At inference it uses a fixed threshold, the
               average smallest kept activation seen in training.
Decoder rows are kept at unit norm, so a latent's activation is its contribution's size.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SparseAutoencoder(nn.Module):
    """SAE with a relu (+ L1), topk or batchtopk activation."""

    def __init__(self, d_in, n_latents, kind="topk", k=32, l1=0.0, seed=0):
        super().__init__()
        if kind not in ("relu", "topk", "batchtopk"):
            raise ValueError(kind)
        self.kind, self.k, self.l1 = kind, k, l1
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n_latents, d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w)
        self.W_enc = nn.Parameter(w.T.clone())  # tied initialization
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("threshold", torch.tensor(0.0))  # batchtopk inference threshold

    def pre_activations(self, x):
        """(n, d) -> (n, latents) before sparsification."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        """(n, d) -> sparse non-negative codes (n, latents)."""
        pre = F.relu(self.pre_activations(x))
        if self.kind == "relu":
            return pre
        if self.kind == "topk":
            vals, idx = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, idx, vals)
        if self.training:  # batchtopk
            flat = pre.flatten()
            vals, idx = flat.topk(min(self.k * pre.shape[0], flat.numel()))
            with torch.no_grad():
                kept_min = vals[-1] if len(vals) else torch.tensor(0.0)
                if self.threshold == 0:  # first batch: start the running average at the observed value
                    self.threshold.copy_(kept_min)
                else:
                    self.threshold.mul_(0.99).add_(0.01 * kept_min)
            return torch.zeros_like(flat).scatter(0, idx, vals).view_as(pre)
        return pre * (pre > self.threshold)

    def decode(self, z):
        """Codes -> reconstruction."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        """(reconstruction, codes, loss): mean squared error per token (+ L1 for relu)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(-1).mean()
        if self.kind == "relu":
            loss = loss + self.l1 * z.sum(-1).mean()
        return x_hat, z, loss

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder rows back to unit norm (after each optimizer step)."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def train_sae(sae, acts, steps, batch=4096, lr=1e-3, seed=0):
    """Adam on random mini-batches of activations (rows of `acts`). b_dec starts at the data mean.
    Returns the loss curve (every 50 steps)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        sae.b_dec.copy_(acts[:100_000].mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()
    curve = []
    for step in range(steps):
        idx = torch.randint(0, len(acts), (batch,), generator=g)
        _, _, loss = sae(acts[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % 50 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": loss.item()})
    sae.eval()
    return curve


@torch.no_grad()
def reconstruction_stats(sae, acts, batch=8192):
    """L0 (mean active latents per token), fraction of variance unexplained, and per-latent firing
    counts on held-out activations."""
    sae.eval()
    l0, sq_err, fires = 0.0, 0.0, torch.zeros(sae.W_dec.shape[0])
    for i in range(0, len(acts), batch):
        x = acts[i : i + batch]
        z = sae.encode(x)
        sq_err += float((sae.decode(z) - x).pow(2).sum())
        l0 += float((z > 0).sum())
        fires += (z > 0).sum(0).float()
    total_var = float((acts - acts.mean(0)).pow(2).sum())
    return {"l0": l0 / len(acts), "fvu": sq_err / total_var, "fires": fires}
