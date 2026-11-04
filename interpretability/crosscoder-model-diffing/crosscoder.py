"""
A sparse crosscoder for model diffing (Lindsey et al., Anthropic 2024).

A crosscoder is a sparse autoencoder that reads the same layer from *two* models on the same tokens
and learns one shared dictionary of latents, each with its own decoder direction in each model:

    f      = sparsify(  W_enc^A x^A + W_enc^B x^B + b_enc )        latent activations (F,)
    x^m_hat = W_dec^m f + b_dec^m                                  per-model reconstruction

Comparing a latent's decoder norms says which model uses it. The relative norm is
r = ||d^B|| / (||d^A|| + ||d^B||): r ~ 0.5 means the latent is shared, r ~ 1 means it exists only in
model B (e.g. a concept introduced by fine-tuning), and r ~ 0 means only in model A.

Two sparsity schemes:
  "l1"        the paper's loss: L1 on latents weighted by the summed decoder norms
              sum_i f_i (||d_i^A|| + ||d_i^B||)
  "batchtopk" keep the k * batch largest f_i (||d_i^A|| + ||d_i^B||) across the batch. Minder et
              al. (2025) found that L1 crosscoders can label shared concepts as model-specific
              (shrinkage and decoupling artifacts), and BatchTopK training reduces this. At
              inference a learned global threshold replaces the batch-level top-k.
"""

import torch
import torch.nn as nn


class Crosscoder(nn.Module):
    """Two-model sparse crosscoder; inputs are (batch, 2, d_model) activations."""

    def __init__(self, d_model, n_latents, sparsity="batchtopk", k=64, dec_init_norm=0.08):
        super().__init__()
        self.d_model, self.n_latents, self.sparsity, self.k = d_model, n_latents, sparsity, k
        w = torch.randn(n_latents, 2, d_model)
        w = w / w.norm(dim=-1, keepdim=True) * dec_init_norm
        self.W_dec = nn.Parameter(w)  # (F, 2, d)
        self.W_enc = nn.Parameter(w.permute(1, 2, 0).clone())  # (2, d, F): tied at init, then free
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(2, d_model))
        self.register_buffer("threshold", torch.tensor(0.0))  # BatchTopK inference threshold (EMA)

    def decoder_norms(self):
        """(F, 2): ||d_i^m|| for every latent i and model m."""
        return self.W_dec.norm(dim=-1)

    def pre_activations(self, x):
        """ReLU(sum_m x^m W_enc^m + b_enc) -> (batch, F)."""
        return torch.relu(torch.einsum("bmd,mdf->bf", x, self.W_enc) + self.b_enc)

    def encode(self, x, train_mode=None):
        """Sparse latent activations (batch, F)."""
        f = self.pre_activations(x)
        if self.sparsity == "l1":
            return f
        score = f * self.decoder_norms().sum(-1)  # rank latents by their contribution to the reconstruction
        if train_mode if train_mode is not None else self.training:
            kth = torch.topk(score.flatten(), min(self.k * x.shape[0], score.numel())).values[-1]
            with torch.no_grad():  # running estimate of the cut-off, used at inference
                self.threshold.mul_(0.99).add_(0.01 * kth)
            return f * (score >= kth)
        return f * (score > self.threshold)

    def decode(self, f):
        """(batch, F) -> (batch, 2, d)."""
        return torch.einsum("bf,fmd->bmd", f, self.W_dec) + self.b_dec

    def forward(self, x):
        """Returns (reconstruction, latent activations)."""
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x, l1_coeff=0.0):
        """Summed squared error over both models (mean over the batch) + the L1 term if sparsity == 'l1'."""
        x_hat, f = self(x)
        mse = (x_hat - x).pow(2).sum(dim=(1, 2)).mean()
        l1 = (
            (f * self.decoder_norms().sum(-1)).sum(-1).mean()
            if self.sparsity == "l1"
            else torch.zeros((), device=x.device)
        )
        stats = {"mse": mse.item(), "l0": (f > 0).float().sum(-1).mean().item()}
        return mse + l1_coeff * l1, f, stats


def fraction_variance_explained(x, x_hat):
    """Per-model FVE: 1 - SSE / total variance, for (batch, 2, d) tensors -> (2,)."""
    sse = (x - x_hat).pow(2).sum(dim=(0, 2))
    var = (x - x.mean(0, keepdim=True)).pow(2).sum(dim=(0, 2))
    return 1 - sse / var


def relative_norms(model):
    """r_i = ||d_i^B|| / (||d_i^A|| + ||d_i^B||) for every latent."""
    n = model.decoder_norms()
    return n[:, 1] / n.sum(-1).clamp(min=1e-8)


def decoder_cosines(model):
    """Cosine similarity between each latent's two decoder directions (are 'shared' latents really the same?)."""
    a, b = model.W_dec[:, 0], model.W_dec[:, 1]
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def classify(r, only_a=0.1, only_b=0.9, shared=(0.3, 0.7)):
    """Masks of A-only, B-only and shared latents from relative norms."""
    return {"A-only": r < only_a, "B-only": r > only_b, "shared": (r > shared[0]) & (r < shared[1])}
