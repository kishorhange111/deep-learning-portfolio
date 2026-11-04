"""
Weight-only post-training quantization, written from the papers: round-to-nearest (RTN) and GPTQ.

Both store each weight matrix W (out x in) as b-bit integers q plus a scale s and zero point z per
group of input columns, W ~ s * (q - z). Asymmetric min-max ranges: s = (max - min) / (2^b - 1),
z = round(-min / s). With group_size = None there is one (s, z) per output row ("per channel").

RTN rounds every weight independently. GPTQ (Frantar et al., 2023, "GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers") quantizes the columns of W one at a time and,
after each, updates the not-yet-quantized columns to cancel the error it just made on calibration
inputs X. It minimizes ||W X - Q X||^2 using H = 2 X X^T / n (the Hessian of that objective):
  for column i:  q_i = quant(w_i);  err = (w_i - q_i) / [H^-1]_ii;  W[:, i+1:] -= err [H^-1]_{i, i+1:}
using the Cholesky factor of H^-1 so the whole pass costs one decomposition, in blocks of 128
columns ("lazy batch" updates). When H is a multiple of the identity (uncorrelated inputs) there is
nothing to compensate and GPTQ reduces exactly to RTN, which the tests check.
"""

import torch
from torch import nn


def minmax_params(w, bits):
    """Asymmetric scale and zero point per row of w (rows x cols) for b-bit codes 0..2^b - 1."""
    qmax = 2**bits - 1
    lo = torch.minimum(w.min(dim=1, keepdim=True).values, torch.zeros(1, dtype=w.dtype))
    hi = torch.maximum(w.max(dim=1, keepdim=True).values, torch.zeros(1, dtype=w.dtype))
    scale = (hi - lo) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # all-zero rows
    zero = torch.round(-lo / scale)
    return scale, zero


def quantize_with(w, scale, zero, bits):
    """Round w to the grid scale * (q - zero), q in 0..2^b - 1; returns the dequantized values."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1)
    return scale * (q - zero)


def rtn(weight, bits, group_size=None):
    """Round-to-nearest quantization of a (out, in) matrix; returns the dequantized matrix."""
    w = weight.detach().float()
    if group_size is None:
        scale, zero = minmax_params(w, bits)
        return quantize_with(w, scale, zero, bits).to(weight.dtype)
    out = torch.empty_like(w)
    for g in range(0, w.shape[1], group_size):
        cols = w[:, g : g + group_size]
        scale, zero = minmax_params(cols, bits)
        out[:, g : g + group_size] = quantize_with(cols, scale, zero, bits)
    return out.to(weight.dtype)


def gptq(weight, hessian, bits, group_size=None, block_size=128, damp=0.01):
    """GPTQ quantization of a (out, in) matrix given H = 2 X X^T / n over calibration inputs (in x in).

    Returns the dequantized matrix. `damp` adds damp * mean(diag H) to the diagonal for stability."""
    w = weight.detach().clone().float()
    h = hessian.detach().clone().float()
    n_cols = w.shape[1]
    if group_size is not None and block_size % group_size:
        raise ValueError("block_size must be a multiple of group_size")
    dead = torch.diag(h) == 0  # inputs that are always zero: their weights do not matter
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(n_cols)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    u = torch.linalg.cholesky(h_inv, upper=True)  # rows of U give the error-propagation weights

    q_out = torch.zeros_like(w)
    scale = zero = None
    if group_size is None:
        scale, zero = minmax_params(w, bits)
    for b0 in range(0, n_cols, block_size):
        b1 = min(b0 + block_size, n_cols)
        w_blk = w[:, b0:b1].clone()
        err_blk = torch.zeros_like(w_blk)
        u_blk = u[b0:b1, b0:b1]
        for i in range(b1 - b0):
            if group_size is not None and (b0 + i) % group_size == 0:
                scale, zero = minmax_params(w_blk[:, i : i + group_size], bits)  # from the updated weights
            col = w_blk[:, i]
            q = quantize_with(col.unsqueeze(1), scale, zero, bits).squeeze(1)
            q_out[:, b0 + i] = q
            err = (col - q) / u_blk[i, i]
            w_blk[:, i:] -= err.unsqueeze(1) @ u_blk[i, i:].unsqueeze(0)
            err_blk[:, i] = err
        w[:, b1:] -= err_blk @ u[b0:b1, b1:]
    return q_out.to(weight.dtype)


def effective_bits(n_rows, n_cols, bits, group_size=None, param_bits=16):
    """Average storage bits per weight, counting a 16-bit scale and 16-bit zero point per group."""
    groups_per_row = 1 if group_size is None else -(-n_cols // group_size)
    return bits + 2 * param_bits * groups_per_row / n_cols


def block_linears(model):
    """{name: nn.Linear} for the linear layers inside the transformer blocks (embeddings and head stay in float)."""
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("blocks.")}


@torch.no_grad()
def collect_hessians(model, layers, calib_x, batch=32):
    """H = 2 X X^T / n for the inputs of each named linear layer, over calibration windows."""
    hess = {n: torch.zeros(m.in_features, m.in_features) for n, m in layers.items()}
    counts = dict.fromkeys(layers, 0)
    hooks = []
    for name, mod in layers.items():

        def hook(_m, inputs, _out, name=name):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            hess[name] += 2 * x.T @ x
            counts[name] += x.shape[0]

        hooks.append(mod.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        for i in range(0, len(calib_x), batch):
            model(torch.as_tensor(calib_x[i : i + batch]))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return {n: hess[n] / max(1, counts[n]) for n in hess}


@torch.no_grad()
def quantize_model(model, method, bits, group_size=None, calib_x=None, only=None):
    """Quantize the block linears of `model` in place. GPTQ runs block by block (as in the paper): each
    block's Hessians are collected with the earlier blocks already quantized, so later layers adapt
    to the errors of earlier ones. `only` restricts quantization to layers whose name contains it.
    Returns {layer name: relative weight error ||W - Q||_F / ||W||_F}."""
    layers = block_linears(model)
    if only is not None:
        layers = {n: m for n, m in layers.items() if only in n}
    errors = {}
    n_blocks = len(model.blocks)
    for b in range(n_blocks):
        group = {n: m for n, m in layers.items() if n.startswith(f"blocks.{b}.")}
        if not group:
            continue
        hess = collect_hessians(model, group, calib_x) if method == "gptq" else {}
        for name, mod in group.items():
            w = mod.weight.data
            if method == "rtn":
                q = rtn(w, bits, group_size)
            elif method == "gptq":
                q = gptq(w, hess[name], bits, group_size)
            else:
                raise ValueError(method)
            errors[name] = float((w - q).norm() / w.norm())
            mod.weight.data.copy_(q)
    return errors
