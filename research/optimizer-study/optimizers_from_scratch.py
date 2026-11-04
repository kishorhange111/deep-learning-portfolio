"""
Muon and Lion written from their papers, plus the parameter grouping shared by every optimizer in
the study.

Muon (Jordan et al., 2024, "Muon: An optimizer for hidden layers in neural networks"):
  momentum SGD whose update for each weight matrix is replaced by the nearest semi-orthogonal
  matrix, U V^T from the SVD of the (Nesterov) momentum. All singular values of the update become
  ~1, so rare directions get as much step as common ones. The orthogonalization is approximated
  with five Newton-Schulz iterations of a quintic polynomial, cheap matrix multiplies instead of an
  SVD. Muon is only for 2-D hidden weights; embeddings, the output head and LayerNorm parameters
  use AdamW.

Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms"):
  update = sign(beta1 * m + (1 - beta1) * g), then m = beta2 * m + (1 - beta2) * g. Every
  coordinate moves by exactly lr, so the learning rate must be ~3-10x smaller than AdamW's and
  decoupled weight decay ~3-10x larger for the same effective decay.
"""

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz coefficients from the Muon reference code


def newton_schulz_orthogonalize(g, steps=5, eps=1e-7):
    """Approximate U V^T for g = U S V^T by iterating X <- a X + (b A + c A^2) X with A = X X^T.

    The coefficients are tuned for speed rather than exact convergence: singular values end up in
    roughly [0.7, 1.2], which works as well as the exact orthogonal factor in practice."""
    a, b, c = NS_COEFFS
    x = g.float() / (g.norm() + eps)  # spectral norm <= Frobenius norm <= 1, inside the convergence region
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T  # iterate on the smaller Gram matrix
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if transposed else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters: Nesterov momentum, then Newton-Schulz orthogonalization, scaled by
    sqrt(max(1, rows / cols)) so tall matrices get comparable per-element steps."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(
            params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        )
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError("Muon is only for 2-D weight matrices; use AdamW for the rest")

    @torch.no_grad()
    def step(self, closure=None):
        """One update of every parameter that has a gradient."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                buf = state.setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz_orthogonalize(g, group["ns_steps"])
                update *= max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


class Lion(torch.optim.Optimizer):
    """Lion: sign of an interpolated momentum, with decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """One update of every parameter that has a gradient."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("exp_avg", torch.zeros_like(p))
                update = (m * beta1 + p.grad * (1 - beta1)).sign_()
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss


class Combined:
    """Several optimizers stepped together (Muon for hidden matrices plus AdamW for the rest)."""

    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def step(self):
        """Step every optimizer."""
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none=True):
        """Clear every optimizer's gradients."""
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)


def split_parameters(model):
    """(hidden matrices, embeddings and head, vectors): the 2-D weights inside the blocks, the
    input/output matrices, and LayerNorm parameters. Every optimizer uses the same split."""
    hidden, io, vectors = [], [], []
    for name, p in model.named_parameters():
        if p.dim() < 2:
            vectors.append(p)
        elif name.startswith("blocks."):
            hidden.append(p)
        else:
            io.append(p)
    return hidden, io, vectors


def make_optimizer(name, model, lr, weight_decay=0.1, adam_lr=3e-3):
    """The study's optimizers. Weight decay applies to matrices only, never to LayerNorm parameters.

    * adamw: AdamW(betas 0.9 / 0.95) on everything, at `lr`.
    * lion:  Lion(betas 0.9 / 0.99) on everything, at `lr`, with 10x the weight decay (lr is ~10x smaller).
    * muon:  Muon(momentum 0.95, Nesterov) at `lr` on the hidden matrices; AdamW at `adam_lr` on
             embeddings, head and LayerNorm parameters (as in the Muon reference setup).
    Returns an object with step() / zero_grad() and param_groups (each carrying an 'lr' to schedule)."""
    hidden, io, vectors = split_parameters(model)
    if name == "adamw":
        groups = [{"params": hidden + io, "weight_decay": weight_decay}, {"params": vectors, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    if name == "lion":
        groups = [{"params": hidden + io, "weight_decay": 10 * weight_decay}, {"params": vectors, "weight_decay": 0.0}]
        return Lion(groups, lr=lr, betas=(0.9, 0.99))
    if name == "muon":
        adam_groups = [{"params": io, "weight_decay": weight_decay}, {"params": vectors, "weight_decay": 0.0}]
        return Combined(Muon(hidden, lr=lr), torch.optim.AdamW(adam_groups, lr=adam_lr, betas=(0.9, 0.95)))
    raise ValueError(name)
