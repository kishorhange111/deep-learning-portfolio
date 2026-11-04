"""
The pieces of the reward-model overoptimization study: a policy over token sequences, a gold reward,
proxy reward models trained on preferences labelled by the gold reward, and the two ways of
optimizing against a proxy (best-of-n sampling and KL-regularized RL).

The setup follows Gao, Schulman and Hilton (2022), "Scaling Laws for Reward Model Overoptimization":
the "gold" reward model plays the role of human judgement, and proxies are trained on a limited
number of comparisons that it labels. Optimizing hard against a proxy first raises the gold reward
and then lowers it again, and the paper measures that curve against the KL distance the policy has
moved from where it started.

Everything is small enough to be exact where it matters:
  * The policy is autoregressive over sequences of L tokens from a vocabulary of V. Each position has
    its own next-token table conditioned on the previous token, so KL(policy || reference) has a
    closed form (a forward pass over the token marginals) and its gradient is exact.
  * The gold reward is built from fixed random networks on the one-hot sequence: a quality with
    diminishing, then negative, returns, optionally plus one with steady returns (see GoldReward).
  * Best-of-n uses the exact unbiased estimator over a large pool of samples, not repeated draws.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- policy
class MarkovPolicy(nn.Module):
    """pi(y) = prod_t pi_t(y_t | y_{t-1}): one (V, V) logit table per position (row = previous token).

    Position 0 uses row 0 of its table as an unconditional distribution."""

    def __init__(self, length, vocab, logits=None):
        super().__init__()
        self.length, self.vocab = length, vocab
        init = torch.zeros(length, vocab, vocab) if logits is None else logits.clone()
        self.logits = nn.Parameter(init)

    def log_tables(self):
        """(L, V, V) log-probabilities log pi_t(next | prev)."""
        return F.log_softmax(self.logits, dim=-1)

    @torch.no_grad()
    def sample(self, n, generator=None):
        """(n, L) int64 sequences."""
        probs = self.log_tables().exp()
        out = torch.empty(n, self.length, dtype=torch.long)
        prev = torch.zeros(n, dtype=torch.long)
        for t in range(self.length):
            p = probs[t, prev] if t > 0 else probs[0, 0].expand(n, -1)
            prev = torch.multinomial(p, 1, generator=generator).squeeze(1)
            out[:, t] = prev
        return out

    def log_prob(self, y):
        """(n,) log pi(y) for (n, L) sequences, differentiable."""
        lt = self.log_tables()
        prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)  # position 0 reads row 0
        t = torch.arange(self.length).expand_as(y)
        return lt[t, prev, y].sum(1)

    def marginals(self):
        """(L, V) probability of each token at each position (differentiable)."""
        probs = self.log_tables().exp()
        m = [probs[0, 0]]
        for t in range(1, self.length):
            m.append(m[-1] @ probs[t])
        return torch.stack(m)


def make_reference(length, vocab, seed, concentration=0.5):
    """A 'pretrained' reference policy: random Dirichlet next-token tables, so some sequences are rare."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(vocab, concentration), size=(length, vocab))
    return MarkovPolicy(length, vocab, torch.log(torch.from_numpy(probs).float()))


def exact_kl(policy, reference):
    """KL(policy || reference) over whole sequences, in nats: sum over positions of the expected
    per-step KL under the policy's distribution of the previous token. Differentiable in the policy."""
    lp, lr = policy.log_tables(), reference.log_tables().detach()
    step_kl = (lp.exp() * (lp - lr)).sum(-1)  # (L, V): KL of the next-token table, per previous token
    m = policy.marginals()
    return step_kl[0, 0] + (m[:-1] * step_kl[1:]).sum()


# ---------------------------------------------------------------- reward models
class SequenceRewardNet(nn.Module):
    """Reward network on the one-hot sequence: Linear(L*V, width) -> tanh -> Linear(width, 1)."""

    def __init__(self, length, vocab, width):
        super().__init__()
        self.length, self.vocab = length, vocab
        self.net = nn.Sequential(nn.Linear(length * vocab, width), nn.Tanh(), nn.Linear(width, 1))
        self.shift, self.scale = 0.0, 1.0

    def forward(self, y):
        """(n, L) tokens -> (n,) rewards (after the normalization set by `normalize`)."""
        x = F.one_hot(y, self.vocab).float().flatten(1)
        return (self.net(x).squeeze(-1) - self.shift) / self.scale

    @torch.no_grad()
    def normalize(self, samples):
        """Shift and scale so rewards have mean 0 and std 1 on `samples` (reference-policy draws)."""
        self.shift, self.scale = 0.0, 1.0
        r = self(samples)
        self.shift, self.scale = float(r.mean()), float(r.std())


def random_reward_net(length, vocab, seed, width=256, gain=1.5):
    """A fixed random reward network (no trainable parameters). `gain` scales the first layer: larger
    means tanh is further from linear, so more interactions between positions."""
    torch.manual_seed(seed)
    net = SequenceRewardNet(length, vocab, width)
    with torch.no_grad():
        nn.init.normal_(net.net[0].weight, std=gain / math.sqrt(length))
        nn.init.normal_(net.net[0].bias, std=1.0)
        nn.init.normal_(net.net[2].weight, std=1 / math.sqrt(width))
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net


class GoldReward(nn.Module):
    """The gold reward (the stand-in for human judgement): two independent qualities of a sequence.

    gold(y) = w_steady * s(y) + d(y) - d(y)^2 / (2 * peak)

    s and d are random networks, each normalized to mean 0 and std 1 under the reference policy.
    More d is better up to `peak` reference standard deviations and worse after that: diminishing
    and then negative returns, the shape behind length hacking, where readers like longer answers
    up to a point. Reference samples rarely reach the peak (it is a tail event, ~0.7 % of samples
    for peak = 2.5), so a proxy trained on them sees almost no data showing the turn. More s is
    always better; w_steady = 0 (the main study) makes d the whole reward."""

    def __init__(self, length, vocab, seed, peak=2.5, w_steady=0.0):
        super().__init__()
        self.length, self.vocab, self.peak, self.w_steady = length, vocab, peak, w_steady
        self.steady = random_reward_net(length, vocab, seed)
        self.diminishing = random_reward_net(length, vocab, seed + 1000)
        self.shift, self.scale = 0.0, 1.0

    def parts(self, y):
        """(steady, diminishing) qualities, each (n,)."""
        return self.steady(y), self.diminishing(y)

    def forward(self, y):
        """(n, L) tokens -> (n,) gold reward (after `normalize`)."""
        s, d = self.parts(y)
        raw = self.w_steady * s + d - d**2 / (2 * self.peak)
        return (raw - self.shift) / self.scale

    @torch.no_grad()
    def normalize(self, samples):
        """Normalize both qualities, then the total, to mean 0 and std 1 on reference samples."""
        self.steady.normalize(samples)
        self.diminishing.normalize(samples)
        self.shift, self.scale = 0.0, 1.0
        r = self(samples)
        self.shift, self.scale = float(r.mean()), float(r.std())


def preference_data(reference, gold, n_pairs, seed):
    """n_pairs comparisons between reference samples, labelled by the gold reward with Bradley-Terry
    noise: P(first preferred) = sigmoid(gold(first) - gold(second)). Returns (first, second, label)."""
    g = torch.Generator().manual_seed(seed)
    a, b = reference.sample(n_pairs, g), reference.sample(n_pairs, g)
    with torch.no_grad():
        p = torch.sigmoid(gold(a) - gold(b))
    label = (torch.rand(n_pairs, generator=g) < p).float()
    return a, b, label


def train_proxy(data, length, vocab, seed, width=128, epochs=200, lr=3e-3, wd=1e-4, val_frac=0.1):
    """Fit a proxy reward model to preference data with the Bradley-Terry loss, keeping the epoch with
    the best held-out loss. Returns (model, info)."""
    a, b, label = data
    n = len(label)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]
    torch.manual_seed(seed)
    model = SequenceRewardNet(length, vocab, width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    batch = min(512, len(tr))
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            loss = F.binary_cross_entropy_with_logits(model(a[idx]) - model(b[idx]), label[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = float(F.binary_cross_entropy_with_logits(model(a[va]) - model(b[va]), label[va]))
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= 20:
            break
    model.load_state_dict(best_state)
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, {"val_loss": best, "best_epoch": best_epoch, "n_train_pairs": len(tr)}


class Ensemble:
    """Combine proxy reward models: 'mean', 'min' (worst case) or 'uwo' (mean - lam * std,
    uncertainty-weighted optimization; Coste et al., 2023)."""

    def __init__(self, members, mode="mean", lam=1.0):
        self.members, self.mode, self.lam = members, mode, lam

    def __call__(self, y):
        """(n, L) tokens -> (n,) combined reward."""
        r = torch.stack([m(y) for m in self.members])
        if self.mode == "mean":
            return r.mean(0)
        if self.mode == "min":
            return r.min(0).values
        if self.mode == "uwo":
            return r.mean(0) - self.lam * r.std(0)
        raise ValueError(self.mode)


@torch.no_grad()
def preference_accuracy(proxy, gold, a, b):
    """How often the proxy orders a pair the same way the gold reward does (noise-free agreement)."""
    return float(((proxy(a) > proxy(b)) == (gold(a) > gold(b))).float().mean())


# ---------------------------------------------------------------- best-of-n
def bon_kl(n):
    """KL(best-of-n || reference) = log n - (n - 1) / n (Stiennon et al., 2020)."""
    return math.log(n) - (n - 1) / n


def best_of_n_curve(proxy_scores, gold_scores, ns, extra=None):
    """Exact expected gold and proxy reward of best-of-n selection by proxy, for each n, from one pool.
    `extra` ({name: scores}) are averaged with the same weights (e.g. the parts of the gold reward).

    Sort the pool of N samples by proxy score. In a uniformly random n-subset, the i-th smallest
    (0-based) is the arg-max with probability C(i, n - 1) / C(N, n), so the expectation needs no
    resampling (the unbiased estimator of Nakano et al., 2021)."""
    order = np.argsort(proxy_scores, kind="stable")
    p_sorted, g_sorted = np.asarray(proxy_scores)[order], np.asarray(gold_scores)[order]
    extra_sorted = {k: np.asarray(v)[order] for k, v in (extra or {}).items()}
    big = len(order)
    lfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, big + 1)))])  # lfact[k] = log k!
    i = np.arange(big)
    rows = []
    for n in ns:
        if n > big:
            break
        ii = i[n - 1 :]  # C(i, n - 1) = 0 below this
        logw = lfact[ii] - lfact[n - 1] - lfact[ii - n + 1]  # log C(i, n - 1); the constant C(N, n) cancels
        w = np.exp(logw - logw.max())
        w /= w.sum()
        row = {
            "n": int(n),
            "kl": bon_kl(n),
            "gold": float(w @ g_sorted[n - 1 :]),
            "proxy": float(w @ p_sorted[n - 1 :]),
        }
        row.update({k: float(w @ v[n - 1 :]) for k, v in extra_sorted.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- RL
def rl_optimize(reference, reward_fn, gold, steps, batch=512, lr=0.01, kl_coef=0.0, eval_every=20, eval_n=4096, seed=0):
    """Maximize E[reward_fn(y)] - kl_coef * KL(pi || reference) from pi = reference.

    The reward term uses REINFORCE with a leave-one-out baseline (RLOO); the KL term is exact.
    Logs the exact KL, and the gold and proxy reward of fresh samples (plus the gold reward's parts,
    if it has them), every `eval_every` steps."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    policy = MarkovPolicy(reference.length, reference.vocab, reference.logits.detach())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log = []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                y = policy.sample(eval_n, g)
                rec = {
                    "step": step,
                    "kl": float(exact_kl(policy, reference)),
                    "gold": float(gold(y).mean()),
                    "proxy": float(reward_fn(y).mean()),
                }
                if hasattr(gold, "parts"):
                    s_part, d_part = gold.parts(y)
                    rec["steady"], rec["diminishing"] = float(s_part.mean()), float(d_part.mean())
                log.append(rec)
        if step == steps:
            break
        y = policy.sample(batch, g)
        with torch.no_grad():
            r = reward_fn(y)
            adv = r - (r.sum() - r) / (batch - 1)  # leave-one-out baseline
        loss = -(adv * policy.log_prob(y)).mean()
        if kl_coef:
            loss = loss + kl_coef * exact_kl(policy, reference)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return log


# ---------------------------------------------------------------- functional forms (Gao et al.)
def fit_functional_form(d, gold, form):
    """Least-squares fit of gold(d) with d = sqrt(KL):
    'bon': d (alpha - beta d);  'rl': d (alpha - beta log d). Returns (alpha, beta, r2)."""
    d, gold = np.asarray(d, dtype=float), np.asarray(gold, dtype=float)
    keep = d > 0
    d, gold = d[keep], gold[keep]
    basis = np.stack([d, -d * d if form == "bon" else -d * np.log(d)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, gold, rcond=None)
    pred = basis @ coef
    ss_res, ss_tot = ((gold - pred) ** 2).sum(), ((gold - gold.mean()) ** 2).sum()
    return float(coef[0]), float(coef[1]), float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
