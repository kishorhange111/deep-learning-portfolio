"""
Statistics for small evals: bootstrap confidence intervals, paired comparisons and rater agreement.

With 900 questions, a 2-point accuracy difference between two models can be noise or real; the
paired bootstrap (resampling questions, keeping each model's answers on the same question together)
says which. For judge validation, Spearman correlation and quadratic-weighted Cohen's kappa measure
how well an LLM judge's 1-5 scores agree with a human's.
"""

import numpy as np


def bootstrap_ci(values, n_boot=10_000, seed=0, alpha=0.05):
    """Mean and percentile CI of a 1-D array."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (
        float(v.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Mean of a - b over paired items, 95 % CI and two-sided p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p": float(p),
        "n": int(len(d)),
    }


def _ranks(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for value in np.unique(x):
        idx = x == value
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a, b):
    """Spearman rank correlation (Pearson on average ranks)."""
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quadratic_weighted_kappa(a, b, min_rating=1, max_rating=5):
    """Cohen's kappa with quadratic weights for ordinal ratings: 1 = perfect, 0 = chance agreement."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    k = max_rating - min_rating + 1
    observed = np.zeros((k, k))
    for x, y in zip(a - min_rating, b - min_rating):
        observed[x, y] += 1
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    i, j = np.indices((k, k))
    weights = (i - j) ** 2 / (k - 1) ** 2
    denom = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denom) if denom else float("nan")


def agreement(a, b):
    """All agreement statistics between two raters' 1-5 scores."""
    a, b = np.asarray(a), np.asarray(b)
    return {
        "n": int(len(a)),
        "spearman": spearman(a, b),
        "qwk": quadratic_weighted_kappa(a, b),
        "exact": float((a == b).mean()),
        "within_1": float((np.abs(a - b) <= 1).mean()),
        "mean_diff": float((a - b).mean()),
    }
