"""
NumPy helpers for evaluating embedding models (used by the triplet-loss project).

All functions take plain arrays of shape (batch, dim), so they are framework-
independent and unit-tested in `tests/test_embeddings.py`.
"""

import numpy as np


def l2_normalize(v):
    """Scale every row of `v` to unit length (so a dot product becomes a cosine similarity)."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def triplet_batch_stats(ea, ep, en):
    """Evaluate one batch of (anchor, positive, negative) embeddings.

    Returns
        correct  - number of triplets with d(A, P) < d(A, N), using squared Euclidean distance
                   (the same distance the triplet loss is trained on)
        total    - number of triplets in the batch
        cos_pos  - list of cosine similarities anchor-positive
        cos_neg  - list of cosine similarities anchor-negative
    """
    dp, dn = ((ea - ep) ** 2).sum(-1), ((ea - en) ** 2).sum(-1)
    na, np_, nn = l2_normalize(ea), l2_normalize(ep), l2_normalize(en)
    return int((dp < dn).sum()), len(dp), list((na * np_).sum(-1)), list((na * nn).sum(-1))
