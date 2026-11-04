"""Tests for the triplet evaluation helper (src/dl_common/embeddings.py)."""

import numpy as np
import pytest

from dl_common.embeddings import l2_normalize, triplet_batch_stats


def test_l2_normalize():
    v = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], [0.6, 0.8])


def test_triplet_batch_stats():
    ea = np.array([[1.0, 0.0], [1.0, 0.0]])
    ep = np.array([[0.9, 0.1], [-1.0, 0.0]])  # first positive close, second far
    en = np.array([[0.0, 1.0], [0.8, 0.2]])
    correct, total, cos_pos, cos_neg = triplet_batch_stats(ea, ep, en)
    assert (correct, total) == (1, 2)
    assert cos_pos[1] == pytest.approx(-1.0)
    assert cos_neg[0] == pytest.approx(0.0)
