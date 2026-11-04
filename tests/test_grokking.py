"""Tests for interpretability/grokking-modular-arithmetic: the data, the last-position-only forward pass,
and the Fourier tools on inputs whose answer is known by construction."""

import json
import math

import numpy as np
import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "interpretability/grokking-modular-arithmetic"
gm = load_module(PROJECT, "grok_model.py")


def test_dataset_labels_and_split():
    x, y, tr, te = gm.make_dataset(11, 0.3, seed=0)
    assert x.shape == (121, 3) and bool((x[:, 2] == 11).all())
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % 11)
    assert len(tr) == round(0.3 * 121) and len(set(tr.tolist()) | set(te.tolist())) == 121
    assert not set(tr.tolist()) & set(te.tolist())


def full_causal_forward(model, x):
    """Reference: the standard causal 1-layer transformer over all three positions, read at the last."""
    resid = model.W_E[x] + model.W_pos
    q = torch.einsum("bpd,hde->bhpe", resid, model.W_Q)
    k = torch.einsum("bpd,hde->bhpe", resid, model.W_K)
    v = torch.einsum("bpd,hde->bhpe", resid, model.W_V)
    scores = torch.einsum("bhqe,bhke->bhqk", q, k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), 1), float("-inf"))
    z = torch.einsum("bhqk,bhke->bhqe", torch.softmax(scores, -1), v)
    mid = resid + torch.einsum("bhqe,hed->bqd", z, model.W_O)
    out = mid + torch.relu(mid @ model.W_in + model.b_in) @ model.W_out + model.b_out
    return (out @ model.W_U)[:, -1]


def test_last_position_forward_equals_full_causal_model():
    torch.manual_seed(0)
    model = gm.ModAddTransformer(gm.GrokConfig(p=13, d_model=32, n_head=4, d_mlp=64))
    with torch.no_grad():
        model.b_in.normal_()
        model.b_out.normal_()
    x, *_ = gm.make_dataset(13, 0.5, 0)
    with torch.no_grad():
        assert torch.allclose(model(x), full_causal_forward(model, x), atol=1e-5)


def test_fourier_basis_is_orthonormal():
    f = gm.fourier_basis(13)
    assert np.allclose(f @ f.T, np.eye(13), atol=1e-10)


def single_frequency_model(p, k, d=8):
    """A model whose token embedding uses only frequency k."""
    model = gm.ModAddTransformer(gm.GrokConfig(p=p, d_model=d, n_head=2, d_mlp=16))
    a = np.arange(p)
    emb = np.zeros((p + 1, d))
    emb[:p, 0], emb[:p, 1] = np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)
    with torch.no_grad():
        model.W_E.copy_(torch.from_numpy(emb).float())
    return model


def test_embedding_sparsity_measures():
    norms = gm.embedding_frequency_norms(single_frequency_model(17, 3))
    assert norms[2] == pytest.approx(1.0, abs=1e-6) and norms.sum() == pytest.approx(1.0, abs=1e-6)
    assert gm.gini(norms) > 0.85
    assert gm.gini(np.ones(8)) == pytest.approx(0.0, abs=1e-12)


def fourier_algorithm_logits(p, keys, scale=4.0):
    """logit(a, b, c) = scale * sum_k cos(w_k (a + b - c)): the algorithm the grokked model implements."""
    a, b, c = np.meshgrid(np.arange(p), np.arange(p), np.arange(p), indexing="ij")
    logits = sum(scale * np.cos(2 * np.pi * k * (a + b - c) / p) for k in keys)
    return torch.from_numpy(logits.astype(np.float64))


def test_key_frequencies_and_progress_measures_on_the_ideal_algorithm():
    p, keys = 23, [2, 7, 9]
    logits = fourier_algorithm_logits(p, keys)
    assert sorted(gm.key_frequencies(logits)) == keys
    assert gm.frequency_shares_of_logits(logits)[[k - 1 for k in keys]].sum() == pytest.approx(1.0, abs=1e-9)
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    y = torch.from_numpy((a + b) % p)
    mask = torch.zeros(p * p, dtype=torch.bool)
    mask[::3] = True
    restricted, excluded = gm.restricted_and_excluded_loss(logits, y, keys, mask)
    full = float(torch.nn.functional.cross_entropy(logits.reshape(p * p, p), y.flatten()))
    assert restricted == pytest.approx(full, abs=1e-9)  # nothing outside the key frequencies to remove
    assert excluded == pytest.approx(math.log(p), abs=1e-6)  # nothing left once they are removed


def test_excluded_loss_is_unchanged_by_removing_unused_frequencies():
    """Memorization-like logits (random) have little mass in any few frequencies."""
    p = 23
    torch.manual_seed(0)
    logits = torch.randn(p, p, p, dtype=torch.float64) * 3
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    y = torch.from_numpy((a + b) % p)
    mask = torch.ones(p * p, dtype=torch.bool)
    _, excluded = gm.restricted_and_excluded_loss(logits, y, [4], mask)
    full = float(torch.nn.functional.cross_entropy(logits.reshape(p * p, p), y.flatten()))
    assert excluded == pytest.approx(full, rel=0.1)


class FakeMLPModel:
    """Stands in for the model: fills the cache with chosen MLP activations over the (a, b) grid."""

    def __init__(self, p, acts):
        self.cfg = gm.GrokConfig(p=p)
        self.acts = acts

    def __call__(self, x, cache=None):
        cache["mlp_post"] = torch.from_numpy(self.acts.reshape(-1, self.acts.shape[-1]))


def test_neuron_purity():
    p = 19
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    w = 2 * np.pi / p
    pure = 2 + np.cos(5 * w * a) * np.cos(5 * w * b)  # one frequency: purity 1
    mixed = 2 + np.cos(3 * w * a) + np.cos(8 * w * b)  # two frequencies, equal energy: purity 0.5
    dead = np.zeros((p, p))
    x, *_ = gm.make_dataset(p, 0.5, 0)
    best, purity = gm.neuron_frequency_purity(FakeMLPModel(p, np.stack([pure, mixed, dead], -1)), x)
    assert best[0] == 5 and purity[0] == pytest.approx(1.0, abs=1e-9)
    assert purity[1] == pytest.approx(0.5, abs=1e-9) and np.isnan(purity[2])


def test_quick_study_writes_runs_and_figures(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    results = train.main(["--out_dir", str(tmp_path / "runs")])
    assert set(results) == {"main_s0", "frac0.5_s0"}
    r = json.loads((tmp_path / "runs" / "main_s0.json").read_text())
    assert r["log"][-1]["train_loss"] < r["log"][0]["train_loss"]
    assert len(r["progress"]) == len(range(0, 61, 20)) and "restricted_loss" in r["progress"][0]
    assert len(r["final"]["embedding_norm_by_frequency"]) == 11
    analyze = load_module(PROJECT, "analyze.py")
    rows = analyze.main(runs_dir=str(tmp_path / "runs"), out_dir=str(tmp_path / "out"))
    assert set(rows) == {"main_s0", "frac0.5_s0"}
    for f in ("summary.md", "grokking.png", "progress_measures.png", "fourier_mechanism.png", "data_fraction.png"):
        assert (tmp_path / "out" / f).exists()
