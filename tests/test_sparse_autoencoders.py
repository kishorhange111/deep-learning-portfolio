"""Tests for interpretability/sparse-autoencoders: sparsity mechanisms, decoder normalization,
recovery of a planted dictionary, the purity statistic, and an end-to-end quick run."""

import numpy as np
import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "interpretability/sparse-autoencoders"
sl = load_module(PROJECT, "sae_lib.py")


def test_topk_and_batchtopk_sparsity():
    torch.manual_seed(0)
    x = torch.randn(50, 16)
    topk = sl.SparseAutoencoder(16, 64, kind="topk", k=5)
    z = topk.encode(x)
    assert ((z > 0).sum(1) <= 5).all() and (z >= 0).all()
    btk = sl.SparseAutoencoder(16, 64, kind="batchtopk", k=5).train()
    z = btk.encode(x)
    assert int((z > 0).sum()) <= 5 * 50 and float(btk.threshold) > 0
    btk.eval()
    assert torch.equal(btk.encode(x) > 0, torch.relu(btk.pre_activations(x)) > btk.threshold)


def test_decoder_rows_stay_unit_norm_and_l1_buys_sparsity():
    torch.manual_seed(0)
    acts = torch.randn(4000, 16)
    l0 = {}
    for l1 in (0.01, 3.0):
        sae = sl.SparseAutoencoder(16, 64, kind="relu", l1=l1)
        sl.train_sae(sae, acts, steps=150, batch=512, lr=3e-3)
        assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)
        l0[l1] = sl.reconstruction_stats(sae, acts)["l0"]
    assert l0[3.0] < 0.7 * l0[0.01]


def test_topk_sae_recovers_a_planted_dictionary():
    """Data = sparse non-negative combinations of 32 random unit atoms: the SAE should find them."""
    g = torch.Generator().manual_seed(0)
    atoms = torch.randn(32, 24, generator=g)
    atoms = atoms / atoms.norm(dim=1, keepdim=True)
    n = 20_000
    codes = torch.zeros(n, 32)
    idx = torch.stack([torch.randperm(32, generator=g)[:3] for _ in range(n)])
    codes.scatter_(1, idx, torch.rand(n, 3, generator=g) + 0.5)
    acts = codes @ atoms
    sae = sl.SparseAutoencoder(24, 64, kind="topk", k=3, seed=1)
    sl.train_sae(sae, acts, steps=1500, batch=1024, lr=3e-3)
    stats = sl.reconstruction_stats(sae, acts)
    assert stats["fvu"] < 0.05
    best_cos = (atoms @ sae.W_dec.detach().T).max(dim=1).values
    assert (best_cos > 0.9).float().mean() >= 0.9  # nearly every planted atom has a matching latent


def test_purity_statistic_on_constructed_activations():
    train = load_module(PROJECT, "train.py")
    cur = np.array([5] * 10 + list(range(10)))
    nxt = np.array([7] * 20)
    scores = torch.zeros(20, 2)
    scores[:10, 0] = torch.arange(10, 0, -1).float()  # direction 0 fires on byte 5 only
    scores[10:, 1] = torch.arange(10, 0, -1).float()  # direction 1 fires on ten different bytes
    p = train.purity_stats(scores, cur, nxt, top_n=10)
    assert p["current"] == [1.0, 0.1] and p["mode_current"][0] == 5 and p["next"] == [1.0, 1.0]


def test_quick_study_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    out = train.main(["--out_dir", str(tmp_path), "--threads", "1"])
    assert len(out["saes"]) == 1 + 2 * 2 and out["clean_bpb"] < out["mean_ablated_bpb"]
    assert {"sae", "residual_dims", "random_dirs"} == set(out["features"]["purity"])
    assert (tmp_path / "features.md").exists()
    analyze = load_module(PROJECT, "analyze.py")
    s = analyze.main(str(tmp_path / "study.json"), str(tmp_path / "out"))
    assert s["rows"][0]["kind"] == "relu" and (tmp_path / "out" / "frontier.png").exists()
