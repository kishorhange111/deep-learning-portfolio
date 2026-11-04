"""Tests for research/optimizer-study: Muon's Newton-Schulz step, Lion, the parameter split, training
with each optimizer, and an end-to-end quick run."""

import numpy as np
import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
from dl_common import byte_corpus as bc  # noqa: E402
from dl_common.bytelm import ByteGPT, ByteGPTConfig  # noqa: E402

OPT = "research/optimizer-study"


def tiny_cfg(**kw):
    return ByteGPTConfig(**{"seq_len": 16, "n_layer": 2, "d_model": 32, "n_head": 4, "d_ff": 64, **kw})


# ---------------------------------------------------------------- optimizer study
ofs = load_module(OPT, "optimizers_from_scratch.py")


def with_spectrum(shape, sigma):
    """A (rows, cols) matrix with singular values sigma and random singular vectors; returns (G, U V^T)."""
    k = min(shape)
    u = torch.linalg.qr(torch.randn(shape[0], k))[0]
    v = torch.linalg.qr(torch.randn(shape[1], k))[0]
    return u @ torch.diag(sigma) @ v.T, u @ v.T


@pytest.mark.parametrize("shape", [(32, 32), (64, 16), (16, 64)])
def test_newton_schulz_flattens_the_spectrum(shape):
    torch.manual_seed(0)
    k = min(shape)
    g, polar = with_spectrum(shape, torch.logspace(-1, 0, k))  # condition number 10
    out = ofs.newton_schulz_orthogonalize(g)
    s = torch.linalg.svdvals(out)
    assert s.min() > 0.6 and s.max() < 1.25  # every direction gets a similar step
    assert (out * polar).sum() / polar.norm() ** 2 > 0.85  # close to the exact factor U V^T
    bad, _ = with_spectrum(shape, torch.logspace(-3, 0, k))  # condition number 1000
    s_out = torch.linalg.svdvals(ofs.newton_schulz_orthogonalize(bad))
    assert s_out.max() / s_out.min() < 100  # five steps shrink the spread a lot but do not remove it


def test_lion_moves_every_coordinate_by_lr():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = ofs.Lion([p], lr=0.1)
    p.grad = torch.tensor([3.0, -0.001, 2.0, -5.0, 0.5])
    opt.step()
    assert torch.allclose(p.data, torch.tensor([-0.1, 0.1, -0.1, 0.1, -0.1]))


def test_muon_rejects_vectors_and_every_optimizer_trains():
    with pytest.raises(ValueError):
        ofs.Muon([torch.nn.Parameter(torch.zeros(3))])
    train, _, _ = bc.synthetic_corpus(20_000)
    for name, lr in (("adamw", 3e-3), ("lion", 1e-3), ("muon", 0.02)):
        torch.manual_seed(0)
        model = ByteGPT(tiny_cfg())
        opt = ofs.make_optimizer(name, model, lr)
        rng = np.random.default_rng(0)
        losses = []
        for _ in range(40):
            xb, yb = bc.random_windows(train, 8, 16, rng)
            loss = model(torch.from_numpy(xb), torch.from_numpy(yb))[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert np.mean(losses[-5:]) < 0.7 * losses[0], name


def test_parameter_split_covers_every_parameter_once():
    model = ByteGPT(tiny_cfg())
    hidden, io, vectors = ofs.split_parameters(model)
    ids = [id(p) for p in hidden + io + vectors]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert all(p.dim() == 2 for p in hidden) and all(p.dim() == 1 for p in vectors) and len(io) == 3


def test_optimizer_study_quick_run(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(OPT, "train.py")
    records = train.main(["--out_dir", str(tmp_path / "runs"), "--threads", "1"])
    assert len(records) == 4 * 2 + 4 * 2 + 2  # 2 LRs each in QUICK mode, 2 more seeds at the best, 2 long runs
    assert sum(r["settings"]["steps"] == 24 for r in records) == 2
    bench = train.main(["--out_dir", str(tmp_path / "runs"), "--threads", "1", "--bench"])
    assert set(bench) == {"adamw", "lion", "muon"} and (tmp_path / "step_time.json").exists()
    analyze = load_module(OPT, "analyze.py")
    summary = analyze.main(str(tmp_path / "runs"), str(tmp_path / "out"))
    assert set(analyze.optimizers(summary)) == {"adamw", "adamw+clip", "lion", "muon"}
    assert summary["muon"]["seeds"] == 3 and len(summary["_horizon"]) == 2
    assert summary["muon"]["seconds_per_step"] == pytest.approx(bench["muon"]["ms_per_step"] / 1000)
    assert (tmp_path / "out" / "lr_sweep.png").exists() and (tmp_path / "out" / "curves.png").exists()


def test_first_crossing_interpolates():
    analyze = load_module(OPT, "analyze.py")
    assert analyze.first_crossing([0, 100, 200], [3.0, 2.0, 1.0], 1.5) == pytest.approx(150.0)
    assert analyze.first_crossing([0, 100], [3.0, 2.0], 1.0) is None
