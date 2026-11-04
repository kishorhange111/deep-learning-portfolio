"""Tests for systems/quantization-from-scratch: the RTN grid and error bound, GPTQ against RTN, storage
accounting, and an end-to-end quick run."""

import json

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
QUANT = "systems/quantization-from-scratch"


# ---------------------------------------------------------------- quantization
wq = load_module(QUANT, "weight_quant.py")


def test_rtn_grid_and_error_bound():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    for bits in (8, 4, 3):
        q = wq.rtn(w, bits)
        scale = (w.max(1).values - w.min(1).values) / (2**bits - 1)
        assert ((w - q).abs() <= scale[:, None] / 2 + 1e-6).all()  # rounding error at most half a step
        assert all(len(torch.unique(row)) <= 2**bits for row in q)
    grouped = wq.rtn(w, 3, group_size=16)
    assert (w - grouped).norm() < (w - wq.rtn(w, 3)).norm()  # finer groups, smaller error


def test_gptq_equals_rtn_when_inputs_are_uncorrelated():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    h = torch.eye(32) * 2.0
    for group in (None, 8):
        assert torch.allclose(wq.gptq(w, h, 3, group, block_size=16, damp=0.0), wq.rtn(w, 3, group), atol=1e-6)


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    mix = torch.randn(64, 64) * 0.3 + torch.eye(64)
    x = torch.randn(2048, 64) @ mix  # correlated features, like real activations
    h = 2 * x.T @ x / len(x)
    out_err = {}
    for name, q in (("rtn", wq.rtn(w, 3)), ("gptq", wq.gptq(w, h, 3, block_size=32))):
        out_err[name] = float(((x @ w.T - x @ q.T) ** 2).mean())
    assert out_err["gptq"] < 0.7 * out_err["rtn"]


def test_effective_bits():
    assert wq.effective_bits(128, 128, 4) == pytest.approx(4 + 32 / 128)
    assert wq.effective_bits(128, 128, 4, group_size=32) == pytest.approx(5.0)


def test_quantization_quick_study(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(QUANT, "train.py")
    out = train.main(["--out", str(tmp_path / "q.json"), "--threads", "1"])
    eight = next(r for r in out["grid"] if r["bits"] == 8 and r["group"] == "channel")
    assert eight["rtn"]["val_bpb"] == pytest.approx(out["float_val_bpb"], abs=0.01)  # 8 bits is nearly lossless
    assert set(out["sensitivity"]["layers"]) == {"attn.qkv", "attn.out", "mlp.fc", "mlp.proj"}
    analyze = load_module(QUANT, "analyze.py")
    s = analyze.main(str(tmp_path / "q.json"), str(tmp_path / "out"))
    assert len(s["rows"]) == 4 and (tmp_path / "out" / "tradeoff.png").exists()
    json.loads((tmp_path / "out" / "summary.json").read_text())
