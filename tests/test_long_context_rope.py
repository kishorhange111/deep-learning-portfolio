"""Tests for the RoPE support in dl_common/bytelm.py and systems/long-context-rope: the scaling
methods' defining properties, rotary invariances, and an end-to-end quick run."""

import math

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
from dl_common.bytelm import ByteGPT, ByteGPTConfig, apply_rope, rope_frequencies  # noqa: E402

PROJECT = "systems/long-context-rope"


def test_scaling_methods_have_their_defining_properties():
    d, factor, train_len = 32, 4.0, 128
    base, _ = rope_frequencies(d)
    pi, _ = rope_frequencies(d, method="pi", factor=factor)
    ntk, _ = rope_frequencies(d, method="ntk", factor=factor)
    yarn, yarn_scale = rope_frequencies(d, method="yarn", factor=factor, train_len=train_len)
    assert torch.allclose(pi, base / factor)  # interpolation: every frequency slowed down
    assert ntk[0] == pytest.approx(float(base[0])) and float(ntk[-1]) == pytest.approx(
        float(base[-1]) / factor, rel=1e-4
    )
    turns = train_len * base / (2 * math.pi)
    assert torch.allclose(yarn[turns > 32], base[turns > 32])  # fast frequencies kept
    assert torch.allclose(yarn[turns < 1], base[turns < 1] / factor)  # slow ones interpolated
    assert yarn_scale == pytest.approx((0.1 * math.log(factor) + 1) ** 2)
    for method in ("none", "pi", "ntk", "yarn"):  # factor 1 changes nothing
        inv, scale = rope_frequencies(d, method=method, factor=1.0, train_len=train_len)
        assert torch.allclose(inv, base) and scale == 1.0


def test_rotary_embedding_preserves_norm_and_encodes_relative_position():
    inv, _ = rope_frequencies(16)
    angles = torch.arange(20, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = angles.cos(), angles.sin()
    q, k = torch.randn(16), torch.randn(16)
    rq, rk = apply_rope(q.expand(20, 16), cos, sin), apply_rope(k.expand(20, 16), cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm().expand(20), atol=1e-5)
    dots = rq @ rk.T
    assert dots[7, 3] == pytest.approx(float(dots[15, 11]), abs=1e-4)  # same offset, same score


def test_rope_model_is_causal_and_scaling_is_switchable():
    torch.manual_seed(0)
    model = ByteGPT(ByteGPTConfig(seq_len=16, n_layer=2, d_model=32, n_head=4, d_ff=64, pos="rope"))
    assert not hasattr(model, "pos")
    x = torch.randint(0, 256, (2, 40))
    x2 = x.clone()
    x2[:, 30:] = (x2[:, 30:] + 1) % 256
    with torch.no_grad():
        a, b = model(x)[0], model(x2)[0]
        assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5)
        model.set_rope_scaling("yarn", 2.5)
        assert not torch.allclose(model(x)[0], a)
    with pytest.raises(ValueError):
        model.set_rope_scaling("bogus", 2.0)
    with pytest.raises(ValueError):
        ByteGPT(ByteGPTConfig(seq_len=16, n_layer=1, d_model=32, n_head=4, d_ff=64)).set_rope_scaling("pi", 2.0)


def test_quick_study_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    out = train.main(["--out", str(tmp_path / "study.json")])
    assert set(out["zero_shot"]) == {"none", "pi", "ntk", "yarn"}
    assert set(out["fine_tuned"]) == {"none", "pi", "ntk", "yarn", "more_short_training"}
    assert len(out["zero_shot"]["none"]["64"]["per_position"]) == 64
    analyze = load_module(PROJECT, "analyze.py")
    s = analyze.main(str(tmp_path / "study.json"), str(tmp_path / "out"))
    assert set(s["ft_vs_plain"]["fixed 64"]) == {"pi", "ntk", "yarn"} and "dynamic 32" in s["ft_vs_plain"]
    assert (tmp_path / "out" / "loss_by_position.png").exists()
