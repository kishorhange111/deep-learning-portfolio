"""Tests for systems/mixture-of-experts: routing, gating, the load-balancing loss, capacity, and an
end-to-end quick run."""

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
MOE = "systems/mixture-of-experts"


# ---------------------------------------------------------------- mixture of experts
ml = load_module(MOE, "moe_layer.py")


def test_moe_with_identical_experts_equals_one_mlp():
    """Routing weights are renormalized, so if every expert is the same MLP the layer is that MLP."""
    torch.manual_seed(0)
    layer = ml.MoELayer(16, 32, n_experts=4, top_k=2)
    for e in layer.experts[1:]:
        e.load_state_dict(layer.experts[0].state_dict())
    x = torch.randn(3, 5, 16)
    assert torch.allclose(layer(x), layer.experts[0](x), atol=1e-5)


def test_moe_routes_each_token_to_its_top_expert():
    torch.manual_seed(0)
    layer = ml.MoELayer(8, 16, n_experts=4, top_k=1)
    x = torch.randn(10, 8)
    out = layer(x)
    top_i, gates, probs, _ = layer.route(x)
    for t in range(10):
        e = int(top_i[t, 0])
        assert torch.allclose(out[t], layer.experts[e](x[t : t + 1])[0] * probs[t, e], atol=1e-6)


def test_load_balancing_loss_is_one_when_uniform_and_larger_when_collapsed():
    torch.manual_seed(0)
    layer = ml.MoELayer(8, 16, n_experts=4, top_k=1, balance_coef=1.0, z_coef=0.0)
    with torch.no_grad():
        layer.router.weight.zero_()
    layer(torch.randn(64, 8))  # all logits equal: P_e = 1/4, and ties all go to expert 0
    assert layer.last_stats["balance_loss"] == pytest.approx(1.0)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[2, 0] = 50.0
    x = torch.randn(64, 8).abs()
    layer(x)
    assert layer.last_stats["balance_loss"] > 3.5  # everything on expert 2: f = P = one-hot -> E * 1 = 4


def test_capacity_drops_overflow_tokens():
    torch.manual_seed(0)
    layer = ml.MoELayer(8, 16, n_experts=4, top_k=1, capacity_factor=1.0)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[1, 0] = 50.0
    x = torch.randn(40, 8).abs()  # all prefer expert 1; capacity = ceil(40 / 4) = 10
    out = layer(x)
    assert layer.last_stats["dropped_frac"] == pytest.approx(0.75)
    assert int((out.abs().sum(1) > 0).sum()) == 10


def test_moe_quick_study(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(MOE, "train.py")
    results = train.main(["--out_dir", str(tmp_path / "runs"), "--threads", "1"])
    by = {r["config"]: r for r in results}
    assert by["moe8_top1"]["mlp_params_total_per_layer"] > 7 * by["dense"]["mlp_params_total_per_layer"]
    assert by["moe8_top1"]["routing"][0]["load"] and by["dense"]["routing"] is None
    assert sum(r["settings"]["steps"] == 20 for r in results) == 2  # the horizon-check runs
    bench = train.main(["--out_dir", str(tmp_path / "runs"), "--threads", "1", "--bench"])
    assert {"dense", "moe8_top1", "moe8_top2", "dense_wide", "moe8_top1_unitgate"} <= set(bench)
    analyze = load_module(MOE, "analyze.py")
    rows = analyze.main(str(tmp_path / "runs"), str(tmp_path / "out"))
    assert rows["moe8_top1"]["ms_per_step_source"] == "benchmark"
    assert "load_entropy_bits" in rows["moe8_top1"] and (tmp_path / "out" / "specialization.png").exists()
    assert set(rows["_horizon"]) == {"dense", "moe8_top1_unitgate"} and (tmp_path / "out" / "horizon.png").exists()


def test_unit_gate_keeps_full_expert_output_and_router_gradient():
    torch.manual_seed(0)
    layer = ml.MoELayer(8, 16, n_experts=4, top_k=1, unit_gate=True, balance_coef=0.0, z_coef=0.0)
    x = torch.randn(10, 8)
    out = layer(x)
    top_i = layer.route(x)[0]
    for t in range(10):
        assert torch.allclose(out[t], layer.experts[int(top_i[t, 0])](x[t : t + 1])[0], atol=1e-6)
    out.sum().backward()
    assert layer.router.weight.grad is not None and layer.router.weight.grad.abs().sum() > 0
