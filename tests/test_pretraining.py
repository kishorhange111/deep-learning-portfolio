"""Tests for systems/gpt-pretraining-scaling-laws: model maths, data shards, the training loop (single
process, DDP and FSDP on CPU) and the scaling-law fits on data drawn from a known law."""

import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, load_module

torch = pytest.importorskip("torch")
PROJECT = "systems/gpt-pretraining-scaling-laws"
gpt = load_module(PROJECT, "gpt.py")


def tiny_cfg(**kw):
    return gpt.GPTConfig(**{"vocab_size": 97, "seq_len": 32, "n_layer": 2, "d_model": 32, "n_head": 4, **kw})


# ---------------------------------------------------------------- model
def test_parameter_and_flop_formulas():
    cfg = tiny_cfg()
    model = gpt.GPT(cfg)
    assert model.num_params(non_embedding=True) == gpt.non_embedding_params(cfg)
    assert model.num_params(non_embedding=False) == gpt.non_embedding_params(cfg) + cfg.vocab_size * cfg.d_model
    expected = 6 * (gpt.non_embedding_params(cfg) + 97 * 32) + 12 * 2 * 32 * 32
    assert gpt.flops_per_token(cfg) == expected


@pytest.mark.parametrize("name", list(gpt.PRESETS))
def test_preset_names_match_parameter_counts(name):
    n = gpt.non_embedding_params(gpt.config_for(name))
    assert round(n / 1e6) == int(name[:-1])


def test_initial_loss_is_near_uniform_and_attention_is_causal():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = gpt.GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, torch.randint(0, cfg.vocab_size, (2, 16)))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % cfg.vocab_size  # change the future...
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)  # ...the past logits must not move
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_rope_preserves_norm_and_encodes_relative_position():
    cos, sin = gpt.rope_tables(16, 8, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    qs, ks = q.expand(1, 1, 16, 8), k.expand(1, 1, 16, 8)
    rq, rk = gpt.apply_rope(qs, cos, sin), gpt.apply_rope(ks, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), qs.norm(dim=-1), atol=1e-5)
    dots = (rq[0, 0, :, None] * rk[0, 0, None, :]).sum(-1)  # dots[i, j] = <q at i, k at j>
    assert torch.allclose(dots[3, 1], dots[9, 7], atol=1e-5)  # same offset (2) -> same score


def test_model_and_optimizer_can_overfit_a_batch():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = gpt.GPT(cfg)
    opt = gpt.configure_optimizer(model, lr=3e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 17))
    first = None
    for _ in range(60):
        _, loss = model(x[:, :-1], x[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])


# ---------------------------------------------------------------- data
def test_shards_round_trip_through_token_windows(tmp_path):
    prep, train = load_module(PROJECT, "prepare_data.py"), load_module(PROJECT, "train.py")
    docs = [np.arange(i * 1000, i * 1000 + 300, dtype=np.uint16) for i in range(10)]  # 3,000 tokens
    val = prep.write_split(iter(docs[:2]), str(tmp_path), "val", max_tokens=500, shard_tokens=256)
    trn = prep.write_split(iter(docs[2:]), str(tmp_path), "train", max_tokens=2000, shard_tokens=1024)
    assert [s["tokens"] for s in val] == [256, 244] and sum(s["tokens"] for s in trn) == 2000
    with open(tmp_path / "meta.json", "w") as f:
        json.dump({"val": val, "train": trn}, f)
    data = train.TokenWindows(str(tmp_path), "train", seq_len=16, rank=0, seed=0)
    x, y = data.batch(8)
    assert x.shape == (8, 16) and torch.equal(x[:, 1:], y[:, :-1])
    other = train.TokenWindows(str(tmp_path), "train", seq_len=16, rank=1, seed=0).batch(8)[0]
    assert not torch.equal(x, other)  # ranks draw different windows


def test_lr_schedule():
    train = load_module(PROJECT, "train.py")
    lrs = [train.lr_at(s, 1.0, 10, 100, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=1e-3) and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


# ---------------------------------------------------------------- training loop
def test_single_process_quick_run(tmp_path, monkeypatch):
    train = load_module(PROJECT, "train.py")
    monkeypatch.setenv("QUICK", "1")
    m = train.main(
        ["--synthetic", "--preset", "3M", "--out_dir", str(tmp_path), "--run_name", "r", "--save_checkpoint"]
    )
    assert m["steps"] == 6 and m["final_val_loss"] is not None and m["world_size"] == 1
    ckpt = torch.load(tmp_path / "r" / "ckpt.pt")
    model = gpt.GPT(gpt.GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])  # checkpoint loads back into a fresh model


@pytest.mark.skipif(
    not hasattr(torch, "distributed") or not torch.distributed.is_available(), reason="no torch.distributed"
)
def test_ddp_and_fsdp_give_identical_losses_on_cpu(tmp_path):
    """Same seeds, same data per rank: FSDP's sharded update must match DDP's replicated one."""
    if shutil.which("torchrun") is None and not os.path.exists(
        os.path.join(os.path.dirname(sys.executable), "torchrun")
    ):
        pytest.skip("torchrun not available")
    losses = {}
    for mode in ("ddp", "fsdp"):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(ROOT / PROJECT / "train.py"),
            "--synthetic",
            "--preset",
            "3M",
            "--parallel",
            mode,
            "--out_dir",
            str(tmp_path),
            "--run_name",
            mode,
        ]
        env = {**os.environ, "QUICK": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        subprocess.run(cmd, check=True, env=env, capture_output=True, timeout=600)
        with open(tmp_path / mode / "metrics.json") as f:
            m = json.load(f)
        assert m["world_size"] == 2
        losses[mode] = [json.loads(line)["loss"] for line in open(tmp_path / mode / "train_log.jsonl")]
    assert np.allclose(losses["ddp"], losses["fsdp"], rtol=1e-5)


# ---------------------------------------------------------------- scaling-law fits
def test_grid_brackets_each_budget():
    scaling = load_module(PROJECT, "scaling.py")
    by_budget = {}
    for c, preset, _tokens in scaling.grid():
        by_budget.setdefault(c, []).append(preset)
    assert set(by_budget) == set(scaling.BUDGETS) and all(len(v) >= 3 for v in by_budget.values())


def synthetic_isoflop(scaling, E=1.7, A=400.0, B=1500.0, alpha=0.34, beta=0.28, budgets=(1e17, 1e18, 1e19, 1e20)):
    """IsoFLOP points from a known law with C = 6 N D: N_opt should scale as C^(beta / (alpha + beta))."""
    pts = []
    for c in budgets:
        n_star = (alpha * A / (beta * B)) ** (1 / (alpha + beta)) * (c / 6) ** (beta / (alpha + beta))
        for f in (0.25, 0.5, 1.0, 2.0, 4.0):
            n = n_star * f
            d = c / (6 * n)
            pts.append({"budget": c, "N": n, "D": d, "loss": E + A / n**alpha + B / d**beta, "preset": f"x{f}"})
    return pts


def test_approach2_recovers_known_exponent():
    scaling = load_module(PROJECT, "scaling.py")
    a2 = scaling.approach2(synthetic_isoflop(scaling))
    assert a2["exponent"] == pytest.approx(0.28 / 0.62, abs=0.02)
    assert all(m["inside_grid"] for m in a2["minima"].values())
    lo, hi = a2["exponent_loo_range"]
    assert lo <= a2["exponent"] <= hi


def test_approach3_recovers_known_exponent():
    pytest.importorskip("scipy")
    scaling = load_module(PROJECT, "scaling.py")
    a3 = scaling.approach3(synthetic_isoflop(scaling), n_starts=32)
    assert a3["exponent"] == pytest.approx(0.28 / 0.62, abs=0.03)
    assert a3["E"] == pytest.approx(1.7, abs=0.1)


def test_fit_writes_summary_and_figures(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    scaling = load_module(PROJECT, "scaling.py")
    for p in synthetic_isoflop(scaling):
        d = tmp_path / "sweep" / f"C{p['budget']:.0e}_{p['preset']}"
        d.mkdir(parents=True)
        m = {"non_embedding_params": p["N"], "tokens": p["D"], "final_val_loss": p["loss"], "preset": p["preset"]}
        (d / "metrics.json").write_text(json.dumps(m))
    out = scaling.cmd_fit(None, sweep_dir=str(tmp_path / "sweep"), out_dir=str(tmp_path / "out"))
    assert out["approach2"]["exponent"] == pytest.approx(0.452, abs=0.02)
    for f in ("scaling_summary.md", "isoflop.png", "frontier.png", "scaling_fit.json"):
        assert (tmp_path / "out" / f).exists()
