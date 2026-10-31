"""Tests for interpretability/crosscoder-model-diffing: the crosscoder recovers planted shared /
A-only / B-only features, and the collect -> train -> analyze pipeline runs on two tiny models."""

import json
import statistics

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "interpretability/crosscoder-model-diffing"
cc = load_module(PROJECT, "crosscoder.py")


def test_shapes_and_batchtopk_budget():
    torch.manual_seed(0)
    model = cc.Crosscoder(d_model=16, n_latents=64, sparsity="batchtopk", k=5)
    x = torch.randn(32, 2, 16)
    x_hat, f = model(x)
    assert x_hat.shape == x.shape and f.shape == (32, 64)
    assert (f > 0).sum().item() == 5 * 32  # k latents per token on average, chosen across the batch
    model.eval()
    assert model(x)[1].shape == (32, 64)  # inference uses the learned threshold instead
    assert model.decoder_norms().shape == (64, 2)


def test_l1_loss_weights_latents_by_decoder_norm():
    torch.manual_seed(0)
    model = cc.Crosscoder(d_model=8, n_latents=16, sparsity="l1")
    x = torch.randn(4, 2, 8)
    total, f, _ = model.loss(x, l1_coeff=1.0)
    mse, _, _ = model.loss(x, l1_coeff=0.0)
    expected_l1 = (f * model.decoder_norms().sum(-1)).sum(-1).mean()
    assert torch.allclose(total - mse, expected_l1, atol=1e-5)


def test_relative_norms_and_classes():
    model = cc.Crosscoder(d_model=4, n_latents=3, sparsity="l1")
    with torch.no_grad():
        model.W_dec.zero_()
        model.W_dec[0, 0, 0] = 1.0  # A-only
        model.W_dec[1, 1, 0] = 1.0  # B-only
        model.W_dec[2, :, 0] = 1.0  # shared
    r = cc.relative_norms(model)
    assert r.tolist() == pytest.approx([0.0, 1.0, 0.5])
    classes = cc.classify(r)
    assert classes["A-only"].tolist() == [True, False, False] and classes["shared"].tolist() == [False, False, True]


@pytest.mark.parametrize("sparsity,b_min,a_max", [("batchtopk", 0.7, 0.3), ("l1", 0.9, 0.1)])
def test_planted_features_are_attributed_to_the_right_model(tmp_path, sparsity, b_min, a_max):
    train = load_module(PROJECT, "train.py")
    m = train.main(["--planted", "--sparsity", sparsity, "--out_dir", str(tmp_path)])
    rec = m["planted_recovery"]
    assert statistics.median(rec["only_b"]["cosine"]) > 0.9 and statistics.median(rec["only_a"]["cosine"]) > 0.9
    assert statistics.median(rec["only_b"]["relative_norm"]) > b_min
    assert statistics.median(rec["only_a"]["relative_norm"]) < a_max
    assert 0.35 < statistics.median(rec["shared"]["relative_norm"]) < 0.7
    assert m["fve_model_a"] > 0.9 and m["fve_model_b"] > 0.9
    analyze = load_module(PROJECT, "analyze.py")
    summary = analyze.main(["--run_dir", str(tmp_path / "planted"), "--samples", "5000"])
    assert summary["alive"] > 0 and (tmp_path / "planted" / "relative_norms.png").exists()


def test_collect_train_analyze_on_tiny_models(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from common.tiny_hf import save_tiny_chat_model

    a = save_tiny_chat_model(str(tmp_path / "a"), seed=0)
    b = save_tiny_chat_model(str(tmp_path / "b"), seed=1)
    texts = tmp_path / "texts.jsonl"
    sentences = ["the quick brown fox jumps over the lazy dog", "How many clips did Natalia sell in May?"]
    texts.write_text("\n".join(json.dumps({"text": s * 3}) for s in sentences))
    collect = load_module(PROJECT, "collect.py")
    data = tmp_path / "acts"
    meta = collect.main(
        [
            "--model_a",
            a,
            "--model_b",
            b,
            "--tokens",
            "200",
            "--seq_len",
            "32",
            "--positions_per_seq",
            "8",
            "--batch",
            "4",
            "--texts_file",
            str(texts),
            "--out_dir",
            str(data),
        ]
    )
    assert meta["n"] == 200 and meta["d_model"] == 32 and meta["layer"] == 1 and len(meta["scale"]) == 2

    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    m = train.main(
        ["--data_dir", str(data), "--n_latents", "64", "--k", "8", "--batch", "64", "--out_dir", str(tmp_path / "runs")]
    )
    assert m["alive_latents"] > 0
    analyze = load_module(PROJECT, "analyze.py")
    run_dir = tmp_path / "runs" / "acts_batchtopk"
    analyze.main(["--run_dir", str(run_dir), "--samples", "200", "--top_latents", "3"])
    assert "«" in (run_dir / "features.md").read_text()  # contexts were decoded with the activating token marked


def test_weight_diff_localises_a_planted_change():
    pytest.importorskip("transformers")
    from common.tiny_hf import make_tiny_qwen2

    base = load_module(PROJECT, "baseline_diff.py")
    ma = make_tiny_qwen2(vocab_size=50, seed=0)
    mb = make_tiny_qwen2(vocab_size=50, seed=0)
    with torch.no_grad():
        mb.model.layers[1].self_attn.q_proj.weight.mul_(1.1)
    wd = base.weight_diff(ma, mb)
    assert wd["attn q"][1] == pytest.approx(0.1, abs=1e-5)
    assert all(v == 0 for g, vals in wd.items() for i, v in enumerate(vals) if (g, i) != ("attn q", 1))
