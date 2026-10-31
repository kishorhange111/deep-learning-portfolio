"""Tests for interpretability/induction-heads: the measurement code on hand-built cases, and a small
training run in which a 2-layer (but not a 1-layer) attention-only model forms induction heads."""

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "interpretability/induction-heads"
ao = load_module(PROJECT, "attn_only.py")


def small_model(n_layer=2, seed=0):
    torch.manual_seed(seed)
    return ao.AttnOnlyTransformer(vocab_size=64, n_layer=n_layer, n_head=4, d_model=32, seq_len=40).eval()


@torch.no_grad()
def test_patterns_are_causal_distributions():
    _, patterns = small_model()(torch.randint(0, 64, (2, 20)), return_patterns=True)
    for p in patterns:
        assert torch.allclose(p.sum(-1), torch.ones(2, 4, 20), atol=1e-5)
        assert torch.all(p.triu(1) == 0)


def test_head_scores_on_ideal_patterns():
    L = 5
    T = 2 * L
    induction = torch.zeros(1, 1, T, T)
    previous = torch.zeros(1, 1, T, T)
    previous[0, 0, 0, 0] = 1.0
    for i in range(1, T):
        previous[0, 0, i, i - 1] = 1.0
    for i in range(T):
        induction[0, 0, i, i - L + 1 if i >= L else 0] = 1.0
    ind, prev = ao.head_scores([induction, previous], L)
    assert ind[0, 0] == pytest.approx(1.0) and prev[1, 0] == pytest.approx(1.0)
    assert ind[1, 0] < 0.1 and prev[0, 0] < 0.6


@torch.no_grad()
def test_ablating_every_head_equals_removing_the_layer():
    model = small_model()
    x = torch.randint(0, 64, (2, 16))
    ablated = model(x, ablate={(1, h) for h in range(4)})
    model.layers[1].W_O.weight.zero_()  # a layer whose output is zero contributes nothing
    assert torch.allclose(ablated, model(x), atol=1e-6)


@torch.no_grad()
def test_head_matrices_reproduce_attention_scores():
    model = small_model()
    attn = model.layers[0]
    x = torch.randn(1, 6, 32)
    qk, _ = attn.head_matrices()
    q = attn.W_Q(x).view(1, 6, 4, 8).transpose(1, 2)
    k = attn.W_K(x).view(1, 6, 4, 8).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / 8**0.5
    assert torch.allclose(scores[0, 2], x[0] @ qk[2] @ x[0].T, atol=1e-5)


def test_composition_of_random_heads_is_near_chance():
    comp = ao.composition_scores(small_model())
    assert comp.shape == (4, 4) and comp.abs().mean() < 0.1


def test_icl_score_handles_short_and_long_contexts():
    assert ao.in_context_learning_score(torch.linspace(5, 1, 255)) < 0
    assert ao.in_context_learning_score(torch.ones(47)) == pytest.approx(0.0)


def test_two_layers_form_induction_heads_and_one_layer_does_not(tmp_path):
    """~25 s on CPU: LZ-style synthetic text, the phase change happens within 1,000 steps (checked on 3 seeds)."""
    train = load_module(PROJECT, "train.py")
    flags = "--synthetic --steps 1000 --batch 32 --seq_len 48 --d_model 48 --n_head 4 --vocab 256 --rep_len 16"
    argv = [*flags.split(), "--eval_every", "500", "--lr", "5e-3", "--out_dir", str(tmp_path)]
    res = train.main(argv)
    one, two = res["final"][1], res["final"][2]
    assert max(map(max, two["induction"])) > 0.4 > 0.15 > max(map(max, one["induction"]))
    assert two["rep_loss"] < one["rep_loss"] - 1.0
    abl = res["ablation"]
    assert abl["all L1 heads with induction score > 0.3"]["rep_loss"] > abl["none"]["rep_loss"] + 1.0
    for f in ("phase_change.png", "head_scores.png", "composition.png", "ablation.md", "metrics.json"):
        assert (tmp_path / f).exists()
