"""Tests for research/dpo-vs-rlhf: the DPO and IPO losses, the bigram reward model's function class,
and an end-to-end quick run."""

import itertools
import math

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "research/dpo-vs-rlhf"
da = load_module(PROJECT, "direct_alignment.py")
gh = da.gh


def test_as_pairs_puts_the_preferred_sequence_first():
    a = torch.tensor([[1, 1], [2, 2]])
    b = torch.tensor([[3, 3], [4, 4]])
    chosen, rejected = da.as_pairs((a, b, torch.tensor([1.0, 0.0])))
    assert chosen.tolist() == [[1, 1], [4, 4]] and rejected.tolist() == [[3, 3], [2, 2]]


def test_losses_at_the_reference_and_their_gradient_direction():
    ref = gh.make_reference(4, 3, seed=0)
    policy = gh.MarkovPolicy(4, 3, ref.logits.detach())
    chosen, rejected = ref.sample(64), ref.sample(64)
    loss = da.dpo_loss(policy, ref, chosen, rejected, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)  # h = 0 at the reference
    assert da.ipo_loss(policy, ref, chosen, rejected, tau=0.5).item() == pytest.approx(1.0, abs=1e-6)
    loss.backward()
    with torch.no_grad():
        policy.logits -= 1.0 * policy.logits.grad  # one gradient step
        assert float(da.margin(policy, ref, chosen, rejected).mean()) > 0  # chosen gains on rejected


def test_dpo_training_raises_gold_and_tracks_diagnostics():
    ref = gh.make_reference(6, 4, seed=0)
    gold = gh.random_reward_net(6, 4, seed=3, gain=0.5)
    gold.normalize(ref.sample(20_000, torch.Generator().manual_seed(1)))
    pairs = da.as_pairs(gh.preference_data(ref, gold, 3000, seed=0))
    log = da.train_direct(ref, gold, pairs, "dpo", 0.1, steps=150, lr=0.05, n_evals=3)
    assert log[0]["kl"] == pytest.approx(0.0, abs=1e-6) and log[-1]["kl"] > 0.1
    assert log[-1]["gold"] > log[0]["gold"] + 0.3
    assert {"chosen_logratio", "rejected_logratio"} <= set(log[-1])
    assert da.endpoint(log)["peak_gold"] >= log[-1]["gold"]


def test_bigram_reward_represents_dpo_implicit_reward_exactly():
    """beta * log(pi / ref) of a Markov policy is a sum of per-position bigram terms."""
    ref, pol = gh.make_reference(3, 3, seed=0), gh.make_reference(3, 3, seed=1)
    rm = da.BigramReward(3, 3)
    with torch.no_grad():
        rm.table.copy_(pol.log_tables() - ref.log_tables())
        ys = torch.tensor(list(itertools.product(range(3), repeat=3)))
        assert torch.allclose(rm(ys), pol.log_prob(ys) - ref.log_prob(ys), atol=1e-5)


def test_bigram_proxy_learns_a_bigram_gold():
    ref, target = gh.make_reference(5, 4, seed=0), gh.make_reference(5, 4, seed=2)
    gold = da.BigramReward(5, 4)
    with torch.no_grad():
        gold.table.copy_(target.log_tables() - ref.log_tables())
    gold.normalize(ref.sample(10_000))
    pairs = da.as_pairs(gh.preference_data(ref, gold, 4000, seed=0))
    rm, info = da.train_bigram_proxy(pairs, 5, 4, seed=0)
    a, b = ref.sample(5000), ref.sample(5000)
    assert gh.preference_accuracy(rm, gold, a, b) > 0.85


def test_frontier_is_monotone_upper_envelope():
    pts = [(0.5, 0.1), (1.5, 0.4), (3.0, 0.3), (6.0, 0.9)]
    assert da.frontier(pts, [1, 2, 4, 8]) == [0.1, 0.4, 0.4, 0.9]


def test_quick_study_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    out = train.main(["--out", str(tmp_path / "study.json")])
    assert set(out["dpo"]) == {"0.1"} and len(out["rlhf_bigram"]["0.0"]) == 1
    analyze = load_module(PROJECT, "analyze.py")
    s = analyze.main(str(tmp_path / "study.json"), str(tmp_path / "out"))
    assert set(s["frontier"]) == {"dpo", "ipo", "rlhf_mlp", "rlhf_bigram"}
    for f in ("summary.md", "frontier.png", "dpo_dynamics.png"):
        assert (tmp_path / "out" / f).exists()
