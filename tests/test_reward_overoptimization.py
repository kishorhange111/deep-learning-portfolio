"""Tests for research/reward-model-overoptimization: exact KL, the best-of-n estimator, preference
training, ensembles, RL, and the functional-form fits, each against an answer known in advance."""

import itertools
import math

import numpy as np
import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "research/reward-model-overoptimization"
gh = load_module(PROJECT, "goodhart.py")


def all_sequences(length, vocab):
    """Every sequence, for brute-force checks on tiny spaces."""
    return torch.tensor(list(itertools.product(range(vocab), repeat=length)))


def test_policy_probabilities_sum_to_one_and_match_sampling():
    ref = gh.make_reference(3, 4, seed=0)
    with torch.no_grad():
        p = ref.log_prob(all_sequences(3, 4)).exp()
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)
    samples = ref.sample(40_000, torch.Generator().manual_seed(0))
    first = torch.bincount(samples[:, 0], minlength=4).float() / 40_000
    assert torch.allclose(first, ref.log_tables().exp()[0, 0], atol=0.01)
    assert torch.allclose(ref.marginals()[2], torch.bincount(samples[:, 2], minlength=4).float() / 40_000, atol=0.01)


def test_exact_kl_matches_brute_force():
    ref = gh.make_reference(3, 3, seed=0)
    pol = gh.make_reference(3, 3, seed=1)
    ys = all_sequences(3, 3)
    with torch.no_grad():
        lp, lr = pol.log_prob(ys), ref.log_prob(ys)
        brute = float((lp.exp() * (lp - lr)).sum())
        assert float(gh.exact_kl(pol, ref)) == pytest.approx(brute, rel=1e-4)
        assert float(gh.exact_kl(ref, ref)) == pytest.approx(0.0, abs=1e-6)


def test_bon_kl_formula():
    assert gh.bon_kl(1) == 0.0 and gh.bon_kl(2) == pytest.approx(math.log(2) - 0.5)


def test_best_of_n_estimator_matches_simulation():
    rng = np.random.default_rng(0)
    proxy = rng.normal(size=500)
    gold = proxy + rng.normal(size=500)
    rows = gh.best_of_n_curve(proxy, gold, [1, 4, 16])
    assert rows[0]["gold"] == pytest.approx(gold.mean())
    for row in rows[1:]:
        picks = [max(rng.choice(500, row["n"], replace=False), key=lambda i: proxy[i]) for _ in range(20_000)]
        assert row["gold"] == pytest.approx(gold[picks].mean(), abs=0.03)
        assert row["proxy"] == pytest.approx(proxy[picks].mean(), abs=0.03)


def test_gold_reward_is_normalized_and_has_diminishing_returns():
    ref = gh.make_reference(6, 4, seed=0)
    gold = gh.GoldReward(6, 4, seed=1, peak=2.0)
    samples = ref.sample(20_000, torch.Generator().manual_seed(0))
    gold.normalize(samples)
    with torch.no_grad():
        r = gold(samples)
        s, d = gold.parts(samples)
    assert float(r.mean()) == pytest.approx(0.0, abs=1e-4) and float(r.std()) == pytest.approx(1.0, abs=1e-4)
    assert float(d.std()) == pytest.approx(1.0, abs=1e-4)
    gold.parts = lambda y: (torch.zeros(3), torch.tensor([1.5, 2.0, 2.5]))  # same s, d around the peak
    with torch.no_grad():
        at = gold(torch.zeros(3, 6, dtype=torch.long))
    assert at[1] > at[0] and at[1] > at[2]  # more d helps up to `peak` and hurts after it


def test_proxy_learns_a_linear_gold_from_preferences():
    length, vocab = 6, 4
    ref = gh.make_reference(length, vocab, seed=0)
    gold = gh.random_reward_net(length, vocab, seed=3, gain=0.3)  # nearly linear: easy to learn
    gold.normalize(ref.sample(20_000, torch.Generator().manual_seed(1)))
    proxy, info = gh.train_proxy(
        gh.preference_data(ref, gold, 4000, seed=0), length, vocab, seed=0, width=16, epochs=60
    )
    a, b = ref.sample(5000), ref.sample(5000)
    assert gh.preference_accuracy(proxy, gold, a, b) > 0.85
    assert info["n_train_pairs"] == 3600


def test_ensemble_modes():
    class Const:
        def __init__(self, v):
            self.v = v

        def __call__(self, y):
            return torch.full((len(y),), self.v)

    y = torch.zeros(3, 2, dtype=torch.long)
    members = [Const(1.0), Const(3.0)]
    assert torch.allclose(gh.Ensemble(members, "mean")(y), torch.full((3,), 2.0))
    assert torch.allclose(gh.Ensemble(members, "min")(y), torch.full((3,), 1.0))
    assert torch.allclose(gh.Ensemble(members, "uwo", lam=1.0)(y), torch.full((3,), 2.0 - math.sqrt(2)))


def test_rl_increases_reward_and_kl_penalty_limits_kl():
    length, vocab = 5, 4
    ref = gh.make_reference(length, vocab, seed=0)
    gold = gh.random_reward_net(length, vocab, seed=3)
    gold.normalize(ref.sample(10_000, torch.Generator().manual_seed(1)))
    free = gh.rl_optimize(ref, gold, gold, steps=150, batch=128, lr=0.05, eval_every=50, eval_n=2000)
    tied = gh.rl_optimize(ref, gold, gold, steps=150, batch=128, lr=0.05, kl_coef=1.0, eval_every=50, eval_n=2000)
    assert free[0]["kl"] == pytest.approx(0.0, abs=1e-6)
    assert free[-1]["gold"] > free[0]["gold"] + 1.0
    assert tied[-1]["kl"] < free[-1]["kl"]


@pytest.mark.parametrize("form", ["bon", "rl"])
def test_functional_form_fit_recovers_coefficients(form):
    d = np.linspace(0.1, 5, 40)
    gold = d * (1.3 - 0.2 * (d if form == "bon" else np.log(d)))
    alpha, beta, r2 = gh.fit_functional_form(d, gold, form)
    assert alpha == pytest.approx(1.3, abs=1e-8) and beta == pytest.approx(0.2, abs=1e-8) and r2 == pytest.approx(1.0)


def test_quick_study_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("QUICK", "1")
    train = load_module(PROJECT, "train.py")
    out = train.main(["--out", str(tmp_path / "study.json")])
    assert set(out["data_scaling"]) == {"300", "1000"} and len(out["ensembles"]) == 5
    assert out["data_scaling"]["1000"][0]["bon"][0]["kl"] == 0.0
    analyze = load_module(PROJECT, "analyze.py")
    summary = analyze.main(str(tmp_path / "study.json"), str(tmp_path / "out"))
    assert "rl_fit" in summary["data_scaling"]["300"]
    for f in ("summary.md", "overoptimization.png", "mechanism.png", "kl_penalty.png", "ensembles.png"):
        assert (tmp_path / "out" / f).exists()
