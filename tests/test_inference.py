"""Tests for systems/inference-from-scratch: the KV cache is exact, checkpoints from the pretraining
project load, and speculative decoding reproduces the target model's outputs (greedy) and output
distribution (sampling)."""

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "systems/inference-from-scratch"
kv = load_module(PROJECT, "kv_model.py")
dec = load_module(PROJECT, "decoding.py")


def make_model(seed, n_layer=2, d_model=32, n_head=4, vocab=40, std=0.3):
    torch.manual_seed(seed)
    m = kv.GPT(kv.GPTConfig(vocab_size=vocab, seq_len=128, n_layer=n_layer, d_model=d_model, n_head=n_head))
    for p in m.parameters():  # larger weights than a fresh init, so the distributions are not near-uniform
        torch.nn.init.normal_(p, std=std)
    return m.eval()


@torch.no_grad()
def test_incremental_decoding_matches_full_forward():
    model = make_model(0)
    x = torch.randint(0, 40, (2, 20))
    full = model(x)
    cache = model.new_cache(2, 20)
    parts = [model(x[:, :7], cache), model(x[:, 7:8], cache), model(x[:, 8:15], cache)]
    parts += [model(x[:, t : t + 1], cache) for t in range(15, 20)]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5) and cache.length == 20


@torch.no_grad()
def test_cache_rollback_then_continue():
    model = make_model(1)
    x, y = torch.randint(0, 40, (1, 12)), torch.randint(0, 40, (1, 5))
    cache = model.new_cache(1, 32)
    model(torch.cat([x, torch.randint(0, 40, (1, 6))], 1), cache)  # 6 tokens that get "rejected"
    cache.truncate(12)
    assert torch.allclose(model(y, cache), model(torch.cat([x, y], 1))[:, 12:], atol=1e-5)


@torch.no_grad()
def test_loads_pretraining_checkpoint_with_identical_logits(tmp_path):
    gpt = load_module("systems/gpt-pretraining-scaling-laws", "gpt.py")
    torch.manual_seed(0)
    src = gpt.GPT(gpt.GPTConfig(vocab_size=64, seq_len=32, n_layer=2, d_model=32, n_head=4)).eval()
    torch.save({"config": src.cfg.to_dict(), "model": src.state_dict()}, tmp_path / "ckpt.pt")
    dst = kv.GPT.from_checkpoint(tmp_path / "ckpt.pt")
    x = torch.randint(0, 64, (2, 16))
    assert torch.allclose(dst(x), src(x)[0], atol=1e-6)


def test_greedy_kv_matches_naive_for_a_batch():
    model = make_model(2)
    prompt = torch.randint(0, 40, (2, 5))
    a, _ = dec.generate_naive(model, prompt, 25)
    b, stats = dec.generate_kv(model, prompt, 25)
    assert torch.equal(a, b) and stats.new_tokens == 50


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_speculative_is_token_identical_to_target(k):
    target, draft = make_model(3), make_model(4, n_layer=1, d_model=16, n_head=2)
    prompt = torch.randint(0, 40, (1, 6))
    ref, _ = dec.generate_kv(target, prompt, 40)
    out, stats = dec.generate_speculative(target, draft, prompt, 40, k=k)
    assert torch.equal(out, ref)
    assert stats.target_forwards < 40 or stats.acceptance_rate == 0  # never more target calls than plain decoding


def test_self_drafting_accepts_everything():
    model = make_model(5)
    out, stats = dec.generate_speculative(model, model, torch.randint(0, 40, (1, 4)), 30, k=4)
    assert stats.acceptance_rate == 1.0 and stats.tokens_per_target_forward == pytest.approx(5.0)
    assert torch.equal(out, dec.generate_kv(model, out[:, :4], 30)[0])


def test_acceptance_rule_preserves_target_distribution_exactly():
    """Analytically: P(out = x) = q(x) min(1, p(x)/q(x)) + P(reject) * norm(max(0, p - q))(x) = p(x)."""
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        p, q = torch.rand(7, generator=g).softmax(0), torch.rand(7, generator=g).softmax(0)
        accept = q * torch.clamp(p / q, max=1.0)
        residual = torch.clamp(p - q, min=0)
        out = accept + (1 - accept.sum()) * residual / residual.sum()
        assert torch.allclose(out, p, atol=1e-6)


def test_verify_draft_monte_carlo():
    """Run the implementation many times on one position: outputs must follow p, not q."""
    g = torch.Generator().manual_seed(1)
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    q = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.3])
    counts = torch.zeros(5)
    n = 20000
    for _ in range(n):
        d = torch.multinomial(q, 1, generator=g)
        n_acc, nxt = dec.verify_draft(torch.stack([p, p]), q[None], d, g)
        counts[d[0] if n_acc == 1 else nxt] += 1
    assert (counts / n - p).abs().sum() / 2 < 0.015  # total-variation distance


def test_sampled_speculative_first_token_follows_target():
    target, draft = make_model(6, vocab=8, std=0.6), make_model(7, vocab=8, n_layer=1, d_model=16, n_head=2, std=0.6)
    prompt = torch.randint(0, 8, (1, 5))
    with torch.no_grad():
        p = torch.softmax(target(prompt)[0, -1], -1)
    g = torch.Generator().manual_seed(2)
    counts, n = torch.zeros(8), 2500
    for _ in range(n):
        out, _ = dec.generate_speculative(target, draft, prompt, 1, k=3, temperature=1.0, generator=g)
        counts[out[0, -1]] += 1
    assert (counts / n - p).abs().sum() / 2 < 0.04


def test_probs_from_logits_filters():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    assert torch.equal(dec.probs_from_logits(logits, 0.0), torch.tensor([[1.0, 0, 0, 0]]))
    top2 = dec.probs_from_logits(logits, 1.0, top_k=2)
    assert (top2[0, 2:] == 0).all() and top2.sum().item() == pytest.approx(1.0)
    nucleus = dec.probs_from_logits(logits, 1.0, top_p=0.6)
    assert nucleus[0, 0] == pytest.approx(1.0)  # the top token alone already has >= 60 % of the mass


def test_expected_tokens_formula():
    assert dec.expected_tokens_per_round(0.0, 4) == 1.0
    assert dec.expected_tokens_per_round(1.0, 4) == 5
    assert dec.expected_tokens_per_round(0.5, 1) == pytest.approx(1.5)
