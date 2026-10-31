"""Tests for research/grpo-reasoning-study: reward parsing, advantages, the loss's gradient, and a
complete CPU training run on a tiny random model."""

import json
import math
import os

import numpy as np
import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
PROJECT = "research/grpo-reasoning-study"
grpo = load_module(PROJECT, "grpo.py")


# ---------------------------------------------------------------- answers and rewards
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Natalia sold 48/2 = 24 clips in May.\n#### 72", "72"),
        ("#### 1,234", "1234"),
        ("#### -5", "-5"),
        ("#### 3.50", "3.5"),
        ("#### $18.00", "18"),
    ],
)
def test_gsm8k_gold(text, expected):
    assert grpo.gsm8k_gold(text) == expected


def test_extract_answer_prefers_tag_then_last_number():
    assert grpo.extract_answer("<think>2+2=4, maybe 5</think><answer>4</answer>") == "4"
    assert grpo.extract_answer("so the total is 1,000 dollars.") == "1000"
    assert grpo.extract_answer("no digits here") is None
    assert grpo.extract_answer("<answer>$72.0</answer>") == "72"


def test_format_is_strict():
    assert grpo.format_ok("<think>step</think>\n<answer>3</answer>")
    assert not grpo.format_ok("<answer>3</answer>")
    assert not grpo.format_ok("<think>step</think><answer>3</answer> trailing text")
    assert not grpo.format_ok("<think></think><answer>3</answer>")  # empty reasoning


def test_compute_rewards():
    comps = ["<think>a</think><answer>7</answer>", "the answer is 7", "<think>a</think><answer>8</answer>"]
    total, correct, fmt = grpo.compute_rewards(comps, "7", format_weight=0.5)
    assert correct == [1.0, 1.0, 0.0]
    assert fmt == [1.0, 0.0, 1.0]
    assert total == [1.5, 1.0, 0.5]
    assert grpo.compute_rewards(["7"], None, 0.5)[1] == [0.0]  # unparsable gold never counts as correct


# ---------------------------------------------------------------- advantages
def test_group_advantages_normalised_and_zero_for_flat_groups():
    r = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    a = grpo.group_advantages(r)
    assert torch.allclose(a[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(a[0].abs(), torch.ones(4), atol=1e-3)  # (1 - 0.5) / 0.5
    assert torch.equal(a[1], torch.zeros(4))  # all-correct group: no signal
    assert torch.allclose(grpo.group_advantages(r, scale_by_std=False)[0], torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_informative_groups():
    r = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.5, 1.5], [0.5, 1.5]])
    assert grpo.informative_groups(r).tolist() == [True, False, False, True]


# ---------------------------------------------------------------- loss
def test_kl_k3_nonnegative_and_zero_at_equality():
    logp = torch.randn(100)
    assert torch.all(grpo.kl_k3(logp, logp + torch.randn(100)) >= 0)
    assert torch.allclose(grpo.kl_k3(logp, logp), torch.zeros(100))


def test_single_step_gradient_is_policy_gradient():
    """With old_logp = logp.detach() the ratio is 1, so grad(loss) = -sum(A * grad log pi) / n_tokens."""
    torch.manual_seed(0)
    theta = torch.randn(3, 5, requires_grad=True)
    logp = torch.log_softmax(theta, -1)[:, :4]  # pretend 4 completion tokens per sequence
    adv, mask = torch.tensor([1.0, -0.5, 2.0]), torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    loss, _ = grpo.grpo_loss(logp, logp.detach(), adv, mask)
    (g,) = torch.autograd.grad(loss, theta)
    logp2 = torch.log_softmax(theta, -1)[:, :4]
    ref = -(adv[:, None] * logp2 * mask).sum() / mask.sum()
    (g_ref,) = torch.autograd.grad(ref, theta)
    assert torch.allclose(g, g_ref, atol=1e-6)


def test_clipping_blocks_gradient_outside_trust_region():
    logp = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])  # ratio = e > 1 + eps
    loss, stats = grpo.grpo_loss(logp, old, torch.tensor([1.0]), torch.ones(1, 1), eps_high=0.2)
    loss.backward()
    assert logp.grad.abs().item() == 0.0 and stats["clip_frac"] == 1.0
    logp.grad = None
    loss, _ = grpo.grpo_loss(logp, old, torch.tensor([-1.0]), torch.ones(1, 1))  # negative advantage: not clipped
    loss.backward()
    assert logp.grad.abs().item() > 0


def test_aggregation_modes_differ_for_unequal_lengths():
    logp = torch.zeros(2, 3, requires_grad=True)
    adv, mask = torch.tensor([1.0, 1.0]), torch.tensor([[1, 0, 0], [1, 1, 1]])
    tok, _ = grpo.grpo_loss(logp, logp.detach() + 0.1, adv, mask, aggregation="token")
    seq, _ = grpo.grpo_loss(logp, logp.detach() + 0.1, adv, mask, aggregation="sequence")
    assert tok.item() == pytest.approx(seq.item())  # equal per-token values -> same loss...
    (gt,) = torch.autograd.grad(tok, logp, retain_graph=True)
    (gs,) = torch.autograd.grad(seq, logp)
    assert gs[0, 0].abs() > gt[0, 0].abs()  # ...but the short answer's token weighs more per-sequence


def test_pass_at_k():
    assert grpo.pass_at_k(10, 0, 1) == 0.0
    assert grpo.pass_at_k(10, 10, 5) == 1.0
    assert grpo.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert grpo.pass_at_k(4, 1, 2) == pytest.approx(1 - math.comb(3, 2) / math.comb(4, 2))


# ---------------------------------------------------------------- model-level pieces
@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    pytest.importorskip("transformers")
    from common.tiny_hf import save_tiny_chat_model

    return save_tiny_chat_model(str(tmp_path_factory.mktemp("tiny")))


def test_completion_logprobs_match_full_forward(tiny_model_dir):
    from transformers import AutoModelForCausalLM

    train = load_module(PROJECT, "train.py")
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    prompts, comps = [[5, 6, 7], [8, 9]], [[10, 11], [12, 13, 14]]
    ids, attn, cmask, C = train.collate(prompts, comps, pad_id=0)
    got = grpo.completion_logprobs(model, ids, attn, C)
    for i, (p, c) in enumerate(zip(prompts, comps)):  # unpadded reference, one sequence at a time
        seq = torch.tensor([p + c])
        logits = model(input_ids=seq).logits[0].log_softmax(-1)
        expected = torch.stack([logits[len(p) - 1 + j, t] for j, t in enumerate(c)])
        assert torch.allclose(got[i, : len(c)], expected, atol=1e-4)
        assert cmask[i].sum().item() == len(c)


def _write_tiny_gsm8k(path, n):
    os.makedirs(path, exist_ok=True)
    for split in ("train", "test"):
        with open(os.path.join(path, f"{split}.jsonl"), "w") as f:
            for i in range(n):
                row = {"question": f"What is {i} plus {i}?", "answer": f"{i}+{i}={2 * i}\n#### {2 * i}"}
                f.write(json.dumps(row) + "\n")


VARIANT = ["--dynamic_sampling", "--beta", "0", "--ppo_epochs", "2", "--aggregation", "sequence"]


@pytest.mark.parametrize("extra", [[], VARIANT])
def test_end_to_end_cpu_run_and_analysis(tiny_model_dir, tmp_path, monkeypatch, extra):
    analyze, train = load_module(PROJECT, "analyze.py"), load_module(PROJECT, "train.py")
    monkeypatch.setenv("QUICK", "1")
    data = str(tmp_path / "data")
    _write_tiny_gsm8k(data, 8)
    out = str(tmp_path / "runs")
    for name in ("baseline", "variant"):
        argv = ["--model", tiny_model_dir, "--data_dir", data, "--out_dir", out, "--backend", "hf"]
        args = train.parse_args([*argv, "--run_name", name, "--eval_before", "--passk_n", "4", *extra])
        final = train.train(args)
        assert 0.0 <= final["accuracy"] <= 1.0 and final["steps"] == 2
    log = [json.loads(line) for line in open(os.path.join(out, "baseline", "seed0", "train_log.jsonl"))]
    assert len(log) == 2 and {"reward", "zero_std_frac", "samples_total"} <= set(log[0])
    rows = analyze.main(runs_dir=out, out_dir=str(tmp_path / "analysis"))
    assert "vs_baseline" in rows["variant"]
    assert os.path.exists(tmp_path / "analysis" / "summary.md")
    assert os.path.exists(tmp_path / "analysis" / "training_dynamics.png")


def test_paired_bootstrap_detects_a_real_difference():
    analyze = load_module(PROJECT, "analyze.py")
    rng = np.random.default_rng(0)
    b = (rng.random(1000) < 0.5).astype(float)
    a = b.copy()
    flip = rng.choice(np.where(b == 0)[0], 80, replace=False)
    a[flip] = 1.0  # +8 points, all on the same questions
    res = analyze.paired_bootstrap(a, b)
    assert res["diff"] == pytest.approx(0.08) and res["ci_low"] > 0 and res["p"] < 0.01
    assert analyze.paired_bootstrap(b, b)["p"] == 1.0
