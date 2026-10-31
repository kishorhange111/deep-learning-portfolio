"""Tests for the knowledge-distillation loss (src/dl_common/distillation.py)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from dl_common.distillation import distillation_loss  # noqa: E402


def test_alpha_one_is_plain_cross_entropy():
    s, t = torch.randn(4, 2), torch.randn(4, 2)
    y = torch.tensor([0, 1, 1, 0])
    assert torch.allclose(distillation_loss(s, t, y, alpha=1.0), F.cross_entropy(s, y))


def test_soft_term_is_zero_when_student_matches_teacher():
    logits = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    loss = distillation_loss(logits, logits.clone(), y, temperature=2.0, alpha=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_soft_term_hand_computed():
    s = torch.tensor([[0.0, 0.0]])
    t = torch.tensor([[2.0, 0.0]])
    T = 2.0
    p = torch.softmax(t / T, -1)
    q = torch.softmax(s / T, -1)
    expected = (p * (p.log() - q.log())).sum() * T**2
    got = distillation_loss(s, t, torch.tensor([0]), temperature=T, alpha=0.0)
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_gradient_flows_to_student_only():
    s = torch.randn(3, 2, requires_grad=True)
    t = torch.randn(3, 2, requires_grad=True)
    distillation_loss(s, t.detach(), torch.tensor([0, 1, 0])).backward()
    assert s.grad is not None and t.grad is None
