"""Correctness tests for systems/triton-kernels, run on CPU through Triton's interpreter.

conftest.py sets TRITON_INTERPRET=1 before anything imports Triton. The interpreter executes the same
kernel code with NumPy, so masking, tiling and online-softmax bugs show up exactly as they would on a
GPU (performance does not).
"""

import pytest
from conftest import load_module

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
PROJECT = "systems/triton-kernels"
sm = load_module(PROJECT, "softmax.py")
fa = load_module(PROJECT, "flash_attention.py")


@pytest.mark.parametrize("shape", [(3, 7), (5, 128), (2, 3, 1000), (1, 4097)])
def test_fused_softmax_matches_torch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape) * 5
    assert torch.allclose(sm.softmax(x), torch.softmax(x, -1), atol=1e-6)


def test_softmax_is_stable_for_large_logits():
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-1e4, 0.0, 1e4]])
    out = sm.softmax(x)
    assert torch.isfinite(out).all() and torch.allclose(out, torch.softmax(x, -1), atol=1e-6)


@pytest.mark.parametrize("n,tile", [(1000, 128), (300, 64), (64, 64), (5, 1024)])
def test_online_softmax_matches_torch_across_tiles(n, tile):
    torch.manual_seed(1)
    x = torch.randn(4, n) * 3
    x[0, : n // 2] = -float("inf")  # masked prefix: whole tiles of -inf must not produce NaN
    assert torch.allclose(sm.softmax_online(x, tile=tile), torch.softmax(x, -1), atol=1e-6)


def test_softmax_handles_non_contiguous_rows():
    x = torch.randn(64, 32).t()  # last dim is strided
    assert torch.allclose(sm.softmax(x), torch.softmax(x, -1), atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("T,D", [(40, 16), (16, 32), (33, 16)])
def test_flash_attention_forward_matches_reference(causal, T, D):
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 3, T, D) for _ in range(3))
    out = fa.flash_attention(q, k, v, causal=causal)
    assert torch.allclose(out, fa.attention_reference(q, k, v, causal=causal), atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_backward_matches_autograd(causal):
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 2, 37, 16, requires_grad=True) for _ in range(3))
    out = fa.flash_attention(q, k, v, causal=causal)
    ref = fa.attention_reference(q, k, v, causal=causal)
    g = torch.randn_like(out)
    for a, b in zip(torch.autograd.grad(out, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        assert torch.allclose(a, b, atol=1e-5)


def test_uneven_blocks_and_custom_scale():
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 1, 50, 32, requires_grad=True) for _ in range(3))
    out = fa.flash_attention(q, k, v, causal=True, sm_scale=0.3, block_m=32, block_n=16)
    ref = fa.attention_reference(q, k, v, causal=True, sm_scale=0.3)
    assert torch.allclose(out, ref, atol=1e-5)
    g = torch.randn_like(out)
    for a, b in zip(torch.autograd.grad(out, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        assert torch.allclose(a, b, atol=1e-5)


def test_gradcheck_in_float64():
    """Finite-difference check of the analytic backward (double precision; fast_mode projects onto random
    directions instead of building the full Jacobian, which is slow in the interpreter)."""
    torch.manual_seed(5)
    q, k, v = (torch.randn(1, 1, 20, 16, dtype=torch.float64, requires_grad=True) for _ in range(3))
    fn = lambda a, b, c: fa.flash_attention(a, b, c, causal=True)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (q, k, v), atol=1e-6, fast_mode=True)
