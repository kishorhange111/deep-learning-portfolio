"""Tests for the ViT and ShiftViT building blocks (Keras on the PyTorch backend, CPU)."""

import numpy as np
import pytest
from conftest import load_train

keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

vit = load_train("computer-vision/vision-transformer-cifar100")
shiftvit = load_train("computer-vision/shiftvit-attention-free-transformer")


def test_patches_shape_and_content():
    images = np.arange(2 * 12 * 12 * 3, dtype="float32").reshape(2, 12, 12, 3)
    patches = ops.convert_to_numpy(vit.Patches(6)(images))
    assert patches.shape == (2, 4, 6 * 6 * 3)  # (12 / 6)^2 = 4 patches of 108 values
    assert np.allclose(patches[0, 0], images[0, :6, :6, :].reshape(-1))


def test_patch_encoder_adds_positions():
    enc = vit.PatchEncoder(num_patches=4, projection_dim=8)
    out = ops.convert_to_numpy(enc(np.zeros((2, 4, 108), dtype="float32")))
    assert out.shape == (2, 4, 8)
    assert not np.allclose(out[0, 0], out[0, 1])  # zero input -> only position embeddings differ


def test_vit_forward_pass():
    model = vit.build_model(np.random.randint(0, 255, (8, 32, 32, 3)).astype("float32"))
    logits = model(np.zeros((2, 32, 32, 3), dtype="float32"), training=False)
    assert tuple(logits.shape) == (2, vit.NUM_CLASSES)


@pytest.mark.parametrize("dy, dx", [(0, 1), (0, -1), (1, 0), (-1, 0)])
def test_shift_moves_content_and_zero_fills(dy, dx):
    x = np.arange(16, dtype="float32").reshape(1, 4, 4, 1)
    out = ops.convert_to_numpy(shiftvit.shift(x, dy, dx))[0, ..., 0]
    expected = np.zeros((4, 4), dtype="float32")
    src = x[0, ..., 0]
    for i in range(4):
        for j in range(4):
            si, sj = i - dy, j - dx
            if 0 <= si < 4 and 0 <= sj < 4:
                expected[i, j] = src[si, sj]
    assert np.array_equal(out, expected)


def test_drop_path_identity_at_inference_and_per_sample_in_training():
    layer = shiftvit.DropPath(0.5)
    x = np.ones((64, 3, 3, 2), dtype="float32")
    assert np.array_equal(ops.convert_to_numpy(layer(x, training=False)), x)
    y = ops.convert_to_numpy(layer(x, training=True))
    per_sample = y.reshape(64, -1)
    assert set(np.unique(per_sample)) <= {0.0, 2.0}  # kept samples are rescaled by 1 / keep
    assert np.all(per_sample.min(1) == per_sample.max(1))  # whole sample kept or dropped


def test_shiftvit_block_and_patch_merging_shapes():
    x = np.random.rand(2, 8, 8, 24).astype("float32")
    assert tuple(shiftvit.ShiftViTBlock(0.0)(x).shape) == (2, 8, 8, 24)
    assert tuple(shiftvit.PatchMerging()(x).shape) == (2, 4, 4, 48)


def test_warmup_cosine_schedule():
    sched = shiftvit.WarmUpCosine(lr_start=1e-5, lr_max=1e-3, warmup_steps=10, total_steps=110)
    lr = lambda s: float(ops.convert_to_numpy(sched(s)))  # noqa: E731
    assert lr(0) == pytest.approx(1e-5)
    assert lr(5) == pytest.approx(1e-5 + (1e-3 - 1e-5) * 0.5)
    assert lr(10) == pytest.approx(1e-3)
    assert lr(60) == pytest.approx(5e-4, rel=1e-4)  # half-way through the cosine
    assert lr(110) == pytest.approx(0.0, abs=1e-9)
    assert lr(500) == pytest.approx(0.0, abs=1e-9)  # clipped after the end
    assert sched.get_config()["total_steps"] == 110
