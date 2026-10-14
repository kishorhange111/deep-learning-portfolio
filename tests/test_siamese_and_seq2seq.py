"""Tests for the contrastive Siamese network and the seq2seq addition data (Keras on PyTorch, CPU)."""

import random

import numpy as np
import pytest
from conftest import load_train

keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

siamese = load_train("computer-vision/siamese-network-contrastive-loss")
seq2seq = load_train("nlp/seq2seq-lstm-arithmetic")


def to_np(x):
    return ops.convert_to_numpy(x)


def test_make_pairs_balanced_and_correct():
    random.seed(0)
    y = np.repeat(np.arange(10), 5)
    x = y.astype("float32")[:, None, None, None] * np.ones((1, 28, 28, 1), dtype="float32")
    pairs, labels = siamese.make_pairs(x, y)
    assert pairs.shape == (100, 2, 28, 28, 1)
    assert labels.tolist() == [0.0, 1.0] * 50
    digit = pairs[:, :, 0, 0, 0]  # every pixel encodes the digit
    assert np.all((digit[:, 0] == digit[:, 1]) == (labels == 0))


def test_euclidean_distance():
    a = np.array([[0.0, 0.0], [1.0, 1.0]], dtype="float32")
    b = np.array([[3.0, 4.0], [1.0, 1.0]], dtype="float32")
    d = to_np(siamese.euclidean_distance([a, b]))
    assert d.shape == (2, 1)
    assert d[0, 0] == pytest.approx(5.0)
    assert d[1, 0] == pytest.approx(np.sqrt(keras.backend.epsilon()))  # clamped, never exactly 0


def test_contrastive_loss():
    y = ops.convert_to_tensor(np.array([[0.0], [1.0], [1.0]], dtype="float32"))  # Keras passes tensors
    d = ops.convert_to_tensor(np.array([[0.5], [0.25], [2.0]], dtype="float32"))
    # similar: 0.5^2 ; dissimilar inside the margin: (1 - 0.25)^2 ; dissimilar beyond: 0
    expected = (0.25 + 0.5625 + 0.0) / 3
    assert float(to_np(siamese.contrastive_loss(y, d))) == pytest.approx(expected)


def test_pair_accuracy():
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype="float32")
    d = np.array([[0.1], [0.9], [0.2], [0.7]], dtype="float32")
    assert float(to_np(siamese.pair_accuracy(y, d))) == pytest.approx(0.5)


def test_siamese_shares_weights():
    model, embed = siamese.build_model()
    x = np.random.rand(3, 28, 28, 1).astype("float32")
    assert np.allclose(to_np(model([x, x], training=False)), np.sqrt(keras.backend.epsilon()), atol=1e-6)
    assert embed.count_params() == model.count_params()  # one set of weights, used twice


def test_character_table_roundtrip():
    table = seq2seq.CharacterTable(seq2seq.CHARS)
    enc = table.encode("12+3", 7)
    assert enc.shape == (7, 12)
    assert enc[4:].sum() == 0  # rows past the string stay empty
    assert table.decode(table.encode("12+3   ", 7)) == "12+3   "


def test_generate_data(monkeypatch):
    monkeypatch.setattr(seq2seq, "TRAINING_SIZE", 300)
    questions, answers = seq2seq.generate_data(np.random.default_rng(0))
    assert len(questions) == len(answers) == 300
    seen = set()
    for q, a in zip(questions, answers):
        assert len(q) == seq2seq.MAXLEN and len(a) == seq2seq.DIGITS + 1
        left, right = q[::-1].strip().split("+")  # questions are stored reversed
        assert int(a) == int(left) + int(right)
        key = tuple(sorted((int(left), int(right))))
        assert key not in seen  # a+b and b+a never both appear
        seen.add(key)
