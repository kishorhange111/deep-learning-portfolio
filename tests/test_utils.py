"""Tests for src/dl_common/utils.py: seeding, output folder, results files, timers."""

import json
import time

import numpy as np
import pytest

from dl_common import utils


def test_set_seed_is_reproducible():
    utils.set_seed(123)
    a = np.random.rand(3)
    utils.set_seed(123)
    assert np.array_equal(a, np.random.rand(3))


def test_results_dir_respects_env(tmp_path, monkeypatch):
    target = tmp_path / "out"
    monkeypatch.setenv("RESULTS_DIR", str(target))
    assert utils.results_dir("x") == str(target)
    assert target.is_dir()


def test_save_results_rounds_and_writes_files(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    history = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.7]}
    utils.save_results("demo", {"acc": np.float32(0.123456), "name": "vit"}, history, curves=("loss",))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics == {"acc": 0.1235, "name": "vit"}
    assert json.loads((tmp_path / "history.json").read_text()) == history
    assert (tmp_path / "training_curves.png").stat().st_size > 0


def test_save_results_without_history(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    utils.save_results("demo", {"n": 3})
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"n": 3}
    assert not (tmp_path / "history.json").exists()


def test_timer():
    with utils.Timer() as t:
        time.sleep(0.01)
    assert 0 < t.minutes < 1


def test_time_limit_callback_stops_training():
    keras = pytest.importorskip("keras")
    model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
    cb = utils.make_time_limit_callback(minutes=0)
    cb.set_model(model)
    model.stop_training = False
    cb.on_train_begin()
    cb.deadline -= 1  # the budget is already used up
    cb.on_epoch_end(0)
    assert model.stop_training
