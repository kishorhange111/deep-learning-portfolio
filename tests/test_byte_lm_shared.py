"""Tests for the shared byte-level corpus and GPT in dl_common (used by the optimizer, mixture-of-experts
and quantization projects)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dl_common import byte_corpus as bc  # noqa: E402
from dl_common.bytelm import ByteGPT, ByteGPTConfig, evaluate_bpb, lr_schedule  # noqa: E402


def tiny_cfg(**kw):
    return ByteGPTConfig(**{"seq_len": 16, "n_layer": 2, "d_model": 32, "n_head": 4, "d_ff": 64, **kw})


# ---------------------------------------------------------------- shared corpus and model
def test_stdlib_split_is_by_file_and_stable(tmp_path):
    for i in range(40):
        (tmp_path / f"mod{i}.py").write_text(f"x = {i}\n")
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "site-packages" / "third_party.py").write_text("y = 1\n")
    files = bc.stdlib_files(str(tmp_path))
    assert len(files) == 40 and all("site-packages" not in f for f in files)
    train, val, info = bc.load_corpus(str(tmp_path), val_percent=20)
    assert info["train_files"] + info["val_files"] == 40 and 0 < info["val_files"] < 40
    assert info == bc.load_corpus(str(tmp_path), val_percent=20)[2]  # same split and fingerprint every time
    assert len(train) + len(val) == sum(len(f"x = {i}\n") + 1 for i in range(40))


def test_byte_classes():
    assert [bc.byte_class(ord(c)) for c in "aZ7 \n_'(+"] == [
        "lowercase", "uppercase", "digit", "space", "newline", "underscore", "quote", "bracket", "operator",
    ]  # fmt: skip
    assert bc.byte_class(0xC3) == "other" and len(bc.BYTE_CLASS_TABLE) == 256


def test_windows_are_shifted_by_one():
    train, _, _ = bc.synthetic_corpus(5000)
    x, y = bc.random_windows(train, 4, 16, np.random.default_rng(0))
    assert x.shape == (4, 16) and np.array_equal(x[:, 1:], y[:, :-1])
    assert np.array_equal(bc.fixed_windows(train, 3, 8, seed=1)[0], bc.fixed_windows(train, 3, 8, seed=1)[0])


def test_bytelm_is_causal_and_learns():
    torch.manual_seed(0)
    model = ByteGPT(tiny_cfg())
    x = torch.randint(0, 256, (2, 16))
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 256
    with torch.no_grad():
        a, b = model(x)[0], model(x2)[0]
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5) and not torch.allclose(a[:, 10:], b[:, 10:])
    train, val, _ = bc.synthetic_corpus(20_000)
    vx, vy = bc.fixed_windows(val, 16, 16)
    before = evaluate_bpb(model, vx, vy)
    assert before == pytest.approx(8.0, abs=0.3)  # uniform over 256 bytes = 8 bits
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        xb, yb = bc.random_windows(train, 8, 16, rng)
        loss = model(torch.from_numpy(xb), torch.from_numpy(yb))[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert evaluate_bpb(model, vx, vy) < before - 3


def test_lr_schedule_shape():
    lrs = [lr_schedule(s, 100, 10) for s in range(100)]
    assert lrs[9] == pytest.approx(1.0) and lrs[-1] == pytest.approx(0.1, abs=1e-3)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
