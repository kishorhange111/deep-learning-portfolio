"""Tests for the PyTorch generative-model projects: music tokenizer, SMILES helpers, AdaIN, Gram matrices."""
import math

import numpy as np
import pytest
from conftest import load_train

torch = pytest.importorskip("torch")

music = load_train("generative-models/music-transformer")
mol = load_train("generative-models/molecule-generation-transformer")
adain = load_train("generative-models/adain-style-transfer")
nst = load_train("generative-models/neural-style-transfer")


# ---------------------------------------------------------------- music tokenizer
def test_notes_to_tokens_simple_note():
    tokens = music.notes_to_tokens([(0.0, 0.5, 60, 80)])
    assert tokens == [music.VEL + 20, music.NOTE_ON + 60, music.SHIFT + 49, music.NOTE_OFF + 60]


def test_long_gaps_use_several_time_shifts():
    tokens = music.notes_to_tokens([(0.0, 2.5, 60, 64)])
    shifts = [t - music.SHIFT + 1 for t in tokens if music.SHIFT <= t < music.VEL]
    assert shifts == [100, 100, 50]


def test_velocity_token_only_when_bin_changes():
    tokens = music.notes_to_tokens([(0.0, 0.1, 60, 80), (0.2, 0.3, 62, 81), (0.4, 0.5, 64, 120)])
    vel_tokens = [t for t in tokens if t >= music.VEL]
    assert vel_tokens == [music.VEL + 20, music.VEL + 30]


def test_note_off_sorted_before_note_on_at_same_time():
    tokens = music.notes_to_tokens([(0.0, 0.5, 60, 64), (0.5, 1.0, 60, 64)])
    i_off = tokens.index(music.NOTE_OFF + 60)
    assert tokens.index(music.NOTE_ON + 60, i_off) > i_off           # the re-strike comes after the release


def test_tokens_roundtrip():
    notes = [(0.0, 0.5, 60, 80), (0.25, 1.0, 64, 40)]
    decoded = music.tokens_to_notes(music.notes_to_tokens(notes))
    by_pitch = {p: (v, s, e) for v, p, s, e in decoded}
    assert by_pitch[60][1:] == pytest.approx((0.0, 0.5))
    assert by_pitch[64][1:] == pytest.approx((0.25, 1.0))
    assert by_pitch[60][0] == 82 and by_pitch[64][0] == 42           # centre of the velocity bin


def test_tokens_to_notes_handles_malformed_sequences():
    notes = music.tokens_to_notes([music.NOTE_OFF + 50, music.NOTE_ON + 60])
    assert notes == [(64, 60, 0.0, 0.5)]                              # stray NOTE_OFF ignored, held note closed


def test_music_stats_and_js_similarity():
    tokens = music.notes_to_tokens([(0.0, 0.5, 60, 64), (0.5, 1.0, 72, 64)])
    pc, density, vel = music.music_stats(tokens)
    assert pc[0] == pytest.approx(1.0) and pc.sum() == pytest.approx(1.0)   # both notes are a C
    assert density == pytest.approx(2.0)
    assert vel == pytest.approx(66.0)
    p = np.array([0.5, 0.5, 0.0])
    assert music.js_similarity(p, p) == pytest.approx(1.0)
    assert music.js_similarity(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(0.0, abs=1e-6)


def test_music_gpt_forward_and_generate():
    torch.manual_seed(0)
    model = music.GPT().eval()
    x = torch.randint(0, music.VOCAB, (2, 16))
    assert model(x).shape == (2, 16, music.VOCAB)
    out = model.generate(x[:1, :4], n=3)
    assert out.shape == (1, 7) and int(out.max()) < music.VOCAB


def test_music_gpt_is_causal():
    torch.manual_seed(0)
    model = music.GPT().eval()
    x = torch.randint(0, music.VOCAB, (1, 12))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % music.VOCAB                           # change only the last event
    with torch.no_grad():
        assert torch.allclose(model(x)[0, :-1], model(y)[0, :-1], atol=1e-5)


# ---------------------------------------------------------------- SMILES helpers
def test_smiles_vocab_roundtrip():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    stoi, itos, vocab = mol.build_vocab(smiles)
    assert min(stoi.values()) == 3 and vocab == len(stoi) + 3
    ids = mol.encode_smiles("CC(=O)O", stoi)
    assert ids[0] == mol.BOS and ids[-1] == mol.EOS
    assert mol.decode_sample(ids[1:] + [mol.PAD, 5], itos) == "CC(=O)O"   # stops at the first EOS
    assert mol.decode_sample([stoi["C"], stoi["C"]], itos) is None        # never finished


def test_generation_rates():
    rates, unique, novel = mol.generation_rates(10, ["CCO", "CCO", "CCN", "c1ccccc1"], train_set={"CCO"})
    assert rates == pytest.approx({"validity": 0.4, "uniqueness": 0.75, "novelty": 2 / 3})
    assert unique == {"CCO", "CCN", "c1ccccc1"} and sorted(novel) == ["CCN", "c1ccccc1"]
    assert mol.generation_rates(5, [], set())[0] == {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0}


def test_smiles_gpt_forward_shape():
    model = mol.GPT(vocab=40).eval()
    with torch.no_grad():
        assert model(torch.randint(0, 40, (2, 10))).shape == (2, 10, 40)


# ---------------------------------------------------------------- AdaIN
def test_mean_std():
    f = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2)
    m, s = adain.mean_std(f)
    assert m.shape == (1, 2, 1, 1)
    assert m.flatten().tolist() == pytest.approx([1.5, 5.5])
    assert s.flatten().tolist() == pytest.approx([math.sqrt(5 / 3 + 1e-5)] * 2)   # unbiased variance


def test_adain_transfers_style_statistics():
    torch.manual_seed(0)
    content = torch.randn(2, 4, 8, 8) * 3 + 1
    style = torch.randn(2, 4, 8, 8) * 0.5 - 2
    out = adain.adain(content, style)
    om, os_ = adain.mean_std(out)
    sm, ss = adain.mean_std(style)
    assert torch.allclose(om, sm, atol=1e-4)
    assert torch.allclose(os_, ss, atol=1e-3)


def test_decoder_upsamples_relu4_features_to_an_image():
    dec = adain.decoder()
    assert dec(torch.randn(1, 512, 4, 4)).shape == (1, 3, 32, 32)


def test_stylize_alpha_zero_reconstructs_content_features():
    dec = adain.decoder().eval()
    enc = lambda x: x  # noqa: E731  (identity "encoder" on feature maps)
    fc, fs = torch.randn(1, 512, 4, 4), torch.randn(1, 512, 4, 4)
    assert torch.allclose(adain.stylize(enc, dec, fc, fs, alpha=0.0), dec(fc).clamp(0, 1))


# ---------------------------------------------------------------- Gatys style transfer
def test_gram_matrix():
    f = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(1, 2, 1, 2)       # c=2, h=1, w=2
    g = nst.gram(f)
    assert torch.allclose(g, torch.tensor([[5.0, 11.0], [11.0, 25.0]]) / 4)
    assert torch.allclose(g, g.t())


def test_gram_ignores_spatial_arrangement():
    f = torch.randn(1, 3, 4, 4)
    shuffled = f.flatten(2)[..., torch.randperm(16)].view(1, 3, 4, 4)
    assert torch.allclose(nst.gram(f), nst.gram(shuffled), atol=1e-6)


def test_image_tensor_roundtrip():
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 256, (20, 40, 3), dtype=np.uint8))
    x = nst.to_tensor(img, 40)
    assert x.shape == (1, 3, 20, 40) and 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert x.is_contiguous()
    back = nst.to_pil(x)
    assert back.size == (40, 20)
    assert np.abs(np.asarray(back, dtype=int) - np.asarray(img, dtype=int)).max() <= 1
