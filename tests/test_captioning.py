"""Tests for caption cleaning, parsing, decoding and BLEU (common/captioning.py)."""
import pytest

from common.captioning import bleu_scores, clean_caption, parse_flickr_tokens, tokens_to_caption


def test_clean_caption():
    assert clean_caption("A dog, running!  ") == "a dog running"
    assert clean_caption("Two-year-old 2 kids") == "twoyearold  kids"


def test_parse_flickr_tokens_filters_and_wraps():
    good = "a dog runs on the grass"
    lines = [f"img1.jpg#{i}\t{good}\n" for i in range(5)]
    lines += ["img1.jpg#5\tsixth caption is also fine here\n"]      # only the first 5 are kept
    lines += [f"img2.jpg#{i}\tshort one\n" for i in range(5)]       # too short -> image dropped
    caps = parse_flickr_tokens(lines, seq_length=25)
    assert list(caps) == ["img1.jpg"]
    assert len(caps["img1.jpg"]) == 5
    assert caps["img1.jpg"][0] == f"startseq {good} endseq"


def test_parse_flickr_tokens_respects_max_length():
    long = " ".join(["word"] * 24)
    caps = parse_flickr_tokens([f"a.jpg#{i}\t{long}\n" for i in range(5)], seq_length=25)
    assert caps == {}


def test_tokens_to_caption_stops_at_end_or_padding():
    vocab = ["", "[UNK]", "startseq", "endseq", "a", "dog"]
    assert tokens_to_caption([2, 4, 5, 3, 4], vocab) == "a dog"
    assert tokens_to_caption([2, 4, 0, 5], vocab) == "a"


def test_bleu_scores():
    pytest.importorskip("nltk")
    refs = [[["a", "dog", "runs", "on", "grass"]]]
    perfect = bleu_scores(refs, [["a", "dog", "runs", "on", "grass"]])
    assert set(perfect) == {f"test_bleu_{n}" for n in range(1, 5)}
    assert all(v == pytest.approx(1.0) for v in perfect.values())
    partial = bleu_scores(refs, [["a", "cat", "sits", "on", "grass"]])
    assert partial["test_bleu_1"] == pytest.approx(0.6)
    assert partial["test_bleu_4"] < partial["test_bleu_1"]
