"""
Pure text helpers for the image-captioning project
(`computer-vision/image-captioning-cnn-transformer`).

Kept free of TensorFlow / Keras so the caption pre-processing, the decoding of
generated token ids and the BLEU evaluation can be unit-tested on any machine
(`tests/test_captioning.py`).
"""

import re

START, END = "startseq", "endseq"  # sentence markers added around every caption


def clean_caption(text):
    """lower-case, keep only letters and spaces."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def parse_flickr_tokens(lines, seq_length):
    """Parse Flickr8k.token.txt lines into {image_file: [5 captions]}.

    Each line is "<image>.jpg#<n>\\t<caption>". Captions are cleaned and wrapped
    in start/end markers. Captions shorter than 5 words or longer than
    `seq_length - 2` words (room for the two markers) are dropped, and images
    left with fewer than 5 captions are skipped, as in the original example.
    """
    captions = {}
    for line in lines:
        img, text = line.rstrip("\n").split("\t")
        img = img.split("#")[0]  # "xyz.jpg#3" -> "xyz.jpg"
        words = clean_caption(text).split()
        if 5 <= len(words) <= seq_length - 2:
            captions.setdefault(img, []).append(f"{START} " + " ".join(words) + f" {END}")
    return {k: v[:5] for k, v in captions.items() if len(v) >= 5}


def tokens_to_caption(row, vocab):
    """Turn generated token ids into a sentence.

    `row[0]` is the start token and is skipped; decoding stops at the end
    marker or at padding (the empty string, id 0 in Keras' TextVectorization).
    """
    words = []
    for idx in row[1:]:
        w = vocab[idx]
        if w in (END, ""):
            break
        words.append(w)
    return " ".join(words)


def bleu_scores(refs, hyps, prefix="test_bleu_"):
    """Corpus BLEU-1..4 (NLTK, smoothing method 1).

    refs : per image, a list of reference captions, each a list of words
    hyps : per image, the generated caption as a list of words
    BLEU-n uses uniform weights over 1..n-gram precisions; smoothing avoids a
    zero score when some higher-order n-gram never matches.
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smooth = SmoothingFunction().method1
    return {
        f"{prefix}{n}": corpus_bleu(refs, hyps, weights=tuple([1 / n] * n), smoothing_function=smooth)
        for n in range(1, 5)
    }
