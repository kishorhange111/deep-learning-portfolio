"""
Project 04 - Image captioning: CNN encoder + Transformer decoder (Flickr8k)
=========================================================================

TASK: Given a photo, generate a sentence describing it
      ("a dog is running through the grass").

ARCHITECTURE (the same encoder-decoder idea behind modern vision-language models):
    1. CNN feature extractor - EfficientNetB0 pre-trained on ImageNet (frozen).
       A 299x299 image becomes a 10x10 grid of 1280-d feature vectors
       = 100 "visual tokens".
    2. Transformer encoder - one self-attention block lets the visual tokens
       exchange information (e.g. "dog" region + "grass" region).
    3. Transformer decoder - generates the caption one word at a time:
         * masked (causal) self-attention over the words generated so far,
         * cross-attention: each word looks at the visual tokens to decide
           which part of the image to talk about next,
         * a softmax over the 10,000-word vocabulary.
    Training uses "teacher forcing": input = caption[:-1], target = caption[1:],
    so the model learns to predict every next word in one parallel pass.
    Inference is autoregressive greedy decoding: start with "startseq",
    repeatedly append the most likely next word until "endseq".

ENGINEERING CHOICE: the CNN is frozen, so its output for an image never
changes. We compute the 100 visual tokens for every image ONCE and cache
them, instead of re-running EfficientNet every epoch (~10x faster training).

EVALUATION: BLEU-1..4 on the official 1,000-image test split. BLEU counts how
many 1-, 2-, 3- and 4-word sequences of the generated caption appear in the 5
human reference captions (the standard machine-translation / captioning metric).

Dataset : Flickr8k - 8,091 images, 5 human captions each.
Adapted from the Keras example by A_K_Nain (Apache-2.0):
https://keras.io/examples/vision/image_captioning/
"""
import math
import os
import re
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops

from common.utils import Timer, make_time_limit_callback, results_dir, save_results, set_seed

PROJECT = "04_image_captioning_flickr8k"
QUICK = os.environ.get("QUICK") == "1"

IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25            # max words per caption (incl. start/end markers)
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "30"))
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "25"))
DATA_DIR = os.path.expanduser("~/data/flickr8k")
URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ["Flickr8k_Dataset.zip", "Flickr8k_text.zip"]:
        target = os.path.join(DATA_DIR, name)
        if not os.path.exists(target):
            print(f"downloading {name}")
            urllib.request.urlretrieve(URL + name, target)
            with zipfile.ZipFile(target) as z:
                z.extractall(DATA_DIR)


def clean(text):
    """lower-case, keep only letters and spaces."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def load_captions():
    """Return {image_file: [5 captions]} with 'startseq ... endseq' markers added.

    Captions shorter than 5 words or longer than SEQ_LENGTH are dropped (and
    images left with fewer than 5 captions are skipped), as in the original example.
    """
    captions = {}
    with open(os.path.join(DATA_DIR, "Flickr8k.token.txt")) as f:
        for line in f:
            img, text = line.rstrip("\n").split("\t")
            img = img.split("#")[0]
            words = clean(text).split()
            if 5 <= len(words) <= SEQ_LENGTH - 2:
                captions.setdefault(img, []).append("startseq " + " ".join(words) + " endseq")
    return {k: v[:5] for k, v in captions.items() if len(v) >= 5}


def read_split(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return [l.strip() for l in f if l.strip()]


def extract_features(image_files):
    """Run frozen EfficientNetB0 once over every image -> (N, 100, 1280) float16 array."""
    cnn = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=IMAGE_SIZE + (3,))
    img_dir = os.path.join(DATA_DIR, "Flicker8k_Dataset")

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return tf.image.resize(img, IMAGE_SIZE)  # EfficientNet expects raw 0-255 pixels

    ds = tf.data.Dataset.from_tensor_slices([os.path.join(img_dir, f) for f in image_files])
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(2)
    feats = cnn.predict(ds, verbose=2)                           # (N, 10, 10, 1280)
    return feats.reshape(len(image_files), -1, feats.shape[-1]).astype("float16")


class CaptionBatches(keras.utils.PyDataset):
    """Yields ((image_features, caption_in), caption_out) batches straight from numpy.

    Every (image, caption) pair is one training sample; features are looked up by
    index so the big feature array is never duplicated 5x in memory.
    """

    def __init__(self, feats, img_idx, tokens, shuffle, **kwargs):
        super().__init__(**kwargs)
        self.feats, self.img_idx, self.tokens, self.shuffle = feats, img_idx, tokens, shuffle
        self.order = np.arange(len(img_idx))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.order) / BATCH_SIZE)

    def __getitem__(self, i):
        b = self.order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        tok = self.tokens[b]
        return (self.feats[self.img_idx[b]].astype("float32"), tok[:, :-1]), tok[:, 1:]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class TokenAndPositionEmbedding(layers.Layer):
    """word embedding + learned position embedding (Transformers have no sense of order otherwise)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tok = layers.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos = layers.Embedding(SEQ_LENGTH, EMBED_DIM)

    def call(self, x):
        positions = ops.arange(ops.shape(x)[-1])
        return self.tok(x) * math.sqrt(EMBED_DIM) + self.pos(positions)


def build_model():
    # ----- Encoder: refine the 100 visual tokens with self-attention -----
    img_in = keras.Input((100, 1280), name="image_features")
    x = layers.LayerNormalization()(img_in)
    x = layers.Dense(EMBED_DIM, activation="relu")(x)
    attn = layers.MultiHeadAttention(num_heads=1, key_dim=EMBED_DIM)(x, x)
    enc = layers.LayerNormalization()(x + attn)

    # ----- Decoder: predict the next word -----
    tok_in = keras.Input((SEQ_LENGTH - 1,), dtype="int32", name="caption_in")
    y = TokenAndPositionEmbedding()(tok_in)
    # causal mask: word t may only look at words 0..t (it must not peek at the answer)
    a1 = layers.MultiHeadAttention(num_heads=2, key_dim=EMBED_DIM, dropout=0.1)(y, y, use_causal_mask=True)
    y = layers.LayerNormalization()(y + a1)
    # cross-attention: queries come from the words, keys/values from the image
    a2 = layers.MultiHeadAttention(num_heads=2, key_dim=EMBED_DIM, dropout=0.1)(y, enc)
    y = layers.LayerNormalization()(y + a2)
    ff = layers.Dense(FF_DIM, activation="relu")(y)
    ff = layers.Dropout(0.3)(ff)
    ff = layers.Dense(EMBED_DIM)(ff)
    y = layers.LayerNormalization()(y + ff)
    y = layers.Dropout(0.5)(y)
    logits = layers.Dense(VOCAB_SIZE)(y)
    return keras.Model([img_in, tok_in], logits, name="captioner")


def masked_accuracy(y_true, y_pred):
    """Next-word accuracy, ignoring padding positions (token id 0)."""
    y_true = ops.cast(y_true, "int32")
    match = ops.cast(ops.equal(y_true, ops.cast(ops.argmax(y_pred, axis=-1), "int32")), "float32")
    mask = ops.cast(ops.not_equal(y_true, 0), "float32")
    return ops.sum(match * mask) / ops.maximum(ops.sum(mask), 1.0)


def greedy_decode(model, feats, vectorizer, vocab):
    """Caption many images at once: at each step pick the most likely next word for all of them."""
    start = vectorizer(["startseq"]).numpy()[0, 0]
    tokens = np.zeros((len(feats), SEQ_LENGTH - 1), dtype="int32")
    tokens[:, 0] = start
    for t in range(SEQ_LENGTH - 2):
        logits = model.predict((feats.astype("float32"), tokens), batch_size=256, verbose=0)
        tokens[:, t + 1] = logits[:, t].argmax(-1)
    captions = []
    for row in tokens:
        words = []
        for idx in row[1:]:
            w = vocab[idx]
            if w in ("endseq", ""):
                break
            words.append(w)
        captions.append(" ".join(words))
    return captions


def main():
    set_seed(42)
    download()
    captions = load_captions()
    splits = {s: [f for f in read_split(f"Flickr_8k.{s}Images.txt") if f in captions]
              for s in ["train", "dev", "test"]}
    if QUICK:
        splits = {k: v[:300] for k, v in splits.items()}
    print({k: len(v) for k, v in splits.items()})

    # Vocabulary built from TRAINING captions only (no information leak from test data).
    vectorizer = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=SEQ_LENGTH,
                                          standardize=None)
    vectorizer.adapt([c for f in splits["train"] for c in captions[f]])
    vocab = vectorizer.get_vocabulary()

    feats, data = {}, {}
    for split, files in splits.items():
        print(f"extracting CNN features for {split} ({len(files)} images)")
        feats[split] = extract_features(files)
        img_idx = np.repeat(np.arange(len(files)), 5)                      # 5 captions per image
        tokens = vectorizer([c for f in files for c in captions[f]]).numpy().astype("int32")
        data[split] = (img_idx, tokens)

    train = CaptionBatches(feats["train"], *data["train"], shuffle=True)
    dev = CaptionBatches(feats["dev"], *data["dev"], shuffle=False)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        # ignore_class=0: padding positions do not count in the loss
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
        metrics=[masked_accuracy],
    )
    model.summary()

    with Timer() as t:
        history = model.fit(
            train, validation_data=dev, epochs=EPOCHS,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                       make_time_limit_callback(TIME_LIMIT_MIN)],
            verbose=2,
        )

    # ----- Evaluate with BLEU on the held-out test images -----
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    generated = greedy_decode(model, feats["test"], vectorizer, vocab)
    refs = [[c.split()[1:-1] for c in captions[f]] for f in splits["test"]]
    hyps = [g.split() for g in generated]
    smooth = SmoothingFunction().method1
    bleu = {f"test_bleu_{n}": corpus_bleu(refs, hyps, weights=tuple([1 / n] * n), smoothing_function=smooth)
            for n in range(1, 5)}

    # A few examples for the README.
    with open(os.path.join(results_dir(PROJECT), "sample_captions.md"), "w") as f:
        f.write("| image | generated caption | one human caption |\n|---|---|---|\n")
        for file, g in list(zip(splits["test"], generated))[:15]:
            f.write(f"| {file} | {g} | {' '.join(captions[file][0].split()[1:-1])} |\n")
    save_samples_figure(splits["test"][:6], generated[:6])

    save_results(
        PROJECT,
        {**bleu, "val_masked_accuracy": history.history["val_masked_accuracy"][-1],
         "epochs_trained": len(history.history["loss"]),
         "parameters": model.count_params(), "train_minutes": t.minutes},
        history.history,
        curves=("loss", "masked_accuracy"),
    )


def save_samples_figure(files, generated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, f, g in zip(axes.flat, files, generated):
        ax.imshow(plt.imread(os.path.join(DATA_DIR, "Flicker8k_Dataset", f)))
        ax.set_title(g, fontsize=9, wrap=True)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir(PROJECT), "sample_captions.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
