"""
Project 06 - Image similarity with a Siamese network and triplet loss (ResNet50)
===============================================================================

TASK: Learn an embedding space where visually similar images are close and
      dissimilar images are far apart - the core of image search, product
      recommendation ("more like this") and face recognition (FaceNet).

HOW (Schroff et al., 2015, "FaceNet"):
    Train on TRIPLETS of images:
        anchor   A - an image
        positive P - an image that "looks like" A
        negative N - an unrelated image
    Triplet loss:  L = max( d(A,P) - d(A,N) + margin, 0 )
    i.e. the positive must be closer to the anchor than the negative is,
    by at least `margin`. Unlike contrastive loss it only cares about the
    RELATIVE ordering of distances, which is exactly what search/ranking needs.

TRANSFER LEARNING: the embedding network is ResNet50 pre-trained on ImageNet.
Early layers (edges, textures) are frozen; only the last residual stage
(conv5) plus a new 3-layer head are fine-tuned. Much less data and compute
is needed than training from scratch.

EVALUATION:
    * triplet accuracy - % of held-out triplets where d(A,P) < d(A,N)
    * mean cosine similarity of anchor-positive vs anchor-negative pairs
    both measured BEFORE and AFTER fine-tuning, to show what training added.

Dataset : "Totally Looks Like" (Rosenfeld et al., 2018) - ~6k pairs of images
          that humans judged to look alike.
Adapted from the Keras example by Hazem Essam & Santiago L. Valdarrama (Apache-2.0):
https://keras.io/examples/vision/siamese_network/
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops
from keras.applications import resnet

from common.utils import Timer, make_time_limit_callback, save_results, set_seed

PROJECT = "06_siamese_triplet_resnet"
QUICK = os.environ.get("QUICK") == "1"

TARGET_SHAPE = (200, 200)
MARGIN = 0.5
BATCH_SIZE = 32
EPOCHS = 1 if QUICK else int(os.environ.get("EPOCHS", "10"))
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "25"))
DATA_DIR = os.path.expanduser("~/data/totally_looks_like")
# Google-Drive ids used by the original Keras example.
DRIVE_IDS = {"left": "1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34", "right": "1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW"}


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, file_id in DRIVE_IDS.items():
        if not os.path.isdir(os.path.join(DATA_DIR, name)):
            zip_path = os.path.join(DATA_DIR, f"{name}.zip")
            subprocess.run(["gdown", file_id, "-O", zip_path], check=True)
            subprocess.run(["unzip", "-oq", zip_path, "-d", DATA_DIR], check=True)


def load_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), TARGET_SHAPE)


def make_triplets():
    """left/xxx.jpg and right/xxx.jpg look alike -> (anchor, positive). Negatives are random images."""
    left = sorted(os.path.join(DATA_DIR, "left", f) for f in os.listdir(os.path.join(DATA_DIR, "left")))
    right = sorted(os.path.join(DATA_DIR, "right", f) for f in os.listdir(os.path.join(DATA_DIR, "right")))
    if QUICK:
        left, right = left[:400], right[:400]
    rng = np.random.RandomState(32)
    negatives = np.array(left + right)
    rng.shuffle(negatives)
    negatives = negatives[:len(left)]
    order = rng.permutation(len(left))
    return np.array(left)[order], np.array(right)[order], negatives[order]


def dataset(a, p, n, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((a, p, n))
    if shuffle:
        ds = ds.shuffle(2048, seed=42)
    ds = ds.map(lambda x, y, z: (load_image(x), load_image(y), load_image(z)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_embedding():
    base = resnet.ResNet50(weights="imagenet", input_shape=TARGET_SHAPE + (3,), include_top=False)
    x = layers.Flatten()(base.output)
    x = layers.BatchNormalization()(layers.Dense(512, activation="relu")(x))
    x = layers.BatchNormalization()(layers.Dense(256, activation="relu")(x))
    out = layers.Dense(256)(x)
    embedding = keras.Model(base.input, out, name="embedding")
    # Freeze everything before the last residual stage; fine-tune conv5 + the head.
    trainable = False
    for layer in base.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable
    return embedding


class SiameseModel(keras.Model):
    """Wraps the embedding network and trains it with a custom triplet-loss train_step."""

    def __init__(self, embedding, margin):
        super().__init__()
        self.embedding = embedding
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, data):
        return self.distances(data)

    def distances(self, data, training=False):
        a, p, n = (self.embedding(resnet.preprocess_input(x * 255.0), training=training) for x in data)
        return ops.sum(ops.square(a - p), -1), ops.sum(ops.square(a - n), -1)

    def _loss(self, data, training):
        ap, an = self.distances(data, training)
        return ops.mean(ops.maximum(ap - an + self.margin, 0.0))

    def train_step(self, data):
        # Standard custom training loop: forward pass under a GradientTape,
        # compute gradients of the loss w.r.t. the trainable weights, apply them.
        with tf.GradientTape() as tape:
            loss = self._loss(data, training=True)
        grads = tape.gradient(loss, self.embedding.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.embedding.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        self.loss_tracker.update_state(self._loss(data, training=False))
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def evaluate(model, ds):
    """Triplet accuracy + mean cosine similarities on a dataset."""
    correct = total = 0
    cos_pos, cos_neg = [], []
    for a, p, n in ds:
        ea, ep, en = (model.embedding(resnet.preprocess_input(x * 255.0), training=False).numpy() for x in (a, p, n))
        dp, dn = ((ea - ep) ** 2).sum(-1), ((ea - en) ** 2).sum(-1)
        correct += int((dp < dn).sum())
        total += len(dp)
        norm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
        cos_pos += list((norm(ea) * norm(ep)).sum(-1))
        cos_neg += list((norm(ea) * norm(en)).sum(-1))
    return {"triplet_accuracy": correct / total,
            "cosine_anchor_positive": float(np.mean(cos_pos)),
            "cosine_anchor_negative": float(np.mean(cos_neg))}


def main():
    set_seed(42)
    download()
    a, p, n = make_triplets()
    split = int(0.8 * len(a))
    train_ds = dataset(a[:split], p[:split], n[:split], shuffle=True)
    val_ds = dataset(a[split:], p[split:], n[split:], shuffle=False)
    print(f"train {split} / val {len(a) - split} triplets")

    model = SiameseModel(build_embedding(), MARGIN)
    model.compile(optimizer=keras.optimizers.Adam(1e-4))

    before = evaluate(model, val_ds)
    print("before fine-tuning:", before)
    with Timer() as t:
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                            callbacks=[make_time_limit_callback(TIME_LIMIT_MIN)], verbose=2)
    after = evaluate(model, val_ds)
    print("after fine-tuning:", after)

    save_results(
        PROJECT,
        {**{f"val_{k}_before": v for k, v in before.items()},
         **{f"val_{k}_after": v for k, v in after.items()},
         "epochs_trained": len(history.history["loss"]),
         "trainable_parameters": int(sum(np.prod(w.shape) for w in model.embedding.trainable_weights)),
         "train_minutes": t.minutes},
        history.history,
        curves=("loss",),
    )


if __name__ == "__main__":
    main()
