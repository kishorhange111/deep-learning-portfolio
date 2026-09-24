"""
Project 05 - Siamese network with contrastive loss (MNIST)
==========================================================

TASK: Given TWO images, decide whether they show the same digit.
      We never train a "which digit is this?" classifier; the network learns
      a notion of *similarity* instead.

WHY IT MATTERS: this is the idea behind face verification (Face ID), signature
verification, duplicate detection and one-/few-shot learning. A new class
can be recognised from a single reference example without retraining.

HOW:
    * One small CNN (the "embedding network") maps an image to a 10-d vector.
    * The SAME network (shared weights -> "Siamese twins") embeds both images.
    * We compute the Euclidean distance between the two vectors.
    * Contrastive loss (Hadsell, Chopra & LeCun, 2006):
          similar pair    (y=0):  loss = d^2                 -> pull together
          dissimilar pair (y=1):  loss = max(margin - d, 0)^2 -> push apart until
                                                                at least `margin`
    * Prediction: distance < 0.5  =>  "same digit".

Dataset : MNIST, turned into balanced pairs (half same digit, half different).
Adapted from the Keras example by Mehdi (Apache-2.0):
https://keras.io/examples/vision/siamese_contrastive/
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import numpy as np
from keras import layers, ops

from common.utils import Timer, make_time_limit_callback, results_dir, save_results, set_seed

PROJECT = "05_siamese_contrastive_mnist"
QUICK = os.environ.get("QUICK") == "1"

EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = 64
MARGIN = 1.0
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "10"))


def make_pairs(x, y):
    """For every image create one positive pair (same digit, label 0) and one negative pair (label 1)."""
    by_digit = [np.where(y == d)[0] for d in range(10)]
    pairs, labels = [], []
    for i in range(len(x)):
        same = random.choice(by_digit[y[i]])
        pairs.append([x[i], x[same]])
        labels.append(0.0)
        other_digit = random.choice([d for d in range(10) if d != y[i]])
        pairs.append([x[i], x[random.choice(by_digit[other_digit])]])
        labels.append(1.0)
    return np.array(pairs), np.array(labels, dtype="float32")


def embedding_network():
    """Tiny LeNet-style CNN -> 10-d embedding (only ~5k parameters)."""
    inp = layers.Input((28, 28, 1))
    x = layers.BatchNormalization()(inp)
    x = layers.Conv2D(4, 5, activation="tanh")(x)
    x = layers.AveragePooling2D(2)(x)
    x = layers.Conv2D(16, 5, activation="tanh")(x)
    x = layers.AveragePooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="tanh")(x)
    return keras.Model(inp, x, name="embedding")


def euclidean_distance(vectors):
    a, b = vectors
    # max(..., epsilon) avoids sqrt(0), whose gradient is infinite.
    return ops.sqrt(ops.maximum(ops.sum(ops.square(a - b), axis=1, keepdims=True), keras.backend.epsilon()))


def contrastive_loss(y_true, d):
    return ops.mean((1 - y_true) * ops.square(d) + y_true * ops.square(ops.maximum(MARGIN - d, 0)))


def pair_accuracy(y_true, d):
    """Distance > 0.5 means 'different'; compare that decision with the label."""
    return ops.mean(ops.equal(ops.reshape(y_true, [-1]), ops.cast(ops.reshape(d, [-1]) > 0.5, "float32")))


def build_model():
    embed = embedding_network()
    a, b = layers.Input((28, 28, 1)), layers.Input((28, 28, 1))
    dist = layers.Lambda(euclidean_distance, output_shape=(1,))([embed(a), embed(b)])  # same `embed` twice = shared weights
    return keras.Model([a, b], dist, name="siamese"), embed


def main():
    set_seed(42)
    random.seed(42)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = (x_train / 255.0).astype("float32")[..., None], (x_test / 255.0).astype("float32")[..., None]
    x_val, y_val = x_train[50000:], y_train[50000:]
    x_train, y_train = x_train[:50000], y_train[:50000]
    if QUICK:
        x_train, y_train, x_val, y_val = x_train[:2000], y_train[:2000], x_val[:500], y_val[:500]

    p_train, l_train = make_pairs(x_train, y_train)
    p_val, l_val = make_pairs(x_val, y_val)
    p_test, l_test = make_pairs(x_test, y_test)

    model, embed = build_model()
    model.compile(loss=contrastive_loss, optimizer=keras.optimizers.RMSprop(), metrics=[pair_accuracy])
    model.summary()

    with Timer() as t:
        history = model.fit(
            [p_train[:, 0], p_train[:, 1]], l_train,
            validation_data=([p_val[:, 0], p_val[:, 1]], l_val),
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            callbacks=[make_time_limit_callback(TIME_LIMIT_MIN)], verbose=2,
        )
    loss, acc = model.evaluate([p_test[:, 0], p_test[:, 1]], l_test, verbose=0)
    save_embedding_plot(embed, x_test[:3000], y_test[:3000])
    save_results(
        PROJECT,
        {"test_pair_accuracy": acc, "test_loss": loss,
         "epochs_trained": len(history.history["loss"]),
         "parameters": model.count_params(), "train_minutes": t.minutes},
        history.history,
        curves=("loss", "pair_accuracy"),
    )


def save_embedding_plot(embed, x, y):
    """2-D PCA of the learned 10-d embeddings: digits should form separate clusters
    even though the network was never told which digit is which."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = embed.predict(x, verbose=0)
    e = e - e.mean(0)
    _, _, vt = np.linalg.svd(e, full_matrices=False)
    xy = e @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=y, cmap="tab10", s=4)
    ax.legend(*sc.legend_elements(), title="digit", fontsize=7, loc="best")
    ax.set_title("Siamese embeddings of test digits (PCA)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir(PROJECT), "embeddings_pca.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
