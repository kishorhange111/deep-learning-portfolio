"""
Project 01 - Vision Transformer (ViT) from scratch on CIFAR-100
===============================================================

QUESTION: Can a model with *no convolutions at all* classify images?

IDEA (Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words"):
    1. Cut the image into small square patches (here 6x6 pixels).
    2. Flatten each patch and project it to a vector  -> a "word" (token).
    3. Add a learned position embedding so the model knows where each patch was.
    4. Feed the sequence of patch-tokens through standard Transformer encoder
       blocks (self-attention + MLP), exactly like BERT does with words.
    5. Classify from the final representation.

Self-attention lets every patch look at every other patch from the very first
layer, so the model sees global context immediately (a CNN needs many layers).
The catch: Transformers have no built-in notion of "nearby pixels are related"
(no inductive bias), so on small datasets they need strong augmentation and
lots of epochs. That trade-off is exactly what this project demonstrates.

Dataset : CIFAR-100 - 60,000 32x32 colour images, 100 classes.
Adapted from the Keras example by Khalid Salama (Apache-2.0):
https://keras.io/examples/vision/image_classification_with_vision_transformer/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import keras
import numpy as np
from keras import layers, ops

from common.utils import Timer, load_cifar, make_time_limit_callback, save_results, set_seed

PROJECT = "01_vision_transformer_cifar100"
QUICK = os.environ.get("QUICK") == "1"  # QUICK=1 -> tiny run to test the pipeline

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
IMAGE_SIZE = 72                 # images are up-scaled 32 -> 72 so we get more patches
PATCH_SIZE = 6                  # 72 / 6 = 12 patches per side
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 144 tokens per image
PROJECTION_DIM = 64             # size of each token vector
NUM_HEADS = 4                   # attention heads per layer
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]  # MLP inside each block
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]   # classifier on top
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "60"))
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "40"))


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class Patches(layers.Layer):
    """Split a batch of images into flattened, non-overlapping patches.

    (batch, 72, 72, 3) -> (batch, 144, 6*6*3 = 108)
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch = ops.shape(images)[0]
        patches = ops.image.extract_patches(images, size=self.patch_size)
        # extract_patches returns (batch, rows, cols, patch_pixels); merge rows*cols into one sequence axis.
        return ops.reshape(patches, (batch, -1, patches.shape[-1]))


class PatchEncoder(layers.Layer):
    """Project each patch to PROJECTION_DIM and add a learnable position embedding."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        positions = ops.expand_dims(ops.arange(self.num_patches), axis=0)  # [[0, 1, ..., 143]]
        return self.projection(patches) + self.position_embedding(positions)


def mlp(x, hidden_units, dropout_rate):
    """Stack of Dense + GELU + Dropout layers."""
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(x_train):
    # Data augmentation lives *inside* the model: it runs on the GPU and is
    # automatically switched off at evaluation time.
    augmentation = keras.Sequential(
        [
            layers.Normalization(),                 # learns per-channel mean/std from training data
            layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.2, 0.2),
        ],
        name="augmentation",
    )
    augmentation.layers[0].adapt(x_train)

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = augmentation(inputs)
    x = Patches(PATCH_SIZE)(x)
    tokens = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(x)

    # Transformer encoder blocks ("pre-norm" variant: LayerNorm before attention/MLP).
    for _ in range(TRANSFORMER_LAYERS):
        h = layers.LayerNormalization(epsilon=1e-6)(tokens)
        h = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1)(h, h)
        tokens2 = layers.Add()([h, tokens])                    # residual connection 1
        h = layers.LayerNormalization(epsilon=1e-6)(tokens2)
        h = mlp(h, TRANSFORMER_UNITS, dropout_rate=0.1)
        tokens = layers.Add()([h, tokens2])                    # residual connection 2

    # Classification head: flatten all tokens -> MLP -> 100 logits.
    x = layers.LayerNormalization(epsilon=1e-6)(tokens)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = mlp(x, MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(NUM_CLASSES)(x)
    return keras.Model(inputs, logits, name="vit")


def main():
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = load_cifar("cifar100")
    if QUICK:
        x_train, y_train, x_test, y_test = x_train[:2048], y_train[:2048], x_test[:512], y_test[:512]
    print(f"train {x_train.shape}, test {x_test.shape}")

    model = build_model(x_train)
    model.compile(
        # AdamW = Adam with decoupled weight decay; the standard optimizer for Transformers.
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5_accuracy"),
        ],
    )
    model.summary()

    with Timer() as t:
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,                      # 10% of training data to watch over-fitting
            callbacks=[make_time_limit_callback(TIME_LIMIT_MIN)],
            verbose=2,
        )

    _, acc, top5 = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    save_results(
        PROJECT,
        {
            "test_accuracy": acc,
            "test_top5_accuracy": top5,
            "epochs_trained": len(history.history["loss"]),
            "parameters": model.count_params(),
            "train_minutes": t.minutes,
        },
        history.history,
        curves=("loss", "accuracy"),
    )


if __name__ == "__main__":
    main()
