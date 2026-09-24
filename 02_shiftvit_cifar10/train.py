"""
Project 02 - ShiftViT: a Vision Transformer *without attention* (CIFAR-10)
=========================================================================

QUESTION: Is self-attention really what makes Vision Transformers work?

IDEA (Wang et al., 2022, "When Shift Operation Meets Vision Transformer"):
    Replace the attention layer with a *zero-parameter, zero-FLOP* operation:
    take a small fraction of the feature channels and shift them by one pixel
    (1/12 left, 1/12 right, 1/12 up, 1/12 down, the rest untouched).
    After the shift, each position contains information from its neighbours,
    and the following MLP mixes it. Stack this in a hierarchical design
    (like Swin Transformer: 4 stages, resolution halves and channels double
    between stages) and it performs on par with attention-based models.

    Takeaway for interviews: the *overall architecture* (patchify -> residual
    blocks of token mixing + MLP -> hierarchical stages, LayerNorm, stochastic
    depth, AdamW + warm-up cosine LR) matters at least as much as attention.

Dataset : CIFAR-10 - 60,000 32x32 images, 10 classes.
Adapted from the Keras example by Aritra Roy Gosthipaty & Ritwik Raha (Apache-2.0):
https://keras.io/examples/vision/shiftvit/
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import numpy as np
from keras import layers, ops

from common.utils import Timer, load_cifar, make_time_limit_callback, save_results, set_seed

PROJECT = "02_shiftvit_cifar10"
QUICK = os.environ.get("QUICK") == "1"

# ---------------------------------------------------------------------------
# Hyper-parameters (same as the paper's "tiny" setting, scaled to CIFAR)
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
IMAGE_SIZE = 48                    # 32 -> resize 52 -> random crop 48
PATCH_SIZE = 4                     # 48/4 = 12x12 tokens in stage 1
PROJECTED_DIM = 96                 # channels in stage 1 (192, 384, 768 in later stages)
BLOCKS_PER_STAGE = [2, 4, 8, 2]
STOCHASTIC_DEPTH_RATE = 0.2        # max probability of skipping a whole residual block
MLP_DROPOUT = 0.2
NUM_DIV = 12                       # channels are split in 12 groups; 4 groups get shifted
MLP_EXPAND = 2
LR_START, LR_MAX = 1e-5, 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "60"))
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "40"))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DropPath(layers.Layer):
    """Stochastic depth: randomly skip the residual branch for whole samples during training.

    Acts like dropout at the level of entire blocks; a strong regulariser for deep ViTs.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed_gen = keras.random.SeedGenerator(1337)

    def call(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)   # one coin flip per sample
        mask = ops.floor(keep + keras.random.uniform(shape, seed=self.seed_gen))
        return x / keep * mask                                   # rescale so the expected value is unchanged


def shift(x, dy, dx):
    """Shift a feature map by (dy, dx) pixels, filling the empty border with zeros."""
    h, w = x.shape[1], x.shape[2]
    padded = ops.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    return padded[:, 1 - dy:1 - dy + h, 1 - dx:1 - dx + w, :]


class ShiftViTBlock(layers.Layer):
    """shift some channels -> LayerNorm -> MLP -> residual add (with stochastic depth)."""

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = DropPath(drop_path_rate)

    def build(self, input_shape):
        c = input_shape[-1]
        self.group = c // NUM_DIV
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(c * MLP_EXPAND, activation="gelu"),
            layers.Dropout(MLP_DROPOUT),
            layers.Dense(c),
            layers.Dropout(MLP_DROPOUT),
        ])

    def call(self, x, training=False):
        g = self.group
        # The "token mixing" step: 4 channel groups move one pixel in 4 directions.
        x = ops.concatenate([
            shift(x[..., 0 * g:1 * g], 0, -1),   # left
            shift(x[..., 1 * g:2 * g], 0, 1),    # right
            shift(x[..., 2 * g:3 * g], -1, 0),   # up
            shift(x[..., 3 * g:4 * g], 1, 0),    # down
            x[..., 4 * g:],                      # the other 8/12 of channels stay in place
        ], axis=-1)
        # The "channel mixing" step: a per-pixel MLP with a residual connection.
        return x + self.drop_path(self.mlp(self.norm(x), training=training), training=training)


class PatchMerging(layers.Layer):
    """Between stages: halve height/width and double channels (a strided 2x2 conv)."""

    def build(self, input_shape):
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.reduce = layers.Conv2D(2 * input_shape[-1], 2, strides=2, use_bias=False)

    def call(self, x):
        return self.reduce(self.norm(x))


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up from LR_START to LR_MAX, then cosine decay to zero.

    Warm-up avoids large, destabilising updates while the randomly initialised
    network (and Adam's statistics) are still settling.
    """

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        super().__init__()
        self.lr_start, self.lr_max = lr_start, lr_max
        self.warmup_steps, self.total_steps = warmup_steps, total_steps

    def __call__(self, step):
        step = ops.cast(step, "float32")
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * self.lr_max * (1 + ops.cos(math.pi * ops.clip(progress, 0.0, 1.0)))
        warmup = self.lr_start + (self.lr_max - self.lr_start) * step / max(1, self.warmup_steps)
        return ops.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"lr_start": self.lr_start, "lr_max": self.lr_max,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


def build_model():
    augmentation = keras.Sequential([
        layers.Resizing(IMAGE_SIZE + 4, IMAGE_SIZE + 4),
        layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.Rescaling(1 / 255.0),
    ], name="augmentation")

    inputs = keras.Input((32, 32, 3))
    x = augmentation(inputs)
    # Patchify with a strided conv: every 4x4 patch -> one 96-dim token.
    x = layers.Conv2D(PROJECTED_DIM, PATCH_SIZE, strides=PATCH_SIZE)(x)
    for stage, n_blocks in enumerate(BLOCKS_PER_STAGE):
        # Deeper blocks get a higher chance of being dropped (linearly increasing).
        rates = np.linspace(0, STOCHASTIC_DEPTH_RATE, n_blocks)
        for r in rates:
            x = ShiftViTBlock(float(r))(x)
        if stage < len(BLOCKS_PER_STAGE) - 1:
            x = PatchMerging()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return keras.Model(inputs, outputs, name="shiftvit")


def main():
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = load_cifar("cifar10")
    x_train, y_train, x_val, y_val = x_train[:40000], y_train[:40000], x_train[40000:], y_train[40000:]
    if QUICK:
        x_train, y_train, x_val, y_val = x_train[:2048], y_train[:2048], x_val[:512], y_val[:512]
        x_test, y_test = x_test[:512], y_test[:512]

    total_steps = math.ceil(len(x_train) / BATCH_SIZE) * EPOCHS
    schedule = WarmUpCosine(LR_START, LR_MAX, int(0.15 * total_steps), total_steps)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5_accuracy")],
    )
    model.summary()

    with Timer() as t:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
                make_time_limit_callback(TIME_LIMIT_MIN),
            ],
            verbose=2,
        )

    _, acc, top5 = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    save_results(
        PROJECT,
        {"test_accuracy": acc, "test_top5_accuracy": top5,
         "epochs_trained": len(history.history["loss"]),
         "parameters": model.count_params(), "train_minutes": t.minutes},
        history.history,
        curves=("loss", "accuracy"),
    )


if __name__ == "__main__":
    main()
