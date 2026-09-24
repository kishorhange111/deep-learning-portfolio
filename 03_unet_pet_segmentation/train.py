"""
Project 03 - Semantic segmentation of pets with a U-Net (Oxford-IIIT Pets)
=========================================================================

TASK: For *every pixel* of a photo, predict one of 3 classes:
      0 = pet, 1 = background, 2 = pet outline/border.
      (Classification says "there is a cat"; segmentation says "these exact pixels are the cat".)

MODEL: a U-Net-style encoder/decoder.
    * Encoder (down-sampling) - convolutions + pooling shrink the image
      160 -> 80 -> 40 -> 20 -> 10 while increasing channels. It learns WHAT is in the image.
    * Decoder (up-sampling)   - transposed convolutions grow it back to 160x160
      so we get a prediction per pixel. It recovers WHERE things are.
    * Residual (skip) connections carry information around each block, so
      gradients flow easily and fine details are not lost.
    * SeparableConv2D (depthwise + pointwise, as in Xception) gives the same
      receptive field as a normal conv with far fewer parameters (~2M total).
    * Final layer: a softmax over 3 classes *at each pixel*.

METRICS: pixel accuracy and mean Intersection-over-Union (mIoU), the standard
segmentation metric: for each class, |prediction AND truth| / |prediction OR truth|.

Dataset : Oxford-IIIT Pet - 7,390 images with pixel-level "trimap" masks.
Adapted from the Keras example by François Chollet (Apache-2.0):
https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""
import os
import random
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from common.utils import Timer, make_time_limit_callback, results_dir, save_results, set_seed

PROJECT = "03_unet_pet_segmentation"
QUICK = os.environ.get("QUICK") == "1"

IMG_SIZE = (160, 160)
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "40"))
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "20"))
DATA_DIR = os.path.expanduser("~/data/oxford_pets")
MIRRORS = ["https://thor.robots.ox.ac.uk/datasets/pets/", "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"]


def download():
    """Download and extract images + annotations once (~800 MB)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ["images.tar.gz", "annotations.tar.gz"]:
        target = os.path.join(DATA_DIR, name)
        if not os.path.exists(target):
            for base in MIRRORS:
                try:
                    print(f"downloading {base + name}")
                    urllib.request.urlretrieve(base + name, target)
                    break
                except Exception as e:  # try the next mirror
                    print(f"  failed: {e}")
            with tarfile.open(target) as t:
                t.extractall(DATA_DIR)


def list_pairs():
    """Match every image with its mask. Masks are .png 'trimaps' with values 1, 2, 3."""
    img_dir, mask_dir = os.path.join(DATA_DIR, "images"), os.path.join(DATA_DIR, "annotations/trimaps")
    images = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg"))
    masks = sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                   if f.endswith(".png") and not f.startswith("."))
    return images, masks


def load_pair(img_path, mask_path):
    # decode_image (not decode_jpeg): a few files in this dataset are really PNG/GIF.
    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    # "nearest" resizing: never invent in-between label values like 1.5.
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.uint8) - 1          # labels 1,2,3 -> 0,1,2
    return img, mask


def make_dataset(images, masks, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(len(images), seed=42)
    return ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    inputs = keras.Input(IMG_SIZE + (3,))

    # ----- Encoder -----
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous = x  # kept for the residual connection

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous)  # match shape
        x = layers.add([x, residual])
        previous = x

    # ----- Decoder -----
    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        residual = layers.Conv2D(filters, 1, padding="same")(layers.UpSampling2D(2)(previous))
        x = layers.add([x, residual])
        previous = x

    outputs = layers.Conv2D(NUM_CLASSES, 3, activation="softmax", padding="same")(x)  # per-pixel softmax
    return keras.Model(inputs, outputs, name="unet")


def save_examples(model, images, masks, n=4):
    """Save a figure: input photo | true mask | predicted mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        img, mask = load_pair(images[i], masks[i])
        pred = np.argmax(model.predict(img[None], verbose=0)[0], axis=-1)
        for ax, im, title in zip(axes[i], [img, mask[..., 0], pred], ["image", "ground truth", "prediction"]):
            ax.imshow(im, vmin=0, vmax=2) if title != "image" else ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir(PROJECT), "predictions.png"), dpi=90)
    plt.close(fig)


def main():
    set_seed(42)
    download()
    images, masks = list_pairs()
    pairs = list(zip(images, masks))
    random.Random(1337).shuffle(pairs)
    images, masks = [p[0] for p in pairs], [p[1] for p in pairs]
    if QUICK:
        images, masks = images[:600], masks[:600]
    n_val = len(images) // 7   # ~1,000 validation images
    train_ds = make_dataset(images[n_val:], masks[n_val:], shuffle=True)
    val_ds = make_dataset(images[:n_val], masks[:n_val], shuffle=False)
    print(f"train {len(images) - n_val} / val {n_val} images")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",    # classification loss applied at every pixel
        metrics=["accuracy",
                 keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true=True, sparse_y_pred=False, name="mean_iou")],
    )
    model.summary()

    with Timer() as t:
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                make_time_limit_callback(TIME_LIMIT_MIN),
            ],
            verbose=2,
        )

    loss, acc, miou = model.evaluate(val_ds, verbose=0)
    save_examples(model, images[:n_val], masks[:n_val])
    save_results(
        PROJECT,
        {"val_pixel_accuracy": acc, "val_mean_iou": miou, "val_loss": loss,
         "epochs_trained": len(history.history["loss"]),
         "parameters": model.count_params(), "train_minutes": t.minutes},
        history.history,
        curves=("loss", "accuracy", "mean_iou"),
    )


if __name__ == "__main__":
    main()
