"""
Shared helpers used by every project in this repository.

Every training script follows the same pattern so results are comparable:
    1. set_seed()            -> reproducible runs
    2. build data + model
    3. model.fit(..., callbacks=[TimeLimit(...)])   -> hard cap on GPU time
    4. save_results()        -> metrics.json, history.json and a training-curve PNG
"""
import json
import os
import random
import time

import numpy as np

# All results land in <repo>/results/<project_name>/ so they can be committed to GitHub.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))


def set_seed(seed=42):
    """Fix every random number generator we use, so a re-run gives (nearly) the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def results_dir(project):
    """Create (if needed) and return the output folder of one project."""
    path = os.path.join(RESULTS_ROOT, project)
    os.makedirs(path, exist_ok=True)
    return path


def make_time_limit_callback(minutes):
    """Return a Keras callback that stops training after `minutes` of wall-clock time.

    Cloud GPUs are billed by the hour, so every project gets a hard time budget.
    Training stops cleanly at the end of the current epoch once the budget is used.
    """
    import keras

    class TimeLimit(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.deadline = time.time() + minutes * 60

        def on_epoch_end(self, epoch, logs=None):
            if time.time() > self.deadline:
                print(f"[time limit] {minutes} min reached after epoch {epoch + 1}; stopping.")
                self.model.stop_training = True

    return TimeLimit()


def save_results(project, metrics, history=None, curves=("loss",)):
    """Write metrics.json, history.json and training-curve plots for one project.

    metrics : dict of final numbers (e.g. {"test_accuracy": 0.81})
    history : dict of per-epoch lists, e.g. keras History.history
    curves  : which history keys to plot; the matching "val_<key>" is drawn too
    """
    out = results_dir(project)
    metrics = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in metrics.items()}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{project}] metrics: {json.dumps(metrics)}")

    if not history:
        return
    history = {k: [float(x) for x in v] for k, v in history.items()}
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f)

    import matplotlib
    matplotlib.use("Agg")  # no screen on a cloud VM
    import matplotlib.pyplot as plt

    keys = [k for k in curves if k in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(project)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=110)
    plt.close(fig)


def load_cifar(name):
    """CIFAR-10 / CIFAR-100 as numpy arrays, same format as keras.datasets.

    Loaded from the Hugging Face Hub mirror: keras.datasets downloads from the
    original University of Toronto server, which can take 45+ minutes from a cloud VM.
    Returns (x_train, y_train), (x_test, y_test) with x uint8 (N, 32, 32, 3), y int (N, 1).
    """
    from datasets import load_dataset

    ds = load_dataset(f"uoft-cs/{name}")
    label = "fine_label" if name == "cifar100" else "label"

    def to_numpy(split):
        x = np.stack([np.asarray(img.convert("RGB")) for img in ds[split]["img"]]).astype("uint8")
        y = np.asarray(ds[split][label], dtype="int64")[:, None]
        return x, y

    return to_numpy("train"), to_numpy("test")


class Timer:
    """`with Timer() as t: ...` then `t.minutes` - used to report training time."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.minutes = (time.time() - self.start) / 60
