"""
Project 09 - Parameter-efficient fine-tuning of GPT-2 with LoRA vs. full fine-tuning
====================================================================================

QUESTION: Do we really need to update all 124M weights of a language model to
          adapt it to a new kind of text?

LoRA (Hu et al., 2021, "Low-Rank Adaptation of Large Language Models"):
    For a frozen weight matrix W (d x k), learn an update  dW = B @ A
    with A: (d x r), B: (r x k) and a tiny rank r (here 4).
        output = W x + (alpha / r) * B A x
    * W never changes -> no gradients / optimizer state for it -> less memory.
    * B starts at zero, so training starts exactly from the pre-trained model.
    * After training, B A can be MERGED into W: zero extra inference latency.
    * Applied to the query and value projections of every attention layer
      (the original paper's choice).
    Trainable parameters: 12 layers x 2 matrices x (768*4 + 4*768) = 147,456
    = 0.12% of GPT-2.

EXPERIMENT: fine-tune GPT-2 (124M) on Reddit "TIFU" stories twice - full
fine-tuning and LoRA - with identical data and settings, and compare:
    trainable parameters, peak GPU memory, training time, validation perplexity.
Each mode runs in its own process so GPU-memory numbers are not mixed up.

Stack: Keras 3 + KerasHub (GPT2CausalLM), mixed precision (float16).
Adapted from the Keras example by Abheesht Sharma & Matthew Watson (Apache-2.0):
https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = "09_gpt2_lora_finetuning"
QUICK = os.environ.get("QUICK") == "1"
MODE = os.environ.get("LORA_MODE")          # set internally: "full" or "lora"

BATCH_SIZE = 32
NUM_BATCHES = 20 if QUICK else 500          # 500 x 32 = 16,000 stories
VAL_BATCHES = 5 if QUICK else 50
MAX_SEQUENCE_LENGTH = 128
RANK, ALPHA = 4, 32.0
PRESET = "gpt2_base_en"
PROMPTS = ["I like basketball", "That Italian restaurant is", "Today I messed up by"]


def load_data():
    """Reddit TIFU: user stories starting with 'Today I F***ed Up by ...'.
    We only need the raw text: the model learns to continue it (next-token prediction)."""
    import tensorflow as tf
    from datasets import load_dataset

    # The Hub's original loader is a Python script (no longer supported by `datasets` v4),
    # so we read the Hub's automatic Parquet conversion of the same data instead.
    ds = load_dataset("ctr4si/reddit_tifu", split="train", revision="refs/convert/parquet")
    column = "documents" if "documents" in ds.column_names else "document"
    texts = [t for t in ds.shuffle(seed=42)[column] if t and len(t) > 50]
    need = (NUM_BATCHES + VAL_BATCHES) * BATCH_SIZE
    texts = texts[:need]
    val, train = texts[:VAL_BATCHES * BATCH_SIZE], texts[VAL_BATCHES * BATCH_SIZE:]
    make = lambda t: tf.data.Dataset.from_tensor_slices(t).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return make(train), make(val)


def run_one(mode):
    """Train GPT-2 in one mode and write <mode>.json with its measurements."""
    import keras
    import keras_hub
    import numpy as np
    import tensorflow as tf

    from common.utils import results_dir, set_seed

    set_seed(42)
    keras.mixed_precision.set_global_policy("mixed_float16")  # fp16 compute, fp32 master weights
    train_ds, val_ds = load_data()

    preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(PRESET, sequence_length=MAX_SEQUENCE_LENGTH)
    model = keras_hub.models.GPT2CausalLM.from_preset(PRESET, preprocessor=preprocessor)

    if mode == "lora":
        # KerasHub's built-in LoRA: freezes the backbone and wraps the query/value
        # projections of every attention layer with trainable low-rank A/B matrices.
        model.backbone.enable_lora(rank=RANK)
        # Keras' enable_lora applies the update B@A without the alpha/r = 8 scale
        # the paper uses. With Adam, multiplying the LoRA output by 8 is roughly
        # equivalent to an 8x larger learning rate for the LoRA weights, so we use
        # lr * alpha / r for the LoRA run.
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    print(f"[{mode}] trainable {trainable:,} / total {total:,}")

    optimizer = keras.optimizers.AdamW(
        learning_rate=(5e-5 * ALPHA / RANK) if mode == "lora" else 5e-5,
        weight_decay=0.01, epsilon=1e-6, global_clipnorm=1.0)
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  weighted_metrics=["accuracy"])

    # Perplexity = exp(average cross-entropy): "how many words is the model choosing between".
    def perplexity():
        loss = model.evaluate(val_ds, verbose=0)[0]
        return float(np.exp(loss))

    ppl_before = perplexity()
    tf.config.experimental.reset_memory_stats("GPU:0")
    start = time.time()
    history = model.fit(train_ds, epochs=1, verbose=2)
    minutes = (time.time() - start) / 60
    peak_gb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
    ppl_after = perplexity()

    samples = {p: model.generate(p, max_length=80) for p in PROMPTS}
    result = {"mode": mode, "trainable_params": trainable, "total_params": total,
              "trainable_percent": 100 * trainable / total,
              "peak_gpu_memory_gb": peak_gb, "train_minutes": minutes,
              "val_perplexity_before": ppl_before, "val_perplexity_after": ppl_after,
              "final_train_loss": history.history["loss"][-1], "samples": samples}
    with open(os.path.join(results_dir(PROJECT), f"{mode}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def compare():
    """Run both modes in separate processes, then build the comparison table + chart."""
    from common.utils import results_dir, save_results

    for mode in ["full", "lora"]:
        subprocess.run([sys.executable, __file__], env=dict(os.environ, LORA_MODE=mode), check=True)
    out = results_dir(PROJECT)
    full, lora = (json.load(open(os.path.join(out, f"{m}.json"))) for m in ["full", "lora"])

    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write("| metric | full fine-tuning | LoRA (r=4) |\n|---|---|---|\n")
        for key in ["trainable_params", "trainable_percent", "peak_gpu_memory_gb", "train_minutes",
                    "val_perplexity_before", "val_perplexity_after"]:
            fmt = (lambda v: f"{v:,}") if key == "trainable_params" else (lambda v: f"{v:.2f}")
            f.write(f"| {key} | {fmt(full[key])} | {fmt(lora[key])} |\n")
        f.write("\n### Sample generations (LoRA model)\n\n")
        for p, s in lora["samples"].items():
            f.write(f"- **{p}** -> {s!r}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [("peak_gpu_memory_gb", "peak GPU memory (GB)"), ("train_minutes", "training time (min)"),
            ("val_perplexity_after", "validation perplexity")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (k, title) in zip(axes, keys):
        ax.bar(["full", "LoRA"], [full[k], lora[k]], color=["#888", "#2a7ab9"])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "full_vs_lora.png"), dpi=110)
    plt.close(fig)

    save_results(PROJECT, {
        "lora_trainable_params": lora["trainable_params"], "full_trainable_params": full["trainable_params"],
        "memory_saving_percent": 100 * (1 - lora["peak_gpu_memory_gb"] / full["peak_gpu_memory_gb"]),
        "speedup_percent": 100 * (1 - lora["train_minutes"] / full["train_minutes"]),
        "full_val_perplexity": full["val_perplexity_after"], "lora_val_perplexity": lora["val_perplexity_after"],
        "base_val_perplexity": full["val_perplexity_before"],
    })


if __name__ == "__main__":
    run_one(MODE) if MODE else compare()
