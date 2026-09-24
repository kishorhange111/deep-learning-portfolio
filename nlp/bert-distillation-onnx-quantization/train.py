"""
Project 10 - Making BERT fast: knowledge distillation + ONNX Runtime + int8 quantization
========================================================================================

QUESTION: A fine-tuned BERT is accurate but slow and large (110M parameters,
~420 MB). How much faster/smaller can we make it for CPU inference, and what
does each step cost in accuracy?

THE OPTIMIZATION LADDER (each step is measured, not assumed):
    1. Teacher   - bert-base-uncased fine-tuned on IMDB (12 layers).
    2. Student   - distilbert-base-uncased (6 layers, 66M params, ~40% smaller).
       a) trained normally on the labels             (baseline student)
       b) trained with KNOWLEDGE DISTILLATION         (Hinton et al., 2015)
          loss = a * CE(student, label)
               + (1 - a) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )
          The teacher's *soft* probabilities ("this review is 70% positive")
          carry more information than hard 0/1 labels, so the small model
          learns more from the same data.
    3. ONNX export - a framework-independent graph that ONNX Runtime executes
       with fused operators (attention, LayerNorm, GELU) -> faster on CPU.
    4. Dynamic int8 quantization - weights stored as 8-bit integers instead
       of 32-bit floats: ~4x smaller, faster matrix multiplies, activations
       quantized on the fly.

BENCHMARK: CPU, batch size 1, sequence length 128 (a typical online request),
median latency over 200 runs after warm-up; accuracy on the IMDB test set.

Stack: PyTorch, Hugging Face transformers, ONNX, ONNX Runtime.
"""
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.utils import Timer, results_dir, save_results, set_seed

PROJECT = "10_bert_distillation_onnx_quantization"
QUICK = os.environ.get("QUICK") == "1"
OUT = os.path.expanduser("~/checkpoints/bert_opt")
MAX_LEN = 256
TEACHER, STUDENT = "bert-base-uncased", "distilbert-base-uncased"
N_EVAL_CPU = 200 if QUICK else 2000          # test reviews used to measure accuracy of every CPU variant
BENCH_RUNS = 20 if QUICK else 200
TEMPERATURE, ALPHA = 2.0, 0.5
FP16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def train_args(name, epochs):
    return TrainingArguments(
        output_dir=os.path.join(OUT, name), num_train_epochs=epochs, learning_rate=3e-5 if "student" in name else 2e-5,
        per_device_train_batch_size=32, per_device_eval_batch_size=128, warmup_steps=0.06, weight_decay=0.01,
        bf16=BF16, fp16=FP16, eval_strategy="no", save_strategy="no", logging_steps=100,
        report_to="none", seed=42, remove_unused_columns=False)


def accuracy_metric(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, -1) == labels).mean())}


class DistillationTrainer(Trainer):
    """Trainer whose loss mixes the true labels with the teacher's softened predictions."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits = inputs.pop("teacher_logits", None)   # absent at evaluation time
        outputs = model(**inputs)
        hard = F.cross_entropy(outputs.logits, inputs["labels"])
        if teacher_logits is None:
            return (hard, outputs) if return_outputs else hard
        soft = F.kl_div(F.log_softmax(outputs.logits / TEMPERATURE, -1),
                        F.softmax(teacher_logits / TEMPERATURE, -1),
                        reduction="batchmean") * TEMPERATURE ** 2   # T^2 keeps gradient scale comparable
        loss = ALPHA * hard + (1 - ALPHA) * soft
        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(42)
    torch.manual_seed(42)
    imdb = load_dataset("stanfordnlp/imdb")
    train, test = imdb["train"].shuffle(seed=42), imdb["test"].shuffle(seed=42)
    if QUICK:
        train, test = train.select(range(1000)), test.select(range(500))
    epochs = 1 if QUICK else 2
    tok = AutoTokenizer.from_pretrained(TEACHER)   # BERT and DistilBERT share the same WordPiece vocab
    tok.model_input_names = ["input_ids", "attention_mask"]  # DistilBERT has no token_type_ids input
    enc = lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN)
    train_tok = train.map(enc, batched=True, remove_columns=["text"])
    test_tok = test.map(enc, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tok)
    times = {}

    # ---- 1) Teacher -----------------------------------------------------------
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=2)
    t_trainer = Trainer(model=teacher, args=train_args("teacher", epochs), train_dataset=train_tok,
                        data_collator=collator, compute_metrics=accuracy_metric)
    with Timer() as t:
        t_trainer.train()
    times["teacher_train_min"] = t.minutes
    teacher_acc = t_trainer.evaluate(test_tok)["eval_accuracy"]
    # The teacher's logits on the training set = the "soft labels" for distillation.
    teacher_logits = t_trainer.predict(train_tok).predictions
    train_distill = train_tok.add_column("teacher_logits", [list(map(float, r)) for r in teacher_logits])

    # ---- 2a) Student without distillation ------------------------------------
    plain = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2)
    p_trainer = Trainer(model=plain, args=train_args("student_plain", epochs), train_dataset=train_tok,
                        data_collator=collator, compute_metrics=accuracy_metric)
    with Timer() as t:
        p_trainer.train()
    times["student_plain_train_min"] = t.minutes
    plain_acc = p_trainer.evaluate(test_tok)["eval_accuracy"]

    # ---- 2b) Student WITH distillation ---------------------------------------
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2)
    def distill_collator(features):
        # The padding collator only keeps tokenizer fields, so carry the teacher logits separately.
        if "teacher_logits" not in features[0]:          # evaluation batches
            return collator(features)
        logits = torch.tensor([f.pop("teacher_logits") for f in features], dtype=torch.float32)
        batch = collator(features)
        batch["teacher_logits"] = logits
        return batch

    d_trainer = DistillationTrainer(model=student, args=train_args("student_distilled", epochs),
                                    train_dataset=train_distill, data_collator=distill_collator,
                                    compute_metrics=accuracy_metric)
    with Timer() as t:
        d_trainer.train()
    times["student_distilled_train_min"] = t.minutes
    distilled_acc = d_trainer.evaluate(test_tok)["eval_accuracy"]
    print(f"GPU test accuracy: teacher {teacher_acc:.4f}  student {plain_acc:.4f}  distilled {distilled_acc:.4f}")

    # ---- 3) Export teacher + distilled student to ONNX, 4) int8-quantize ------
    teacher.to("cpu").eval()
    student.to("cpu").eval()
    paths = {}
    for name, model in [("teacher", teacher), ("student", student)]:
        paths[f"{name}_fp32"] = export_onnx(model, tok, os.path.join(OUT, f"{name}.onnx"))
        paths[f"{name}_int8"] = quantize(paths[f"{name}_fp32"])

    # ---- Benchmark on CPU ------------------------------------------------------
    eval_texts, eval_labels = test["text"][:N_EVAL_CPU], np.array(test["label"][:N_EVAL_CPU])
    torch.set_num_threads(os.cpu_count())
    rows = [
        bench_torch("BERT-base · PyTorch fp32 (teacher)", teacher, tok, eval_texts, eval_labels),
        bench_onnx("BERT-base · ONNX Runtime fp32", paths["teacher_fp32"], tok, eval_texts, eval_labels),
        bench_onnx("BERT-base · ONNX Runtime int8", paths["teacher_int8"], tok, eval_texts, eval_labels),
        bench_torch("DistilBERT (distilled) · PyTorch fp32", student, tok, eval_texts, eval_labels),
        bench_onnx("DistilBERT (distilled) · ONNX Runtime fp32", paths["student_fp32"], tok, eval_texts, eval_labels),
        bench_onnx("DistilBERT (distilled) · ONNX Runtime int8", paths["student_int8"], tok, eval_texts, eval_labels),
    ]
    base = rows[0]
    for r in rows:
        r["speedup_vs_bert_pytorch"] = base["latency_ms_p50"] / r["latency_ms_p50"]
        r["latency_reduction_percent"] = 100 * (1 - r["latency_ms_p50"] / base["latency_ms_p50"])
        r["size_reduction_percent"] = 100 * (1 - r["size_mb"] / base["size_mb"])
    write_report(rows, teacher_acc, plain_acc, distilled_acc, times)

    best = rows[-1]
    save_results(PROJECT, {
        "teacher_test_accuracy": teacher_acc, "student_plain_test_accuracy": plain_acc,
        "student_distilled_test_accuracy": distilled_acc,
        "final_latency_reduction_percent": best["latency_reduction_percent"],
        "final_speedup_x": best["speedup_vs_bert_pytorch"],
        "final_size_reduction_percent": best["size_reduction_percent"],
        "final_cpu_accuracy": best["accuracy"], **times,
    })


def export_onnx(model, tok, path):
    """Export a classifier to ONNX with dynamic batch/sequence dimensions."""
    dummy = tok(["an example review"], return_tensors="pt", padding="max_length", max_length=128)
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}}
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes=axes)
    try:  # classic TorchScript-based exporter
        torch.onnx.export(model, inputs, path, opset_version=17, dynamo=False, **kwargs)
    except Exception as e:  # newer PyTorch versions: the torch.export-based exporter
        print(f"legacy ONNX export failed ({e}); using dynamo exporter")
        torch.onnx.export(model, inputs, path, dynamo=True, external_data=False, **kwargs)
    return path


def quantize(fp32_path):
    """Dynamic quantization: int8 weights, activations quantized at runtime (no calibration data needed)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = fp32_path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


def latency(fn, runs):
    for _ in range(10):      # warm-up: caches, thread pools, lazy initialisation
        fn()
    ts = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def bench_torch(name, model, tok, texts, labels):
    one = tok([texts[0]], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.inference_mode():
        p50, p95 = latency(lambda: model(**one), BENCH_RUNS)
        preds = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
            preds += model(**b).logits.argmax(-1).tolist()
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    return {"variant": name, "latency_ms_p50": p50, "latency_ms_p95": p95, "size_mb": size,
            "accuracy": float((np.array(preds) == labels).mean())}


def bench_onnx(name, path, tok, texts, labels):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL   # operator fusion
    opts.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    feed = lambda enc: {"input_ids": enc["input_ids"].astype("int64"), "attention_mask": enc["attention_mask"].astype("int64")}
    one = feed(tok([texts[0]], return_tensors="np", truncation=True, padding="max_length", max_length=128))
    p50, p95 = latency(lambda: sess.run(None, one), BENCH_RUNS)
    preds = []
    for i in range(0, len(texts), 32):
        b = feed(tok(texts[i:i + 32], return_tensors="np", truncation=True, padding=True, max_length=MAX_LEN))
        preds += sess.run(None, b)[0].argmax(-1).tolist()
    return {"variant": name, "latency_ms_p50": p50, "latency_ms_p95": p95, "size_mb": os.path.getsize(path) / 2**20,
            "accuracy": float((np.array(preds) == labels).mean())}


def write_report(rows, teacher_acc, plain_acc, distilled_acc, times):
    out = results_dir(PROJECT)
    json.dump({"rows": rows, "cpu": platform.processor() or platform.machine(), "threads": os.cpu_count(),
               "gpu_accuracy": {"teacher": teacher_acc, "student_plain": plain_acc, "student_distilled": distilled_acc},
               **times}, open(os.path.join(out, "benchmark.json"), "w"), indent=2)
    with open(os.path.join(out, "benchmark.md"), "w") as f:
        f.write(f"CPU: {os.cpu_count()} threads · batch 1 · 128 tokens · accuracy on {N_EVAL_CPU} IMDB test reviews\n\n")
        f.write("| variant | p50 latency (ms) | p95 (ms) | size (MB) | accuracy | speed-up |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['variant']} | {r['latency_ms_p50']:.1f} | {r['latency_ms_p95']:.1f} | {r['size_mb']:.0f} "
                    f"| {100 * r['accuracy']:.1f} % | {r['speedup_vs_bert_pytorch']:.1f}x |\n")
        f.write(f"\nFull test set (25k, GPU): teacher {100 * teacher_acc:.2f} % · DistilBERT without distillation "
                f"{100 * plain_acc:.2f} % · DistilBERT with distillation {100 * distilled_acc:.2f} %\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [r["variant"].replace(" · ", "\n") for r in rows]
    ax.barh(labels, [r["latency_ms_p50"] for r in rows], color=["#999"] * 3 + ["#2a7ab9"] * 3)
    for i, r in enumerate(rows):
        ax.text(r["latency_ms_p50"], i, f"  {r['latency_ms_p50']:.0f} ms · {100 * r['accuracy']:.1f}%", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median CPU latency, batch 1, 128 tokens (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "latency.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
