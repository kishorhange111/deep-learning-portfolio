"""
Project 08 - Fine-tuning BERT for sentiment analysis (IMDB movie reviews)
=========================================================================

TASK: classify a movie review as positive or negative.

WHAT BERT IS (Devlin et al., 2018):
    A Transformer *encoder* pre-trained on Wikipedia + books with two
    self-supervised tasks: predict masked-out words ("masked language
    modelling") and predict whether two sentences follow each other.
    It reads text in BOTH directions at once, so every token's vector
    depends on its full context ("bank" in "river bank" vs "bank loan").

FINE-TUNING:
    * Tokenise with WordPiece:  [CLS] this movie was great ! [SEP]
    * Put a small linear classifier on the final vector of the [CLS] token.
    * Train the WHOLE network (110M parameters) for 2 epochs with a small
      learning rate (2e-5), linear warm-up/decay - the recipe from the paper.
    This is transfer learning: the model already understands English, it
    only has to learn what makes a review positive.

BASELINE: a TF-IDF + logistic-regression model is trained on the same data,
so the gain from the Transformer is measured, not assumed.

ENGINEERING: bf16 mixed precision, dynamic padding, max 256 tokens
(covers most reviews; longer ones are truncated).

Dataset : IMDB - 25,000 training and 25,000 test reviews, balanced.
Stack   : PyTorch + Hugging Face transformers/datasets (the industry-standard
          NLP stack; the other projects use Keras, so both are covered).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.utils import Timer, results_dir, save_results, set_seed

PROJECT = "08_bert_sentiment_imdb"
QUICK = os.environ.get("QUICK") == "1"

MODEL = "bert-base-uncased"
MAX_LEN = 256
EPOCHS = 1 if QUICK else float(os.environ.get("EPOCHS", "2"))
LR = 2e-5
BATCH = 32


def main():
    set_seed(42)
    torch.manual_seed(42)
    imdb = load_dataset("stanfordnlp/imdb")
    train, test = imdb["train"].shuffle(seed=42), imdb["test"].shuffle(seed=42)
    if QUICK:
        train, test = train.select(range(1000)), test.select(range(500))

    # ---- 1) Classical baseline: bag-of-words TF-IDF + logistic regression ----
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    xtr, xte = tfidf.fit_transform(train["text"]), tfidf.transform(test["text"])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(xtr, train["label"])
    base_pred = clf.predict(xte)
    baseline = {"baseline_tfidf_logreg_accuracy": accuracy_score(test["label"], base_pred),
                "baseline_tfidf_logreg_f1": f1_score(test["label"], base_pred)}
    print(baseline)

    # ---- 2) BERT ----
    tok = AutoTokenizer.from_pretrained(MODEL)

    def tokenize(batch):
        # Truncate only; padding is done per batch by the collator ("dynamic padding").
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train_tok = train.map(tokenize, batched=True, remove_columns=["text"])
    test_tok = test.map(tokenize, batched=True, remove_columns=["text"])

    # A fresh 2-class classification head goes on top of the pre-trained encoder.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})

    def metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, pred), "f1": f1_score(labels, pred)}

    out = os.path.expanduser("~/checkpoints/bert_imdb")
    args = TrainingArguments(
        output_dir=out,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=128,
        warmup_steps=0.06,              # 6% linear warm-up, then linear decay (BERT recipe)
        weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=test_tok,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    with Timer() as t:
        trainer.train()
    final = trainer.evaluate()

    # Save the fine-tuned model locally (not committed: ~440 MB).
    trainer.save_model(os.path.join(out, "final"))
    tok.save_pretrained(os.path.join(out, "final"))

    # A few predictions on hand-written reviews for the README.
    demo = ["An absolute masterpiece, I was moved to tears.",
            "Two hours of my life I will never get back.",
            "The acting was fine but the plot made no sense at all.",
            "Not bad at all - better than I expected!"]
    enc = tok(demo, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
    with open(os.path.join(results_dir(PROJECT), "sample_predictions.md"), "w") as f:
        f.write("| review | P(positive) |\n|---|---|\n")
        for text, p in zip(demo, probs):
            f.write(f"| {text} | {p:.3f} |\n")

    # Turn the Trainer log into per-epoch/per-step curves.
    logs = trainer.state.log_history
    history = {"loss": [l["loss"] for l in logs if "loss" in l]}
    with open(os.path.join(results_dir(PROJECT), "trainer_log.json"), "w") as f:
        json.dump(logs, f)

    save_results(
        PROJECT,
        {"test_accuracy": final["eval_accuracy"], "test_f1": final["eval_f1"], **baseline,
         "epochs": EPOCHS, "parameters": model.num_parameters(), "train_minutes": t.minutes},
        history,
        curves=("loss",),
    )


if __name__ == "__main__":
    main()
