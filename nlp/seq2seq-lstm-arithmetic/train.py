"""
Project 07 - Teaching an LSTM to add numbers (sequence-to-sequence)
===================================================================

TASK: input the *string* "535+61", output the *string* "596".
      The network is never told what "+" means or how digits work; it must
      learn addition (including carrying) purely from 50,000 examples.

WHY IT'S INTERESTING: it is the smallest possible example of the
encoder-decoder (seq2seq) architecture used for machine translation before
Transformers (Sutskever et al., 2014):
    * Encoder LSTM reads the input characters one by one and compresses the
      whole question into one 128-d vector (its final hidden state).
    * RepeatVector copies that vector once per output position.
    * Decoder LSTM unrolls it into the answer, one character per step.
    * A softmax over the 12 possible characters at each output position.

TRICK: the input string is REVERSED ("16+535"). Addition works right-to-left
(carries flow from the last digit), so reversing puts the digits that matter
first close to the output that needs them - shorter dependency paths,
much faster learning (Zaremba & Sutskever, 2014, "Learning to Execute").

Adapted from the Keras example by Smerity & others (Apache-2.0):
https://keras.io/examples/nlp/addition_rnn/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import keras
import numpy as np
from keras import layers

from common.utils import Timer, make_time_limit_callback, results_dir, save_results, set_seed

PROJECT = "07_seq2seq_addition_lstm"
QUICK = os.environ.get("QUICK") == "1"

TRAINING_SIZE = 50000
DIGITS = 3                      # numbers up to 999
REVERSE = True
MAXLEN = DIGITS + 1 + DIGITS    # longest question: "999+999" = 7 chars
CHARS = "0123456789+ "          # 12 symbols (space = padding)
EPOCHS = 2 if QUICK else int(os.environ.get("EPOCHS", "40"))
BATCH_SIZE = 32
TIME_LIMIT_MIN = float(os.environ.get("TIME_LIMIT_MIN", "15"))


class CharacterTable:
    """One-hot encode strings to (len, 12) arrays and decode predictions back to strings."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s, rows):
        x = np.zeros((rows, len(self.chars)), dtype="float32")
        for i, c in enumerate(s):
            x[i, self.index[c]] = 1
        return x

    def decode(self, x):
        return "".join(self.chars[i] for i in x.argmax(-1))


def generate_data(rng):
    """50,000 unique random 'a+b' questions (a+b and b+a count as the same)."""
    questions, answers, seen = [], [], set()
    while len(questions) < TRAINING_SIZE:
        a, b = (int("".join(rng.choice(list("0123456789")) for _ in range(rng.integers(1, DIGITS + 1))))
                for _ in range(2))
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        q = f"{a}+{b}".ljust(MAXLEN)
        questions.append(q[::-1] if REVERSE else q)
        answers.append(str(a + b).ljust(DIGITS + 1))
    return questions, answers


def main():
    set_seed(42)
    rng = np.random.default_rng(42)
    table = CharacterTable(CHARS)
    questions, answers = generate_data(rng)
    x = np.stack([table.encode(q, MAXLEN) for q in questions])        # (50000, 7, 12)
    y = np.stack([table.encode(a, DIGITS + 1) for a in answers])      # (50000, 4, 12)
    split = len(x) - len(x) // 10
    x_train, x_val, y_train, y_val = x[:split], x[split:], y[:split], y[split:]

    model = keras.Sequential([
        layers.Input((MAXLEN, len(CHARS))),
        layers.LSTM(128),                                  # encoder -> one 128-d "thought vector"
        layers.RepeatVector(DIGITS + 1),                   # feed it to each of the 4 output steps
        layers.LSTM(128, return_sequences=True),           # decoder
        layers.Dense(len(CHARS), activation="softmax"),    # one character per step
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    with Timer() as t:
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(x_val, y_val),
                            callbacks=[make_time_limit_callback(TIME_LIMIT_MIN)], verbose=2)

    # Character accuracy is what Keras reports; *exact-match* accuracy (every
    # digit of the answer right) is the honest metric for arithmetic.
    pred = model.predict(x_val, verbose=0).argmax(-1)
    exact = float((pred == y_val.argmax(-1)).all(axis=1).mean())

    with open(os.path.join(results_dir(PROJECT), "sample_predictions.md"), "w") as f:
        f.write("| question | true | predicted | |\n|---|---|---|---|\n")
        for i in range(20):
            q = table.decode(x_val[i])
            q = q[::-1] if REVERSE else q
            true, guess = table.decode(y_val[i]).strip(), "".join(table.chars[j] for j in pred[i]).strip()
            f.write(f"| {q.strip()} | {true} | {guess} | {'OK' if true == guess else 'X'} |\n")

    save_results(
        PROJECT,
        {"val_char_accuracy": history.history["val_accuracy"][-1], "val_exact_match_accuracy": exact,
         "epochs_trained": len(history.history["loss"]),
         "parameters": model.count_params(), "train_minutes": t.minutes},
        history.history,
        curves=("loss", "accuracy"),
    )


if __name__ == "__main__":
    main()
