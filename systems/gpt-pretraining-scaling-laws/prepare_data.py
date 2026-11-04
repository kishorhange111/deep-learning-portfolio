"""
Tokenise FineWeb-Edu into flat uint16 shards for pretraining.

FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT) is web text filtered for educational value by a
classifier. Hugging Face's ablations found it trains small models better than raw web text. Documents
are encoded with GPT-2's BPE (tiktoken), joined with an <|endoftext|> separator, and written as raw
uint16 arrays, which np.memmap can read without loading into RAM.

    python prepare_data.py --train_tokens 2e9         # what the default IsoFLOP sweep needs (~4 GB)
    python prepare_data.py --train_tokens 5e7         # a quick 100 MB version
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "fineweb_edu")
SHARD_TOKENS = 100_000_000  # 200 MB per shard

_enc = None


def _init_worker():
    """Load the tokenizer once per worker process."""
    global _enc
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")


def encode_document(text):
    """GPT-2 BPE tokens of one document, prefixed with <|endoftext|> as a document boundary."""
    return np.array([_enc.eot_token, *_enc.encode_ordinary(text)], dtype=np.uint16)


class ShardWriter:
    """Accumulate token arrays and flush them to <prefix>_000.bin, <prefix>_001.bin, ..."""

    def __init__(self, out_dir, prefix, shard_tokens=SHARD_TOKENS):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.buffer, self.filled, self.shards, self.total = np.empty(shard_tokens, np.uint16), 0, [], 0

    def add(self, tokens):
        """Append tokens, writing out every full shard."""
        while len(tokens):
            take = min(len(tokens), self.shard_tokens - self.filled)
            self.buffer[self.filled : self.filled + take] = tokens[:take]
            self.filled, tokens = self.filled + take, tokens[take:]
            if self.filled == self.shard_tokens:
                self.flush()

    def flush(self):
        """Write the partially filled buffer as a shard."""
        if not self.filled:
            return
        path = os.path.join(self.out_dir, f"{self.prefix}_{len(self.shards):03d}.bin")
        self.buffer[: self.filled].tofile(path)
        self.shards.append({"path": os.path.basename(path), "tokens": int(self.filled)})
        self.total += self.filled
        self.filled = 0


def write_split(token_arrays, out_dir, prefix, max_tokens, shard_tokens=SHARD_TOKENS):
    """Consume an iterable of token arrays until max_tokens; returns the shard list."""
    writer = ShardWriter(out_dir, prefix, shard_tokens)
    for tokens in token_arrays:
        remaining = max_tokens - writer.total - writer.filled
        if remaining <= 0:
            break
        writer.add(tokens[:remaining])
    writer.flush()
    return writer.shards


def main(argv=None):
    """Stream, tokenise and shard: validation first (a fixed held-out slice), then training."""
    p = argparse.ArgumentParser()
    p.add_argument("--train_tokens", type=float, default=2e9)
    p.add_argument("--val_tokens", type=float, default=1e7)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args(argv)

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = (row["text"] for row in stream)
    with Pool(args.workers, initializer=_init_worker) as pool:
        tokens = pool.imap(encode_document, texts, chunksize=64)
        val = write_split(tokens, args.out_dir, "val", int(args.val_tokens))
        train = write_split(tokens, args.out_dir, "train", int(args.train_tokens))
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu sample-10BT",
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "val": val,
        "train": train,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {sum(s['tokens'] for s in train):,} train / {sum(s['tokens'] for s in val):,} val tokens")


if __name__ == "__main__":
    main()
