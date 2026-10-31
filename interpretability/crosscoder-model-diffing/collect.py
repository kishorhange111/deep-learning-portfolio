"""
Collect paired residual-stream activations from two models on identical tokens.

Both models (e.g. a base model and its chat fine-tune, or mini-chatgpt's SFT and DPO checkpoints)
read the same text. The residual stream after layer L is recorded for both at the same positions.
The text is half chat transcripts (UltraFeedback, formatted with the chat template) and half web
text (FineWeb-Edu), so latents specific to either model have a chance to show up.

Stored, as memory-mapped arrays in --out_dir:
  acts.bin    float16 (N, 2, d_model)     model A and model B activations at the same position
  tokens.bin  int32   (n_seqs, seq_len)   the token sequences (to show a latent's top contexts)
  index.bin   int32   (N, 2)              (sequence, position) of each activation
  meta.json   shapes, models, layer, and a per-model scale making mean ||x|| = sqrt(d_model)

Position 0 is skipped: the first token acts as an attention sink with norms orders of magnitude
larger than other positions, and it would dominate the reconstruction loss.

    python collect.py --model_a Qwen/Qwen2.5-0.5B --model_b Qwen/Qwen2.5-0.5B-Instruct --tokens 8e6
    python collect.py --model_a ~/mini-chatgpt/out/final --model_b ~/mini-chatgpt/out/dpo   # SFT vs DPO
"""

import argparse
import itertools
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def chat_texts(tokenizer):
    """UltraFeedback conversations (prompt + preferred answer) rendered with the chat template."""
    from datasets import load_dataset

    for row in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True):
        yield tokenizer.apply_chat_template(row["chosen"], tokenize=False)


def web_texts():
    """FineWeb-Edu documents."""
    from datasets import load_dataset

    for row in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
        yield row["text"]


def file_texts(path):
    """Texts from a local .jsonl file with a 'text' field (offline use and tests)."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


def text_stream(args, tokenizer):
    """Alternate chat and web documents (or read a local file)."""
    if args.texts_file:
        return itertools.cycle(list(file_texts(args.texts_file)))
    sources = {"chat": chat_texts(tokenizer), "web": web_texts()}
    streams = [sources[s] for s in args.sources]
    return (next(s) for s in itertools.cycle(streams))


def residual_at(model, input_ids, attention_mask, layer):
    """Residual stream after `layer` blocks (hidden_states[0] is the embedding output)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer]


@torch.no_grad()
def collect(args):
    """Run both models over the text stream and write the memmaps; returns meta."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_a)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    probe = "Model diffing needs identical tokens: 123 <|im_start|>"
    assert tok(probe)["input_ids"] == tok_b(probe)["input_ids"], "the two models must share a tokenizer"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # real tokens start at position 0, so position ids need no correction
    models = [
        AutoModelForCausalLM.from_pretrained(m, dtype=dtype).to(device).eval() for m in (args.model_a, args.model_b)
    ]
    d = models[0].config.hidden_size
    assert models[1].config.hidden_size == d
    layer = args.layer if args.layer is not None else models[0].config.num_hidden_layers // 2

    n = int(args.tokens)
    n_seqs = math.ceil(n / args.positions_per_seq) + args.batch
    os.makedirs(args.out_dir, exist_ok=True)
    acts = np.memmap(os.path.join(args.out_dir, "acts.bin"), dtype=np.float16, mode="w+", shape=(n, 2, d))
    tokens = np.memmap(
        os.path.join(args.out_dir, "tokens.bin"), dtype=np.int32, mode="w+", shape=(n_seqs, args.seq_len)
    )
    index = np.memmap(os.path.join(args.out_dir, "index.bin"), dtype=np.int32, mode="w+", shape=(n, 2))
    rng = np.random.default_rng(args.seed)
    stream = text_stream(args, tok)
    filled, seq_base, norm_sums = 0, 0, np.zeros(2)
    while filled < n:
        texts = [next(stream) for _ in range(args.batch)]
        enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.seq_len)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        h = torch.stack([residual_at(m, ids, mask, layer) for m in models], dim=2).float()  # (B, T, 2, d)
        tokens[seq_base : seq_base + len(texts)] = enc["input_ids"].numpy()
        for b in range(len(texts)):
            valid = np.nonzero(enc["attention_mask"][b].numpy())[0]
            valid = valid[valid > 0]  # skip the attention-sink first position
            if len(valid) == 0:
                continue
            take = min(args.positions_per_seq, len(valid), n - filled)
            pos = np.sort(rng.choice(valid, size=take, replace=False))
            chunk = h[b, torch.from_numpy(pos).to(device)].cpu().numpy()
            acts[filled : filled + take] = chunk.astype(np.float16)
            index[filled : filled + take] = np.stack([np.full(take, seq_base + b), pos], axis=1)
            norm_sums += np.linalg.norm(chunk, axis=-1).sum(0)
            filled += take
            if filled >= n:
                break
        seq_base += len(texts)
        if seq_base % (args.batch * 50) == 0:
            print(f"{filled:,}/{n:,} activations")
    acts.flush()
    tokens.flush()
    index.flush()
    mean_norm = norm_sums / n
    meta = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "layer": layer,
        "d_model": d,
        "n": n,
        "n_seqs": seq_base,
        "seq_len": args.seq_len,
        "sources": args.sources if not args.texts_file else "file",
        "mean_norm": mean_norm.tolist(),
        "scale": (math.sqrt(d) / mean_norm).tolist(),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_a", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--model_b", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layer", type=int, default=None, help="default: the middle layer")
    p.add_argument("--tokens", type=float, default=8e6, help="activations to store (x2 models x d x 2 bytes)")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--positions_per_seq", type=int, default=64, help="subsample positions for diversity")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sources", nargs="*", default=["chat", "web"])
    p.add_argument("--texts_file", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "data", "base_vs_instruct"))
    return collect(p.parse_args(argv))


if __name__ == "__main__":
    main()
