"""
Project 13 - Generating piano music with a Transformer (MAESTRO, PyTorch)
========================================================================

TASK: learn to continue a piano performance, note by note, and generate new music.

REPRESENTATION - "performance events" (Oore et al. 2018; used by Music Transformer):
    A MIDI performance becomes a sequence from a 388-token vocabulary:
        NOTE_ON(pitch)    x128   a key is pressed
        NOTE_OFF(pitch)   x128   a key is released
        TIME_SHIFT(t)     x100   advance time by 10 ms ... 1 s
        VELOCITY(v)       x32    how hard the next notes are played
    This captures expressive timing and dynamics, not just the score - so
    music generation becomes language modelling, exactly like GPT on text.

MODEL: a decoder-only Transformer (GPT) written from scratch in PyTorch:
    token + learned position embeddings -> 8 blocks of causal multi-head self-
    attention (PyTorch scaled_dot_product_attention = FlashAttention kernels)
    + MLP -> next-event softmax. Trained with random 1,024-event windows.

IMPROVEMENTS over the Keras example: our own tokenizer (no external package),
mixed precision, a proper held-out split by piece (MAESTRO's official test split),
and quantitative evaluation of generated music:
    * test perplexity per event
    * pitch-class histogram similarity: does generated music use the 12 notes
      in the same proportions as real music? (1 - Jensen-Shannon distance)
    * note density and mean velocity vs. real performances
Generated samples are saved as .mid files.

Data: MAESTRO v3 (Google Magenta) - ~200 h of virtuoso piano performances, CC BY-NC-SA 4.0.
"""
import json
import math
import os
import random
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import results_dir

PROJECT = "13_music_transformer_midi"
QUICK = os.environ.get("QUICK") == "1"
DEVICE = "cuda"
URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
DATA = os.path.expanduser("~/data/maestro")
SEQ = 1024
D_MODEL, N_LAYERS, N_HEADS = 512, 8, 8
BATCH = 16
TIME_BUDGET_MIN = 2 if QUICK else float(os.environ.get("TIME_BUDGET_MIN", "35"))
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ---------------------------------------------------------------------------
# Tokenizer: MIDI <-> performance events
# ---------------------------------------------------------------------------
NOTE_ON, NOTE_OFF, SHIFT, VEL = 0, 128, 256, 356
VOCAB = 388
PAD = VOCAB  # extra id used only for padding (never predicted)


def encode_midi(path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append((n.start, 1, n.pitch, n.velocity))   # 1 = on
            events.append((n.end, 0, n.pitch, 0))              # 0 = off (sorted before 'on' at the same time)
    events.sort()
    tokens, t, last_vel = [], 0.0, None
    for time_s, kind, pitch, vel in events:
        steps = int(round((time_s - t) * 100))                   # 10 ms resolution
        while steps > 0:
            s = min(steps, 100)
            tokens.append(SHIFT + s - 1)
            steps -= s
        t = time_s
        if kind == 1:
            v = min(vel // 4, 31)
            if v != last_vel:
                tokens.append(VEL + v)
                last_vel = v
            tokens.append(NOTE_ON + pitch)
        else:
            tokens.append(NOTE_OFF + pitch)
    return tokens


def decode_tokens(tokens, path):
    import pretty_midi

    pm, piano = pretty_midi.PrettyMIDI(), pretty_midi.Instrument(program=0)
    t, vel, open_notes = 0.0, 64, {}
    for tok in tokens:
        if tok < NOTE_OFF:
            open_notes[tok] = (t, vel)
        elif tok < SHIFT:
            p = tok - NOTE_OFF
            if p in open_notes:
                s, v = open_notes.pop(p)
                if t > s:
                    piano.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=t))
        elif tok < VEL:
            t += (tok - SHIFT + 1) / 100
        else:
            vel = (tok - VEL) * 4 + 2
    for p, (s, v) in open_notes.items():                        # close anything still held
        piano.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=s + 0.5))
    pm.instruments.append(piano)
    pm.write(path)
    return piano.notes


# ---------------------------------------------------------------------------
# Model: GPT
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.qkv, self.proj = nn.Linear(D_MODEL, 3 * D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, N_HEADS, c // N_HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1 if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, c)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB + 1, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, prime, n, temperature=1.0, top_p=0.95):
        idx = prime
        for _ in range(n):
            logits = self(idx[:, -SEQ:])[:, -1].float() / temperature
            probs = F.softmax(logits, -1)
            sp, si = probs.sort(descending=True)
            mask = sp.cumsum(-1) - sp > top_p                    # nucleus sampling
            sp[mask] = 0
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            idx = torch.cat([idx, nxt], 1)
        return idx


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def music_stats(tokens):
    on = [t for t in tokens if t < NOTE_OFF]
    secs = sum((t - SHIFT + 1) / 100 for t in tokens if SHIFT <= t < VEL) or 1e-6
    vels = [(t - VEL) * 4 + 2 for t in tokens if t >= VEL]
    pc = np.bincount([p % 12 for p in on], minlength=12).astype(float)
    return pc / max(pc.sum(), 1), len(on) / secs, float(np.mean(vels)) if vels else 0.0


def js_similarity(p, q):
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2((a + 1e-12) / (b + 1e-12)), 0)))
    return 1 - math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def main():
    import csv

    random.seed(0); torch.manual_seed(0)
    out = results_dir(PROJECT)
    os.makedirs(DATA, exist_ok=True)
    zpath = os.path.join(DATA, "maestro.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(URL, zpath)
        zipfile.ZipFile(zpath).extractall(DATA)
    root = os.path.join(DATA, "maestro-v3.0.0")
    rows = list(csv.DictReader(open(os.path.join(root, "maestro-v3.0.0.csv"))))
    split = {s: [os.path.join(root, r["midi_filename"]) for r in rows if r["split"] == s] for s in ["train", "validation", "test"]}
    if QUICK:
        split = {k: v[:20] for k, v in split.items()}
    t0 = time.time()
    from multiprocessing import Pool

    with Pool(os.cpu_count()) as pool:                          # MIDI parsing is CPU-bound: use every core
        enc = {k: pool.map(encode_midi, v) for k, v in split.items()}
    n_tok = {k: sum(map(len, v)) for k, v in enc.items()}
    print(f"tokenized in {(time.time() - t0) / 60:.1f} min: {n_tok}")
    train = [torch.tensor(s) for s in enc["train"] if len(s) > SEQ + 1]

    def batch():
        xs = []
        for _ in range(BATCH):
            s = random.choice(train)
            i = random.randint(0, len(s) - SEQ - 1)
            xs.append(s[i:i + SEQ + 1])
        x = torch.stack(xs).to(DEVICE)
        return x[:, :-1], x[:, 1:]

    model = GPT().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=AMP_DTYPE == torch.float16)
    history, it, start = [], 0, time.time()
    while time.time() - start < TIME_BUDGET_MIN * 60:
        lr = 3e-4 * min(1.0, (it + 1) / 500)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch()
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            loss = F.cross_entropy(model(x).view(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        it += 1
        if it % 200 == 0:
            history.append({"iter": it, "train_loss": loss.item()})
            print(history[-1], flush=True)
    train_min = (time.time() - start) / 60

    # ---- test perplexity on held-out pieces (non-overlapping windows) ----
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in enc["test"]:
            s = torch.tensor(s)
            for i in range(0, len(s) - SEQ - 1, SEQ):
                w = s[i:i + SEQ + 1].to(DEVICE)[None]
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    l = F.cross_entropy(model(w[:, :-1]).float().view(-1, VOCAB), w[0, 1:], reduction="sum")
                total += l.item(); count += SEQ
    ppl = math.exp(total / count)

    # ---- generate continuations of held-out openings and compare statistics with real music ----
    real = [t for s in enc["test"] for t in s]
    real_pc, real_density, real_vel = music_stats(real)
    gen_stats = []
    for k, s in enumerate(random.sample([s for s in enc["test"] if len(s) > 600], 3)):
        prime = torch.tensor(s[:256], device=DEVICE)[None]
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            g = model.generate(prime, 1024 if not QUICK else 64)[0].tolist()
        decode_tokens(g, os.path.join(out, f"sample_{k + 1}_continuation.mid"))
        decode_tokens(s[:256], os.path.join(out, f"sample_{k + 1}_prompt.mid"))
        gen_stats.append(music_stats(g[256:]))
    gen_pc = np.mean([g[0] for g in gen_stats], 0)
    metrics = {"params": params, "train_iterations": it, "train_minutes": train_min,
               "train_tokens": n_tok["train"], "test_perplexity": ppl,
               "pitch_class_similarity_generated_vs_real": js_similarity(gen_pc, real_pc),
               "notes_per_second_real": real_density,
               "notes_per_second_generated": float(np.mean([g[1] for g in gen_stats])),
               "mean_velocity_real": real_vel, "mean_velocity_generated": float(np.mean([g[2] for g in gen_stats])),
               "gpu": torch.cuda.get_device_name(0)}
    json.dump({"metrics": metrics, "history": history}, open(os.path.join(out, "metrics.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    x = np.arange(12)
    ax.bar(x - 0.2, real_pc, 0.4, label="real (MAESTRO test)")
    ax.bar(x + 0.2, gen_pc, 0.4, label="generated")
    ax.set_xticks(x, names); ax.legend(); ax.set_title("Pitch-class distribution")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pitch_classes.png"), dpi=110); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
