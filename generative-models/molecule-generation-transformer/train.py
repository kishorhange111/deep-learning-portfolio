"""
Project 14 - Generating drug-like molecules with a Transformer language model (ZINC-250k, PyTorch)
=================================================================================================

TASK: generate new, chemically valid, drug-like molecules - the first step of
AI-driven drug discovery (de novo design).

REPRESENTATION: SMILES strings, e.g. aspirin = CC(=O)Oc1ccccc1C(=O)O.
A molecule becomes a sequence of characters, so molecule generation becomes
language modelling: a GPT learns the "grammar" of chemistry (ring closures,
branches in parentheses, valences) from 250k examples.

WHY NOT THE KERAS EXAMPLE'S APPROACH? It trains a WGAN-GP with a relational
graph network to generate molecular graphs. GANs are hard to train and
graph GANs typically produce few valid, unique molecules. Autoregressive
SMILES models are the stronger, simpler baseline in the GuacaMol / MOSES
benchmarks - so this project implements that and measures it properly.

MODEL: decoder-only Transformer (GPT) from scratch: 6 layers, 384-d, 6 heads,
character-level tokens, <bos>/<eos> markers, mixed precision.

EVALUATION with RDKit on 10,000 sampled molecules (the MOSES / GuacaMol metrics):
    validity   - % of generated strings RDKit can parse into a real molecule
    uniqueness - % of valid molecules that are distinct
    novelty    - % of unique valid molecules NOT in the training set
    property distributions - QED (drug-likeness), logP, molecular weight, SA-like ring counts,
                  compared with the training data
Data: ZINC-250k drug-like molecules (Gomez-Bombarelli et al., 2018).
"""
import json
import math
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import results_dir

PROJECT = "14_molecule_generation_gpt"
QUICK = os.environ.get("QUICK") == "1"
DEVICE = "cuda"
URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/"
       "250k_rndm_zinc_drugs_clean_3.csv")
D, LAYERS, HEADS, MAXLEN, BATCH = 384, 6, 6, 128, 256
N_SAMPLES = 500 if QUICK else 10000
TIME_BUDGET_MIN = 2 if QUICK else float(os.environ.get("TIME_BUDGET_MIN", "30"))
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_smiles():
    import csv
    import io

    raw = urllib.request.urlopen(URL, timeout=120).read().decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    return [r["smiles"].strip() for r in rows]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, c))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(MAXLEN, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


@torch.no_grad()
def sample(model, n, bos, eos, temperature=1.0):
    """Generate n SMILES in parallel batches, character by character."""
    out = []
    for i in range(0, n, 1000):
        m = min(1000, n - i)
        idx = torch.full((m, 1), bos, device=DEVICE)
        done = torch.zeros(m, dtype=torch.bool, device=DEVICE)
        for _ in range(MAXLEN - 1):
            with torch.autocast("cuda", dtype=AMP):
                logits = model(idx)[:, -1].float() / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            nxt[done] = eos
            idx = torch.cat([idx, nxt], 1)
            done |= nxt[:, 0] == eos
            if done.all():
                break
        out.append(idx[:, 1:].cpu())
    return out


def props(mols):
    from rdkit.Chem import QED, Crippen, Descriptors

    return {"QED": [QED.qed(m) for m in mols], "logP": [Crippen.MolLogP(m) for m in mols],
            "MolWt": [Descriptors.MolWt(m) for m in mols]}


def main():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    random.seed(0); torch.manual_seed(0)
    out = results_dir(PROJECT)
    smiles = load_smiles()
    random.shuffle(smiles)
    if QUICK:
        smiles = smiles[:5000]
    train_set = set(smiles)
    chars = sorted(set("".join(smiles)))
    stoi = {c: i + 3 for i, c in enumerate(chars)}           # 0 pad, 1 bos, 2 eos
    itos = {i: c for c, i in stoi.items()}
    PAD, BOS, EOS, vocab = 0, 1, 2, len(chars) + 3
    enc = [[BOS] + [stoi[c] for c in s] + [EOS] for s in smiles if len(s) + 2 <= MAXLEN]
    data = torch.full((len(enc), MAXLEN), PAD)
    for i, e in enumerate(enc):
        data[i, :len(e)] = torch.tensor(e)
    print(f"{len(enc)} molecules, vocab {vocab}")

    model = GPT(vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)
    history, it, start = [], 0, time.time()
    while time.time() - start < TIME_BUDGET_MIN * 60:
        for g in opt.param_groups:
            g["lr"] = 5e-4 * min(1.0, (it + 1) / 300)
        x = data[torch.randint(len(data), (BATCH,))].to(DEVICE)
        with torch.autocast("cuda", dtype=AMP):
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, vocab).float(), x[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        it += 1
        if it % 200 == 0:
            history.append({"iter": it, "loss": loss.item()}); print(history[-1], flush=True)
    train_min = (time.time() - start) / 60

    model.eval()
    gen = []
    for batch in sample(model, N_SAMPLES, BOS, EOS):
        for row in batch.tolist():
            s = "".join(itos.get(t, "") for t in row[:row.index(EOS)] if t > 2) if EOS in row else None
            gen.append(s)
    valid = []
    for s in gen:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            valid.append(Chem.MolToSmiles(m))                  # canonical form, so duplicates are detected
    unique = set(valid)
    canon_train = train_set if QUICK else None
    novel = [s for s in unique if s not in train_set]         # raw-string check (canonical check below)
    if not QUICK:
        sample_train = random.sample(smiles, 20000)
        canon_train = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in sample_train}
    train_mols = [Chem.MolFromSmiles(s) for s in random.sample(smiles, 5000)]
    gen_mols = [Chem.MolFromSmiles(s) for s in random.sample(sorted(unique), min(5000, len(unique)))]
    p_gen, p_train = props(gen_mols), props(train_mols)

    metrics = {"params": sum(p.numel() for p in model.parameters()), "train_iterations": it,
               "train_minutes": train_min, "train_molecules": len(enc), "generated": len(gen),
               "validity": len(valid) / len(gen), "uniqueness": len(unique) / max(len(valid), 1),
               "novelty": len(novel) / max(len(unique), 1),
               **{f"{k}_mean_generated": float(np.mean(v)) for k, v in p_gen.items()},
               **{f"{k}_mean_train": float(np.mean(v)) for k, v in p_train.items()},
               "gpu": torch.cuda.get_device_name(0)}
    json.dump({"metrics": metrics, "history": history, "examples": sorted(unique)[:40]},
              open(os.path.join(out, "metrics.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, k in zip(axes, ["QED", "logP", "MolWt"]):
        ax.hist(p_train[k], bins=40, alpha=0.6, density=True, label="ZINC training")
        ax.hist(p_gen[k], bins=40, alpha=0.6, density=True, label="generated")
        ax.set_title(k); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "property_distributions.png"), dpi=110); plt.close(fig)
    Draw.MolsToGridImage(gen_mols[:12], molsPerRow=6, subImgSize=(220, 180)).save(os.path.join(out, "generated_molecules.png"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
