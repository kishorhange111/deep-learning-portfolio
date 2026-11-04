"""
Read the paired activations written by collect.py: random minibatches from the (N, 2, d) float16
memmap, scaled per model so that the mean norm is about sqrt(d_model) (as in the crosscoder paper).
"""

import json
import os

import numpy as np
import torch


class ActivationData:
    """Random minibatches from the (N, 2, d) float16 memmap, scaled so mean ||x|| ~ sqrt(d) per model."""

    def __init__(self, data_dir, seed):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        n, d = self.meta["n"], self.meta["d_model"]
        self.acts = np.memmap(os.path.join(data_dir, "acts.bin"), dtype=np.float16, mode="r", shape=(n, 2, d))
        self.scale = torch.tensor(self.meta["scale"], dtype=torch.float32).view(1, 2, 1)
        self.rng = np.random.default_rng(seed)

    def batch(self, size, device):
        """Sorted random indices read faster from a memmap; order within a batch does not matter."""
        idx = np.sort(self.rng.integers(0, len(self.acts), size=size))  # with replacement: duplicates are negligible
        return (torch.from_numpy(np.asarray(self.acts[idx], dtype=np.float32)) * self.scale).to(device)

    def first(self, size, device):
        """A fixed evaluation slice."""
        return (torch.from_numpy(np.asarray(self.acts[:size], dtype=np.float32)) * self.scale).to(device)
