"""
Latency measurement helper shared by the model-optimisation benchmarks.

Reports percentiles instead of the mean: a few slow outliers (garbage
collection, OS scheduling) would distort a mean, while p50 / p95 match how
online serving latency is usually specified.
"""
import time

import numpy as np


def latency(fn, runs, warmup=10):
    """Call `fn` `warmup` times, then time `runs` calls; return (p50_ms, p95_ms)."""
    for _ in range(warmup):      # warm-up: caches, thread pools, lazy initialisation
        fn()
    ts = []
    for _ in range(runs):
        s = time.perf_counter()  # monotonic, high-resolution clock
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
