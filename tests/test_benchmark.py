"""Tests for the latency helper (src/dl_common/benchmark.py)."""

from dl_common.benchmark import latency


def test_latency_counts_warmup_and_runs():
    calls = []
    p50, p95 = latency(lambda: calls.append(1), runs=20, warmup=5)
    assert len(calls) == 25
    assert 0 <= p50 <= p95
