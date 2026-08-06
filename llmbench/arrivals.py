"""Closed-loop pool + open-loop Poisson/Gamma arrival generator.

Gamma parameterisation verified against vllm/benchmarks/serve.py:393-504
(docs/reference-notes.md §3): shape=burstiness, scale=1/(rate*burstiness) -> mean
inter-arrival is always 1/rate regardless of burstiness; burstiness=1 degenerates to
Exponential(1/rate) (pure Poisson); burstiness -> inf is a literal constant 1/rate delay,
not relied on as a Gamma variance-to-zero limit. request_rate == inf -> zero delays
(send-all-at-once). The generated cumulative delay array is then linearly rescaled so the
last request's cumulative delay lands exactly on total_requests/request_rate (serve.py:
469-485) -- required for Gate E parity, easy to miss.
"""
from __future__ import annotations

import math
import random


def generate_arrival_delays(
    n_requests: int, request_rate: float, burstiness: float, rng: random.Random
) -> list[float]:
    """Returns per-request delay-since-previous-request in seconds (not cumulative)."""
    if n_requests <= 0:
        return []
    if request_rate == math.inf:
        return [0.0] * n_requests
    if burstiness == math.inf:
        return [1.0 / request_rate] * n_requests

    theta = 1.0 / (request_rate * burstiness)
    raw = [rng.gammavariate(burstiness, theta) for _ in range(n_requests)]
    cumsum = []
    total = 0.0
    for d in raw:
        total += d
        cumsum.append(total)
    target_total = n_requests / request_rate
    if cumsum[-1] > 0:
        scale = target_total / cumsum[-1]
        cumsum = [c * scale for c in cumsum]
    deltas = []
    prev = 0.0
    for c in cumsum:
        deltas.append(c - prev)
        prev = c
    return deltas


class ClosedLoopPool:
    """Holds exactly `concurrency` requests in flight at all times (spec §4.6)."""

    def __init__(self, concurrency: int):
        self.concurrency = max(1, concurrency)
