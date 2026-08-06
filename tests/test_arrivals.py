"""Gamma inter-arrival generator distribution properties, per vllm/benchmarks/serve.py:
393-504 (docs/reference-notes.md §3): mean inter-arrival always 1/rate regardless of
burstiness; burstiness=1 -> Exponential(1/rate); the cumulative delay array is rescaled so
the last request's cumulative delay lands on total_requests/request_rate.
"""
import math
import random
import statistics

from llmbench.arrivals import generate_arrival_delays


def test_zero_requests():
    assert generate_arrival_delays(0, 1.0, 1.0, random.Random(0)) == []


def test_infinite_rate_is_all_zero_delay():
    delays = generate_arrival_delays(50, math.inf, 1.0, random.Random(0))
    assert delays == [0.0] * 50


def test_infinite_burstiness_is_constant_delay():
    delays = generate_arrival_delays(10, 2.0, math.inf, random.Random(0))
    assert all(math.isclose(d, 0.5) for d in delays)


def test_mean_inter_arrival_is_one_over_rate_regardless_of_burstiness():
    rng = random.Random(1)
    for burstiness in (0.5, 1.0, 2.0, 5.0):
        delays = generate_arrival_delays(2000, 10.0, burstiness, rng)
        assert math.isclose(statistics.mean(delays), 0.1, rel_tol=0.05)


def test_cumulative_delay_rescaled_to_exact_target_total():
    rng = random.Random(2)
    n, rate = 500, 20.0
    delays = generate_arrival_delays(n, rate, 1.0, rng)
    assert math.isclose(sum(delays), n / rate, rel_tol=1e-9)


def test_all_delays_nonnegative():
    rng = random.Random(3)
    delays = generate_arrival_delays(200, 5.0, 0.3, rng)
    assert all(d >= 0 for d in delays)
