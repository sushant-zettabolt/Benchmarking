"""get_ts()/avg()/stdev() replicated exactly from llama-bench.cpp:102-118,1523-1529
(docs/reference-notes.md §1), verified against hand-computed values.
"""
import math

from llmbench.metrics import TsSample, percentile, population_stddev, sample_stddev


def test_sample_stddev_matches_hand_computed():
    values = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    # numpy ddof=1 reference value
    expected = 5.237229365663817
    assert math.isclose(sample_stddev(values), expected, rel_tol=1e-9)


def test_sample_stddev_zero_at_n_leq_1():
    assert sample_stddev([]) == 0.0
    assert sample_stddev([42.0]) == 0.0


def test_population_stddev_matches_hand_computed():
    values = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    expected = 4.898979485566356  # numpy ddof=0
    assert math.isclose(population_stddev(values), expected, rel_tol=1e-9)


def test_percentile_linear_interpolation():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 50) == 3
    assert math.isclose(percentile(values, 25), 2.0)
    assert math.isclose(percentile(values, 90), 4.6)


def test_get_ts_formula_avg_over_per_rep_values_not_avg_tokens_over_avg_time():
    # Reps with differing token counts *and* differing times -- avg(ts) != avg(tokens)/avg(time)
    samples = [TsSample(n_tokens=100, t_ns=1_000_000_000), TsSample(n_tokens=100, t_ns=2_000_000_000)]
    ts_values = [s.ts for s in samples]  # [100.0, 50.0]
    avg_ts = sum(ts_values) / len(ts_values)
    avg_tokens_over_avg_time = (sum(s.n_tokens for s in samples) / len(samples)) / (
        sum(s.t_ns for s in samples) / len(samples) / 1e9
    )
    assert math.isclose(avg_ts, 75.0)
    assert math.isclose(avg_tokens_over_avg_time, 66.66666666666667)
    assert not math.isclose(avg_ts, avg_tokens_over_avg_time)
