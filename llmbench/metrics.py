"""RawRecord -> Result rows. ONE implementation for both backends.

Enforced by tests/test_metrics_no_backend_branch.py: this module must never import
llmbench.backends, never read a `.backend` / `.backend_version` field, and never contain a
literal "llamacpp" or "vllm" string. All backend identity/provenance is stitched onto
results by runner.py *after* calling this module, from the same raw records, using
generic fields only (timestamps, token counts, server_* numeric fields). If you find
yourself writing an `if` that depends on which backend produced a record, that branch
belongs in backends/*.py's normalisation, not here (course_correct.txt §1/§5).

`t/s` (llama-bench parity, docs/reference-notes.md §1): per-repetition
ts_i = 1e9 * n_tokens / t_ns_i, then avg/stdev **over the per-rep values**, sample stdev
(N-1, explicit 0 at N<=1) -- not avg(tokens)/avg(time). Every other stat column in the
detailed report uses population stdev (ddof=0, matching vLLM/numpy) and that convention is
labelled explicitly wherever it appears (docs/reference-notes.md §6 item 6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def sample_stddev(values: list[float]) -> float:
    """llama-bench.cpp:102-118's stdev(): sample (N-1), explicit 0.0 at N<=1."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    sq_sum = sum(v * v for v in values)
    var = sq_sum / (n - 1) - mean * mean * n / (n - 1)
    return math.sqrt(max(var, 0.0))


def population_stddev(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(max(var, 0.0))


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy.percentile's default ('linear')."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[int(lo)]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


@dataclass
class Stats:
    n: int = 0
    mean: float = 0.0
    stddev_sample: float = 0.0
    stddev_pop: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def from_values(cls, values: list[float]) -> "Stats":
        if not values:
            return cls()
        return cls(
            n=len(values),
            mean=sum(values) / len(values),
            stddev_sample=sample_stddev(values),
            stddev_pop=population_stddev(values),
            median=percentile(values, 50),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
        )


@dataclass
class TsSample:
    """One repetition's llama-bench-parity throughput sample."""

    n_tokens: int
    t_ns: float

    @property
    def ts(self) -> float:
        if self.t_ns <= 0:
            return 0.0
        return 1e9 * self.n_tokens / self.t_ns


@dataclass
class ResultRow:
    src: str  # "client" | "server"; server rows are diagnostics only (docs/contract.md)
    test_name: str
    n_reps: int
    n_reps_valid: int

    tps_mean: float
    tps_stddev: float  # sample stdev -- the llama-bench-parity `t/s` column

    ttft_ms: Stats = field(default_factory=Stats)
    tpot_ms: Stats = field(default_factory=Stats)
    itl_ms: Stats = field(default_factory=Stats)
    e2e_ms: Stats = field(default_factory=Stats)
    prefill_tps: Stats = field(default_factory=Stats)
    decode_tps: Stats = field(default_factory=Stats)

    request_throughput: float | None = None
    total_token_throughput: float | None = None
    overhead_ms: Stats | None = None

    n_prompt_actual_mean: float | None = None
    cached_tokens_mean: float | None = None
    preemptions_delta_total: int | None = None

    flags: list[str] = field(default_factory=list)


def _mean_of(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if not r.get("error") and r.get(key) is not None]
    return sum(values) / len(values) if values else None


def _sum_of(records: list[dict], key: str) -> int | None:
    values = [r[key] for r in records if not r.get("error") and r.get(key) is not None]
    return int(sum(values)) if values else None


class RawFields:
    """Generic accessors over raw record dicts -- avoids importing RawRecord (and therefore
    keeps this module decoupled from records.py's provenance fields) while staying typed
    enough to be useful. Callers pass a list of dicts (RawRecord.to_dict())."""


def _client_ts_samples(records: list[dict]) -> list[TsSample]:
    out = []
    for r in records:
        if r.get("error"):
            continue
        n_tokens = (r.get("n_prompt_actual") or 0) + (r.get("n_gen_actual") or 0)
        t_ns = (r.get("t_end_ns") or 0) - (r.get("t_send_ns") or 0)
        if t_ns > 0 and n_tokens > 0:
            out.append(TsSample(n_tokens=n_tokens, t_ns=t_ns))
    return out


def _server_ts_samples(records: list[dict]) -> list[TsSample]:
    out = []
    for r in records:
        if r.get("error"):
            continue
        prompt_ms = r.get("server_prompt_ms")
        pred_ms = r.get("server_predicted_ms")
        if prompt_ms is None and pred_ms is None:
            continue
        t_ns = ((prompt_ms or 0.0) + (pred_ms or 0.0)) * 1e6
        n_tokens = (r.get("server_prompt_n") or r.get("n_prompt_actual") or 0) + (
            r.get("server_predicted_n") or r.get("n_gen_actual") or 0
        )
        if t_ns > 0 and n_tokens > 0:
            out.append(TsSample(n_tokens=n_tokens, t_ns=t_ns))
    return out


def _ttft_ms_values(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("error") or r.get("t_first_token_ns") is None:
            continue
        out.append((r["t_first_token_ns"] - r["t_send_ns"]) / 1e6)
    return out


def _e2e_ms_values(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("error"):
            continue
        out.append((r["t_end_ns"] - r["t_send_ns"]) / 1e6)
    return out


def _tpot_ms_values(records: list[dict]) -> list[float]:
    """(E2E - TTFT) / (n_gen - 1), guarded n_gen > 1 -- serve.py:607-613's formula."""
    out = []
    for r in records:
        if r.get("error") or r.get("t_first_token_ns") is None:
            continue
        n_gen = r.get("n_gen_actual") or 0
        if n_gen <= 1:
            continue
        e2e_ns = r["t_end_ns"] - r["t_send_ns"]
        ttft_ns = r["t_first_token_ns"] - r["t_send_ns"]
        out.append((e2e_ns - ttft_ns) / 1e6 / (n_gen - 1))
    return out


def _itl_ms_values(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("error"):
            continue
        out.extend(ns / 1e6 for ns in (r.get("itl_ns") or []))
    return out


def _prefill_tps_values(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("error") or r.get("t_first_token_ns") is None:
            continue
        n_prompt = r.get("n_prompt_actual") or 0
        t_ns = r["t_first_token_ns"] - r["t_send_ns"]
        if n_prompt > 0 and t_ns > 0:
            out.append(1e9 * n_prompt / t_ns)
    return out


def _decode_tps_values(records: list[dict]) -> list[float]:
    out = []
    for r in records:
        if r.get("error") or r.get("t_first_token_ns") is None:
            continue
        n_gen = r.get("n_gen_actual") or 0
        t_ns = r["t_end_ns"] - r["t_first_token_ns"]
        if n_gen > 1 and t_ns > 0:
            out.append(1e9 * (n_gen - 1) / t_ns)
    return out


def _run_duration_s(records: list[dict]) -> float:
    valid = [r for r in records if not r.get("error")]
    if not valid:
        return 0.0
    start = min(r["t_send_ns"] for r in valid)
    end = max(r["t_end_ns"] for r in valid)
    return max(end - start, 1) / 1e9


def aggregate_client(records: list[dict], test_name: str) -> ResultRow:
    ts_samples = _client_ts_samples(records)
    ts_values = [s.ts for s in ts_samples]
    n_valid = sum(1 for r in records if not r.get("error"))
    dur_s = _run_duration_s(records)
    total_output = sum(r.get("n_gen_actual") or 0 for r in records if not r.get("error"))
    total_input = sum(r.get("n_prompt_actual") or 0 for r in records if not r.get("error"))
    row = ResultRow(
        src="client",
        test_name=test_name,
        n_reps=len(records),
        n_reps_valid=n_valid,
        tps_mean=sum(ts_values) / len(ts_values) if ts_values else 0.0,
        tps_stddev=sample_stddev(ts_values),
        ttft_ms=Stats.from_values(_ttft_ms_values(records)),
        tpot_ms=Stats.from_values(_tpot_ms_values(records)),
        itl_ms=Stats.from_values(_itl_ms_values(records)),
        e2e_ms=Stats.from_values(_e2e_ms_values(records)),
        prefill_tps=Stats.from_values(_prefill_tps_values(records)),
        decode_tps=Stats.from_values(_decode_tps_values(records)),
        request_throughput=n_valid / dur_s if dur_s > 0 else None,
        total_token_throughput=(total_input + total_output) / dur_s if dur_s > 0 else None,
        overhead_ms=compute_overhead_ms(records),
        n_prompt_actual_mean=_mean_of(records, "n_prompt_actual"),
        cached_tokens_mean=_mean_of(records, "cached_tokens"),
        preemptions_delta_total=_sum_of(records, "preemptions_delta"),
    )
    _apply_trap_flags(row, records)
    return row


def aggregate_server(records: list[dict], test_name: str) -> ResultRow | None:
    """Returns None if no server-side timing data is present at all (src=server -> N/A,
    never faked -- spec §3)."""
    ts_samples = _server_ts_samples(records)
    if not ts_samples:
        return None
    ts_values = [s.ts for s in ts_samples]
    n_valid = sum(1 for r in records if not r.get("error"))
    return ResultRow(
        src="server",
        test_name=test_name,
        n_reps=len(records),
        n_reps_valid=n_valid,
        tps_mean=sum(ts_values) / len(ts_values) if ts_values else 0.0,
        tps_stddev=sample_stddev(ts_values),
        prefill_tps=Stats.from_values(
            [1e9 * r["server_prompt_n"] / (r["server_prompt_ms"] * 1e6) for r in records
             if not r.get("error") and r.get("server_prompt_n") and r.get("server_prompt_ms")]
        ),
        decode_tps=Stats.from_values(
            [1e9 * r["server_predicted_n"] / (r["server_predicted_ms"] * 1e6) for r in records
             if not r.get("error") and r.get("server_predicted_n") and r.get("server_predicted_ms")]
        ),
    )


def compute_overhead_ms(records: list[dict]) -> Stats:
    """overhead_ms = client_e2e - server_total (spec §3). Not comparable in meaning across
    backends -- see docs/contract.md; this function only computes the number, the meaning
    caveat lives in the printer legend."""
    values = []
    for r in records:
        if r.get("error"):
            continue
        prompt_ms = r.get("server_prompt_ms")
        pred_ms = r.get("server_predicted_ms")
        if prompt_ms is None and pred_ms is None:
            continue
        server_total_ms = (prompt_ms or 0.0) + (pred_ms or 0.0)
        client_e2e_ms = (r["t_end_ns"] - r["t_send_ns"]) / 1e6
        values.append(client_e2e_ms - server_total_ms)
    return Stats.from_values(values)


def _apply_trap_flags(row: ResultRow, records: list[dict]) -> None:
    """Trap detection (spec §11): prefix-caching defeat, preemption, depth verification.

    cache_suspected (rep 2's prefill >2x rep 1's) is a c=1, closed-loop heuristic: under
    concurrency>1 or open-loop dispatch, list order is dispatch order but *not* execution
    order, so a rep-2-faster-than-rep-1 reading can reflect queueing/scheduling variance
    rather than an actual prefix-cache hit (observed live: a false positive under
    --request-rate on a config with cache_prompt explicitly False). Only trust this flag at
    concurrency==1, closed-loop.
    """
    prefills = _prefill_tps_values(records)
    concurrency_values = {r.get("concurrency") for r in records if not r.get("error")}
    is_closed_loop_serial = concurrency_values == {1} and not any(r.get("load_mode") == "open" for r in records)
    if is_closed_loop_serial and len(prefills) >= 2 and prefills[0] > 0 and prefills[1] > 2 * prefills[0]:
        row.flags.append("cache_suspected")
    if any((r.get("preemptions_delta") or 0) > 0 for r in records):
        row.flags.append("preempted")
    if any("depth_unverified" in (r.get("flags") or []) for r in records):
        row.flags.append("depth_unverified")
    if any("context_shift_risk" in (r.get("flags") or []) for r in records):
        row.flags.append("context_shift_risk")
