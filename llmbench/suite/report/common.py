"""Shared row flattening for every report format.

All three formats render the same table, so the column set and the value formatting live here
once. A metric a trial did not measure renders as empty/`N/A`, never as 0 -- a zero in a
latency column would read as an extraordinary result rather than as missing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..execute import STATUS_OK, TrialResult

# (column header, source, key). source: "top" = TrialResult attr, "axis", "metric".
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("status", "top", "status"),
    ("kind", "top", "kind"),
    ("src", "top", "src"),
    ("backend", "top", "backend"),
    ("model", "top", "model"),
    ("test", "top", "test"),
    ("instances", "axis", "instances"),
    ("cores/inst", "axis", "cores_per_instance"),
    ("threads/inst", "axis", "threads_per_instance"),
    ("np", "axis", "n_parallel"),
    ("n_ctx", "axis", "n_ctx"),
    ("batch", "axis", "batch"),
    ("ubatch", "axis", "ubatch"),
    ("lb", "axis", "lb"),
    ("tool", "axis", "tool"),
    ("batch_size", "axis", "batch_size"),
    ("conc", "axis", "concurrency"),
    ("rate", "axis", "request_rate"),
    ("t/s", "metric", "tps_mean"),
    ("t/s sd", "metric", "tps_stddev"),
    ("tok tput", "metric", "total_token_throughput"),
    ("req tput", "metric", "request_throughput"),
    ("ttft p50", "metric", "ttft_ms_p50"),
    ("ttft p99", "metric", "ttft_ms_p99"),
    ("itl mean", "metric", "itl_ms_mean"),
    ("itl p95", "metric", "itl_ms_p95"),
    ("tok/chunk", "metric", "tokens_per_chunk"),
    ("tpot mean", "metric", "tpot_ms_mean"),
    ("e2e p50", "metric", "e2e_ms_mean"),
    ("overhead ms", "metric", "overhead_ms_mean"),
    ("reps", "metric", "n_reps_valid"),
    ("trial", "top", "trial_id"),
    ("error", "top", "error"),
)

# Columns whose values are latency-like and therefore better when smaller. Used to colour
# the HTML heatmap in the right direction.
LOWER_IS_BETTER = {
    "ttft p50", "ttft p99", "itl mean", "itl p95", "tpot mean", "e2e p50", "overhead ms",
}


def cell(result: TrialResult, source: str, key: str) -> Any:
    if source == "top":
        return getattr(result, key, None)
    if source == "axis":
        return result.axes.get(key)
    return result.metrics.get(key)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def used_columns(results: Iterable[TrialResult]) -> list[tuple[str, str, str]]:
    """Drop columns that are empty for every row, so a llama.cpp-only run does not carry a
    column of blank vLLM fields (and vice versa)."""
    results = list(results)
    keep = []
    for col in COLUMNS:
        header, source, key = col
        if header in ("status", "backend", "test", "trial"):
            keep.append(col)
            continue
        if any(cell(r, source, key) not in (None, "", []) for r in results):
            keep.append(col)
    return keep


def rows_for(results: list[TrialResult], columns: list[tuple[str, str, str]]) -> list[list[str]]:
    return [[fmt(cell(r, s, k)) for _, s, k in columns] for r in results]


@dataclass
class ReportContext:
    """Everything the renderers need, assembled once."""

    manifest: dict[str, Any]
    results: list[TrialResult]
    ranking: Any                    # objective.RankingReport
    plan: dict[str, Any]

    @property
    def ok_results(self) -> list[TrialResult]:
        return [r for r in self.results if r.status == STATUS_OK]

    @property
    def online(self) -> list[TrialResult]:
        return [r for r in self.results if r.kind == "online"]

    @property
    def offline(self) -> list[TrialResult]:
        return [r for r in self.results if r.kind == "offline"]

    @property
    def failures(self) -> list[TrialResult]:
        return [r for r in self.results if r.status not in (STATUS_OK,)]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def all_warnings(self) -> list[str]:
        seen: dict[str, None] = {}
        for w in self.plan.get("warnings", []):
            seen.setdefault(str(w), None)
        for r in self.results:
            for w in r.warnings:
                seen.setdefault(str(w), None)
        return list(seen)


__all__ = ["COLUMNS", "LOWER_IS_BETTER", "ReportContext", "cell", "fmt", "used_columns", "rows_for"]
