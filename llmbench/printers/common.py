"""Shared row model + formatting helpers replicated from llama-bench's markdown_printer
(docs/reference-notes.md §1): size MiB-below-1GiB-else-GiB (binary, 1024^3 threshold),
params M-below-1e9-else-B (decimal threshold -- deliberately inconsistent with size, matched
as-is), test name `pp{n}` / `tg{n}` / `pp{n}+tg{n}` with a leading-space ` @ d{n}` suffix,
`t/s` formatted `"%.2f ± %.2f"`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ..metrics import ResultRow

SCHEMA_VERSION = 1


@dataclass
class PrintRow:
    model: str
    size_bytes: int | None
    n_params: int | None
    backend: str
    ngl: int | None
    src: str  # "client" | "server"
    test: str
    tps_mean: float
    tps_stddev: float
    url: str = ""
    tag: str | None = None
    load_mode: str = "closed"
    request_rate: float | None = None
    concurrency: int = 1
    shared_prefix_n: int = 0
    threads: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    ctk: str | None = None
    ctv: str | None = None
    flash_attn: str | None = None
    n_parallel: int | None = None
    n_ctx: int | None = None
    ttft_ms_mean: float | None = None
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    ttft_ms_p99: float | None = None
    tpot_ms_mean: float | None = None
    itl_ms_mean: float | None = None
    itl_ms_p50: float | None = None
    itl_ms_p95: float | None = None
    itl_p99_ms: float | None = None
    e2e_ms_mean: float | None = None
    e2e_ms_p50: float | None = None
    e2e_ms_p99: float | None = None
    prefill_tps_mean: float | None = None
    decode_tps_mean: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    overhead_ms_mean: float | None = None
    n_prompt_actual: int | None = None
    cached_tokens: int | None = None
    preemptions_delta: int | None = None
    flags: list[str] = field(default_factory=list)
    samples_ts: list[float] = field(default_factory=list)  # per-rep t/s; JSON/JSONL only (llama-bench parity)


def format_size(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "N/A"
    gib = 1024 ** 3
    mib = 1024 ** 2
    if n_bytes < gib:
        return f"{n_bytes / mib:.2f} MiB"
    return f"{n_bytes / gib:.2f} GiB"


def format_params(n: int | None) -> str:
    if n is None:
        return "N/A"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f} M"
    return f"{n / 1_000_000_000:.2f} B"


def format_ts(mean: float, stddev: float) -> str:
    return f"{mean:.2f} ± {stddev:.2f}"


def field_order() -> list[str]:
    """Canonical CSV/JSON/JSONL/SQL field order -- llama-bench's get_fields() extended per
    spec §10 with src, ttft_ms, tpot_ms, itl_p99_ms, overhead_ms, n_prompt_actual,
    cached_tokens, preemptions_delta, load_mode, url, tag -- and further extended with the
    detailed-report stats from spec §9 (median/p95/p99, prefill/decode t/s, throughput) that
    metrics.py already computes but earlier only partially reached any printer."""
    return [
        "model", "size_bytes", "n_params", "backend", "ngl", "threads", "batch", "ubatch",
        "ctk", "ctv", "flash_attn", "n_parallel", "n_ctx", "src", "test",
        "tps_mean", "tps_stddev",
        "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
        "tpot_ms_mean",
        "itl_ms_mean", "itl_ms_p50", "itl_ms_p95", "itl_p99_ms",
        "e2e_ms_mean", "e2e_ms_p50", "e2e_ms_p99",
        "prefill_tps_mean", "decode_tps_mean",
        "request_throughput", "total_token_throughput",
        "overhead_ms_mean", "n_prompt_actual", "cached_tokens", "preemptions_delta",
        "load_mode", "request_rate", "concurrency", "shared_prefix_n", "url", "tag",
    ]


def row_to_dict(row: PrintRow) -> dict[str, Any]:
    d = dataclasses.asdict(row)
    return {k: d[k] for k in field_order() if k in d} | {"flags": d["flags"], "schema": SCHEMA_VERSION}
