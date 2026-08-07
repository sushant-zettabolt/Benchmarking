"""Native offline benchmark drivers, dispatched by tool name.

See base.py for why these rows are marked not-comparable across backends.
"""
from __future__ import annotations

from pathlib import Path

from ..plan import OfflinePlan
from ..spec import CpuSpec
from .base import OfflineResult
from .llamacpp import run_batched_bench, run_llama_bench
from .vllm import run_latency, run_throughput

_DISPATCH = {
    "llama-bench": run_llama_bench,
    "llama-batched-bench": run_batched_bench,
    "vllm-latency": run_latency,
    "vllm-throughput": run_throughput,
}

SUPPORTED_TOOLS = tuple(_DISPATCH)


def run_offline(
    plan: OfflinePlan, cpu: CpuSpec, *, log_dir: Path, timeout_s: float,
) -> list[OfflineResult]:
    driver = _DISPATCH.get(plan.tool)
    if driver is None:
        raise ValueError(f"unknown offline tool {plan.tool!r}; known: {list(SUPPORTED_TOOLS)}")
    return driver(plan, cpu, log_dir=log_dir, timeout_s=timeout_s)


__all__ = ["run_offline", "OfflineResult", "SUPPORTED_TOOLS"]
