"""The parity axis table: probe both backends, classify, refuse. course_correct.txt §2/§5/§7.

"Nothing downstream is trustworthy until this exists" -- run before any cross-backend
comparison and write parity.json next to the results. Gate G2: deliberately mismatch each
axis one at a time; the tool must detect and refuse every one.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends.base import Backend, Capacity, ServerInfo

Verdict = str  # "matched" | "mismatched" | "unverifiable"


@dataclass
class AxisResult:
    axis: str
    verdict: Verdict
    value_a: Any
    value_b: Any
    detail: str = ""


@dataclass
class ParityReport:
    url_a: str
    url_b: str
    axes: list[AxisResult] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(a.verdict == "matched" for a in self.axes)

    @property
    def mismatched_axes(self) -> list[str]:
        return [a.axis for a in self.axes if a.verdict == "mismatched"]

    @property
    def unverifiable_axes(self) -> list[str]:
        return [a.axis for a in self.axes if a.verdict == "unverifiable"]

    def to_dict(self) -> dict:
        return {
            "url_a": self.url_a,
            "url_b": self.url_b,
            "all_matched": self.all_matched,
            "axes": [dataclasses.asdict(a) for a in self.axes],
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def headline(self) -> str:
        if self.all_matched:
            return "PARITY OK -- all axes matched."
        bad = self.mismatched_axes
        unk = self.unverifiable_axes
        parts = []
        if bad:
            parts.append(f"mismatched: {', '.join(bad)}")
        if unk:
            parts.append(f"unverifiable: {', '.join(unk)}")
        return "Not a valid comparison. " + "; ".join(parts) + (
            ". Per-backend results below are individually valid and jointly meaningless."
        )


def _classify(value_a, value_b, *, tolerant_ratio: float | None = None) -> Verdict:
    if value_a is None or value_b is None:
        return "unverifiable"
    if tolerant_ratio is not None and isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
        if value_a == 0 or value_b == 0:
            return "matched" if value_a == value_b else "mismatched"
        ratio = max(value_a, value_b) / min(value_a, value_b)
        return "matched" if ratio <= (1 + tolerant_ratio) else "mismatched"
    return "matched" if value_a == value_b else "mismatched"


async def run_parity_check(
    backend_a: Backend, backend_b: Backend, *, requested_sampling: dict[str, Any] | None = None
) -> ParityReport:
    info_a, info_b = await backend_a.info(), await backend_b.info()
    cap_a, cap_b = await backend_a.capacity(), await backend_b.capacity()

    report = ParityReport(url_a=backend_a.base_url, url_b=backend_b.base_url)

    # 2.1 Weights and numerics -- no clean HTTP-probeable answer (see course_correct.txt §3);
    # best we can do without out-of-band agreement is compare declared model identity.
    report.axes.append(AxisResult(
        axis="weights_numerics",
        verdict="unverifiable",
        value_a=info_a.model, value_b=info_b.model,
        detail="Model identity cannot be verified as byte-identical weights over HTTP. "
               "Use --parity-mode to declare the intended relationship explicitly.",
    ))

    # 2.2 KV cache dtype
    report.axes.append(AxisResult(
        axis="kv_cache_dtype",
        verdict=_classify(cap_a.kv_dtype, cap_b.kv_dtype),
        value_a=cap_a.kv_dtype, value_b=cap_b.kv_dtype,
    ))

    # 2.3 Context capacity -- compare the derived per-request budget, not raw flags
    report.axes.append(AxisResult(
        axis="context_capacity_per_request",
        verdict=_classify(cap_a.per_request_ctx, cap_b.per_request_ctx),
        value_a=cap_a.per_request_ctx, value_b=cap_b.per_request_ctx,
    ))

    # 2.4 KV memory budget
    report.axes.append(AxisResult(
        axis="kv_memory_budget_bytes",
        verdict=_classify(cap_a.kv_bytes, cap_b.kv_bytes, tolerant_ratio=0.0),
        value_a=cap_a.kv_bytes, value_b=cap_b.kv_bytes,
        detail="ratio=" + (
            f"{max(cap_a.kv_bytes, cap_b.kv_bytes) / min(cap_a.kv_bytes, cap_b.kv_bytes):.2f}"
            if cap_a.kv_bytes and cap_b.kv_bytes else "n/a"
        ),
    ))

    # 2.5 Batch/admission shaping
    report.axes.append(AxisResult(
        axis="batch_admission_shaping",
        verdict=_classify(cap_a.batch_token_budget, cap_b.batch_token_budget),
        value_a=cap_a.batch_token_budget, value_b=cap_b.batch_token_budget,
    ))
    report.axes.append(AxisResult(
        axis="chunked_prefill",
        verdict=_classify(cap_a.chunked_prefill, cap_b.chunked_prefill),
        value_a=cap_a.chunked_prefill, value_b=cap_b.chunked_prefill,
        detail="llama.cpp's chunked prefill is implicit and cannot be disabled -- "
               "a 'mismatch' here may be structural, not a config error.",
    ))

    # 2.6 Attention backend -- record actual selection, not requested flag
    report.axes.append(AxisResult(
        axis="attention_backend",
        verdict=_classify(cap_a.attn_backend, cap_b.attn_backend),
        value_a=cap_a.attn_backend, value_b=cap_b.attn_backend,
    ))

    # 2.7 Prefix-cache state -- effective, as observed
    report.axes.append(AxisResult(
        axis="prefix_cache_enabled",
        verdict=_classify(cap_a.prefix_cache_enabled, cap_b.prefix_cache_enabled),
        value_a=cap_a.prefix_cache_enabled, value_b=cap_b.prefix_cache_enabled,
    ))

    # 2.8 Sampling -- both sides are commanded by us (runner enforces greedy), so this is
    # "requested" not "observed"; downgrade to unverifiable unless the backend echoes it back.
    report.axes.append(AxisResult(
        axis="sampling_config",
        verdict="matched" if requested_sampling and requested_sampling.get("greedy_both") else "unverifiable",
        value_a=requested_sampling, value_b=requested_sampling,
        detail="Sampling params are commanded identically by the runner; not independently observable "
               "from either backend's HTTP surface, so this reflects the request, not a probe.",
    ))

    # 2.9 Warm state -- filled in by runner.py after convergence warmup completes (see
    # runner.warmup_convergence); left unverifiable at parity-preflight time, before any
    # instance has run.
    report.axes.append(AxisResult(
        axis="warm_state",
        verdict="unverifiable",
        value_a=None, value_b=None,
        detail="Determined per-instance by convergence warmup (docs/contract.md); "
               "not knowable before any instance has run.",
    ))

    return report


def apply_deliberate_mismatch(report: ParityReport, axis: str, value_a: Any, value_b: Any) -> ParityReport:
    """Test/debug hook for gate G2: force one axis to a known mismatched pair and reclassify."""
    for a in report.axes:
        if a.axis == axis:
            a.value_a, a.value_b = value_a, value_b
            a.verdict = _classify(value_a, value_b)
    return report
