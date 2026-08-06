"""Quality divergence check (course_correct.txt §3). Required (refused without it) for any
--parity-mode native comparison -- gate G7.

Fixed prompt set, greedy decoding, max_tokens=128, identical token-ID inputs on both
backends. Reports exact-match rate, first-divergence position distribution, and mean KL
over top-k logprobs if available on both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .backends.base import Backend


@dataclass
class DivergenceResult:
    prompt_idx: int
    token_ids_a: list[int]
    token_ids_b: list[int]
    exact_match: bool
    first_divergence: int | None  # None if exact match; index of first differing token otherwise
    kl_mean: float | None = None


@dataclass
class QualityReport:
    n_prompts: int
    results: list[DivergenceResult] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.exact_match) / len(self.results)

    @property
    def divergence_positions(self) -> list[int]:
        return [r.first_divergence for r in self.results if r.first_divergence is not None]

    @property
    def median_divergence(self) -> float | None:
        pos = sorted(self.divergence_positions)
        if not pos:
            return None
        mid = len(pos) // 2
        if len(pos) % 2:
            return float(pos[mid])
        return (pos[mid - 1] + pos[mid]) / 2.0

    @property
    def mean_kl(self) -> float | None:
        vals = [r.kl_mean for r in self.results if r.kl_mean is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def summary_line(self) -> str:
        pct = f"{self.exact_match_rate * 100:.0f}%"
        med = self.median_divergence
        med_s = f"median divergence @ token {med:.0f}" if med is not None else "no divergence observed"
        return f"quality: {pct} exact match, {med_s}"


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _kl_top_k(logprobs_a: dict[str, float], logprobs_b: dict[str, float]) -> float | None:
    """Mean KL(P_a || P_b) over the union of top-k tokens reported by both sides, renormalised
    over that union. Approximate by construction (only top-k mass is visible); returns None if
    either side has no logprobs to compare."""
    if not logprobs_a or not logprobs_b:
        return None
    keys = set(logprobs_a) | set(logprobs_b)
    floor = -50.0
    pa = {k: math.exp(logprobs_a.get(k, floor)) for k in keys}
    pb = {k: math.exp(logprobs_b.get(k, floor)) for k in keys}
    za, zb = sum(pa.values()), sum(pb.values())
    if za <= 0 or zb <= 0:
        return None
    kl = 0.0
    for k in keys:
        p = pa[k] / za
        q = pb[k] / zb
        if p > 0 and q > 0:
            kl += p * math.log(p / q)
    return kl


async def run_quality_check(
    backend_a: Backend,
    backend_b: Backend,
    prompts: list[list[int]],
    *,
    model_a: str,
    model_b: str,
    max_tokens: int = 128,
) -> QualityReport:
    report = QualityReport(n_prompts=len(prompts))
    for idx, token_ids in enumerate(prompts):
        tokens_a, logprobs_a = await _generate_greedy(backend_a, token_ids, model_a, max_tokens)
        tokens_b, logprobs_b = await _generate_greedy(backend_b, token_ids, model_b, max_tokens)
        first_div = _first_divergence(tokens_a, tokens_b)
        kl = _kl_top_k(logprobs_a, logprobs_b) if (logprobs_a and logprobs_b) else None
        report.results.append(DivergenceResult(
            prompt_idx=idx,
            token_ids_a=tokens_a,
            token_ids_b=tokens_b,
            exact_match=first_div is None,
            first_divergence=first_div,
            kl_mean=kl,
        ))
    return report


async def _generate_greedy(
    backend: Backend, token_ids: list[int], model: str, max_tokens: int
) -> tuple[list[int], dict[str, float]]:
    out_text_parts: list[str] = []
    logprobs: dict[str, float] = {}
    async for chunk in backend.complete_stream(
        token_ids=token_ids, max_tokens=max_tokens, ignore_eos=True, model=model,
        extra={"temperature": 0, "top_p": 1},
    ):
        if chunk.text:
            out_text_parts.append(chunk.text)
        if chunk.raw:
            choices = chunk.raw.get("choices") or []
            if choices and choices[0].get("logprobs"):
                lp = choices[0]["logprobs"]
                top = lp.get("top_logprobs")
                if top:
                    logprobs.update(top[-1] if isinstance(top, list) else top)
    text = "".join(out_text_parts)
    out_ids = await backend.tokenize(text) if text else []
    return out_ids, logprobs
