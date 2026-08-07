"""Drivers for llama.cpp's own offline benchmarks: llama-bench and llama-batched-bench.

Tool choice is made in plan.py: batch_size == 1 goes to `llama-bench` (the canonical,
most-cited llama.cpp number), anything larger must go to `llama-batched-bench`, because
llama-bench drives a single sequence and has no request-batch concept at all -- its `-b` is
the *token* batch fed to one decode call, which is a different quantity entirely and is
already swept as a deployment axis.

Repetition semantics differ between the two, and that changes what a stddev means:

  llama-bench takes `-r N` and repeats internally, with the model loaded once. The spread it
  reports is within-process run-to-run variance.

  llama-batched-bench has no repetition flag, so N reps means N process invocations, each
  reloading the model. Its spread therefore also contains process-start and page-cache
  effects. Rows record which via `raw["reps_mechanism"]`.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..plan import OfflinePlan
from ..spec import CpuSpec
from .base import OfflineResult, failed_result, numactl_wrap, run_tool


def _common_args(plan: OfflinePlan) -> list[str]:
    threads = plan.cores.n_threads
    return ["-t", str(threads), "--numa", "numactl"]


# --- llama-bench ---


def build_llama_bench_argv(plan: OfflinePlan, cpu: CpuSpec) -> list[str]:
    b = plan.backend_spec
    argv = [
        str(Path(b.offline_bin).resolve()) if Path(b.offline_bin).exists() else b.offline_bin,
        "-m", b.model,
        "-p", str(plan.n_prompt),
        "-n", str(plan.n_gen),
        "-r", str(plan.reps),
        "-o", "json",
        *_common_args(plan),
    ]
    argv += list(b.offline_extra_args)
    return numactl_wrap(argv, plan, cpu)


def parse_llama_bench(plan: OfflinePlan, stdout: str) -> list[dict]:
    """llama-bench -o json prints a JSON array; ignore any leading log noise."""
    start = stdout.find("[")
    if start < 0:
        raise ValueError("no JSON array found in llama-bench output")
    return json.loads(stdout[start:])


def run_llama_bench(plan: OfflinePlan, cpu: CpuSpec, *, log_dir: Path, timeout_s: float) -> list[OfflineResult]:
    argv = build_llama_bench_argv(plan, cpu)
    run = run_tool(argv, env_overrides={}, log_path=log_dir / f"{plan.id}-llama-bench.log",
                   timeout_s=timeout_s)
    if not run.ok:
        return [failed_result(plan, run, "llama-bench failed")]
    try:
        rows = parse_llama_bench(plan, run.stdout)
    except (ValueError, json.JSONDecodeError) as e:
        return [failed_result(plan, run, f"could not parse llama-bench output ({e})")]

    out: list[OfflineResult] = []
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        tps = _as_float(row.get("avg_ts"))
        # llama-bench emits one row per test, and a test is pp-only or tg-only. Attribute its
        # single t/s to the phase that row actually measured rather than inventing a split.
        out.append(OfflineResult(
            plan_id=plan.id, backend=plan.backend, tool="llama-bench",
            test=f"pp{n_prompt}" if n_gen == 0 else f"tg{n_gen}",
            batch_size=1, n_prompt=n_prompt, n_gen=n_gen,
            pp_tps=tps if n_gen == 0 else None,
            tg_tps=tps if n_gen > 0 else None,
            total_tps=tps,
            latency_s=(_as_float(row.get("avg_ns")) or 0) / 1e9 or None,
            n_threads=_as_int(row.get("n_threads")),
            cpus=plan.cores.physcpubind, argv=argv, env={},
            stdout_path=str(run.log_path), returncode=run.returncode,
            duration_s=run.duration_s,
            raw={
                **row,
                "reps_mechanism": "in-process (-r)",
                "stddev_ts": row.get("stddev_ts"),
            },
        ))
    return out


# --- llama-batched-bench ---


def build_batched_bench_argv(plan: OfflinePlan, cpu: CpuSpec) -> list[str]:
    b = plan.backend_spec
    # n_ctx must hold the whole batch: every one of the `pl` sequences needs pp+tg tokens
    # (unless the prompt is shared, in which case the prompt is stored once).
    n_ctx = (plan.n_prompt if plan.shared_prompt else plan.batch_size * plan.n_prompt) \
        + plan.batch_size * plan.n_gen
    argv = [
        str(Path(b.batched_bin).resolve()) if Path(b.batched_bin).exists() else b.batched_bin,
        "-m", b.model,
        "-npp", str(plan.n_prompt),
        "-ntg", str(plan.n_gen),
        "-npl", str(plan.batch_size),
        "-c", str(n_ctx),
        "--output-format", "jsonl",
        *_common_args(plan),
    ]
    if plan.shared_prompt:
        argv.append("-pps")
    argv += list(b.offline_extra_args)
    return numactl_wrap(argv, plan, cpu)


def parse_batched_bench(stdout: str) -> list[dict]:
    """One JSON object per line; the tool also logs plain text, so filter to parseable lines."""
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "speed" in obj and "pl" in obj:
            rows.append(obj)
    return rows


def run_batched_bench(plan: OfflinePlan, cpu: CpuSpec, *, log_dir: Path, timeout_s: float) -> list[OfflineResult]:
    argv = build_batched_bench_argv(plan, cpu)
    out: list[OfflineResult] = []
    # No -r flag: repetition is process-level, so each rep reloads the model.
    for rep in range(plan.reps):
        run = run_tool(argv, env_overrides={},
                       log_path=log_dir / f"{plan.id}-batched-bench-r{rep}.log",
                       timeout_s=timeout_s)
        if not run.ok:
            res = failed_result(plan, run, "llama-batched-bench failed")
            res.rep = rep
            out.append(res)
            continue
        rows = parse_batched_bench(run.stdout)
        if not rows:
            res = failed_result(plan, run, "llama-batched-bench produced no parseable jsonl rows")
            res.rep = rep
            out.append(res)
            continue
        for row in rows:
            out.append(OfflineResult(
                plan_id=plan.id, backend=plan.backend, tool="llama-batched-bench",
                test=plan.test_name(), batch_size=int(row.get("pl", plan.batch_size)),
                n_prompt=int(row.get("pp", plan.n_prompt)), n_gen=int(row.get("tg", plan.n_gen)),
                rep=rep,
                pp_tps=_as_float(row.get("speed_pp")),
                tg_tps=_as_float(row.get("speed_tg")),
                total_tps=_as_float(row.get("speed")),
                latency_s=_as_float(row.get("t")),
                n_threads=_as_int(row.get("n_threads")),
                cpus=plan.cores.physcpubind, argv=argv, env={},
                stdout_path=str(run.log_path), returncode=run.returncode,
                duration_s=run.duration_s,
                raw={**row, "reps_mechanism": "process-level (re-invocation)"},
            ))
    return out


def _as_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "run_llama_bench", "run_batched_bench",
    "build_llama_bench_argv", "build_batched_bench_argv",
    "parse_llama_bench", "parse_batched_bench",
]
