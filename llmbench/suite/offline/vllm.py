"""Drivers for vLLM's own offline benchmarks: `vllm bench latency` and `vllm bench throughput`.

`vllm bench latency --batch-size B` is the static-batch analogue of llama-batched-bench's
`-npl B`: it submits B requests of `--input-len` tokens each, generates `--output-len` tokens,
and times the whole iteration. It repeats internally via `--num-iters`.

Its `--output-json` file contains per-iteration wall-clock latencies and percentiles, but no
token-rate fields -- vLLM reports the batch's *latency*, not its throughput. The rates below
are therefore derived, and the derivation is exact rather than estimated because both token
counts are pinned by the flags: B x input_len prompt tokens and B x output_len generated
tokens per iteration. What cannot be recovered is the prefill/decode split, since one
iteration's latency covers both -- `pp_tps`/`tg_tps` stay None and only `total_tps` is
reported. llama-batched-bench does report that split, which is one more reason the two
backends' offline rows are not directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..plan import OfflinePlan
from ..spec import CpuSpec
from .base import OfflineResult, failed_result, numactl_wrap, run_tool


def _vllm_env(plan: OfflinePlan) -> dict[str, str]:
    env = dict(plan.backend_spec.env)
    env.setdefault("VLLM_CPU_OMP_THREADS_BIND", plan.cores.physcpubind)
    env.setdefault("VLLM_CPU_KVCACHE_SPACE", "32")
    env.setdefault("OMP_NUM_THREADS", str(plan.cores.n_threads))
    return env


def _bin(plan: OfflinePlan) -> str:
    b = plan.backend_spec.offline_bin
    return str(Path(b).resolve()) if Path(b).exists() else b


# --- vllm bench latency (static batch) ---


def build_latency_argv(plan: OfflinePlan, cpu: CpuSpec, *, output_json: Path) -> list[str]:
    b = plan.backend_spec
    argv = [
        _bin(plan), "bench", "latency",
        "--model", b.model,
        "--input-len", str(plan.n_prompt),
        "--output-len", str(plan.n_gen),
        "--batch-size", str(plan.batch_size),
        "--num-iters", str(plan.reps),
        "--num-iters-warmup", "1",
        "--output-json", str(output_json),
        "--max-model-len", str(plan.n_prompt + plan.n_gen + 16),
        "--max-num-seqs", str(max(plan.batch_size, 1)),
    ]
    argv += list(b.offline_extra_args)
    return numactl_wrap(argv, plan, cpu)


def run_latency(plan: OfflinePlan, cpu: CpuSpec, *, log_dir: Path, timeout_s: float) -> list[OfflineResult]:
    log_dir.mkdir(parents=True, exist_ok=True)
    output_json = log_dir / f"{plan.id}-vllm-latency.json"
    argv = build_latency_argv(plan, cpu, output_json=output_json)
    env = _vllm_env(plan)
    run = run_tool(argv, env_overrides=env, log_path=log_dir / f"{plan.id}-vllm-latency.log",
                   timeout_s=timeout_s)
    if not run.ok:
        return [failed_result(plan, run, "vllm bench latency failed")]
    if not output_json.exists():
        return [failed_result(plan, run, f"vllm bench latency wrote no {output_json.name}")]

    try:
        data = json.loads(output_json.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [failed_result(plan, run, f"could not parse {output_json.name} ({e})")]

    latencies = [float(x) for x in data.get("latencies", []) if x]
    if not latencies:
        avg = data.get("avg_latency")
        latencies = [float(avg)] if avg else []
    if not latencies:
        return [failed_result(plan, run, "vllm bench latency reported no latencies")]

    tokens_per_iter = plan.batch_size * (plan.n_prompt + plan.n_gen)
    gen_per_iter = plan.batch_size * plan.n_gen

    out: list[OfflineResult] = []
    for rep, latency in enumerate(latencies):
        out.append(OfflineResult(
            plan_id=plan.id, backend=plan.backend, tool="vllm-latency",
            test=plan.test_name(), batch_size=plan.batch_size,
            n_prompt=plan.n_prompt, n_gen=plan.n_gen, rep=rep,
            # No prefill/decode split is recoverable from a single batch latency.
            pp_tps=None, tg_tps=None,
            total_tps=(tokens_per_iter / latency) if latency > 0 else None,
            latency_s=latency,
            n_threads=plan.cores.n_threads, cpus=plan.cores.physcpubind,
            argv=argv, env=env, stdout_path=str(run.log_path),
            returncode=run.returncode, duration_s=run.duration_s,
            raw={
                "avg_latency": data.get("avg_latency"),
                "percentiles": data.get("percentiles"),
                "tokens_per_iter": tokens_per_iter,
                "generated_tokens_per_iter": gen_per_iter,
                "derived_gen_tps": (gen_per_iter / latency) if latency > 0 else None,
                "reps_mechanism": "in-process (--num-iters)",
                "output_json": str(output_json),
            },
        ))
    return out


# --- vllm bench throughput (continuous batching) ---


def build_throughput_argv(plan: OfflinePlan, cpu: CpuSpec, *, output_json: Path) -> list[str]:
    b = plan.backend_spec
    argv = [
        _bin(plan), "bench", "throughput",
        "--model", b.model,
        "--dataset-name", "random",
        "--input-len", str(plan.n_prompt),
        "--output-len", str(plan.n_gen),
        "--num-prompts", str(plan.num_prompts or 200),
        "--output-json", str(output_json),
        "--max-model-len", str(plan.n_prompt + plan.n_gen + 16),
    ]
    argv += list(b.offline_extra_args)
    return numactl_wrap(argv, plan, cpu)


def run_throughput(plan: OfflinePlan, cpu: CpuSpec, *, log_dir: Path, timeout_s: float) -> list[OfflineResult]:
    log_dir.mkdir(parents=True, exist_ok=True)
    output_json = log_dir / f"{plan.id}-vllm-throughput.json"
    argv = build_throughput_argv(plan, cpu, output_json=output_json)
    env = _vllm_env(plan)
    run = run_tool(argv, env_overrides=env,
                   log_path=log_dir / f"{plan.id}-vllm-throughput.log", timeout_s=timeout_s)
    if not run.ok:
        return [failed_result(plan, run, "vllm bench throughput failed")]
    if not output_json.exists():
        return [failed_result(plan, run, f"vllm bench throughput wrote no {output_json.name}")]
    try:
        data = json.loads(output_json.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [failed_result(plan, run, f"could not parse {output_json.name} ({e})")]

    return [OfflineResult(
        plan_id=plan.id, backend=plan.backend, tool="vllm-throughput",
        test=f"pp{plan.n_prompt}+tg{plan.n_gen} x{plan.num_prompts}",
        batch_size=0, n_prompt=plan.n_prompt, n_gen=plan.n_gen,
        pp_tps=None,
        tg_tps=_as_float(data.get("output_throughput")),
        total_tps=_as_float(data.get("total_token_throughput")),
        latency_s=_as_float(data.get("elapsed_time")),
        n_threads=plan.cores.n_threads, cpus=plan.cores.physcpubind,
        argv=argv, env=env, stdout_path=str(run.log_path),
        returncode=run.returncode, duration_s=run.duration_s,
        raw={**data, "reps_mechanism": "single run over --num-prompts",
             "output_json": str(output_json)},
    )]


def _as_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "run_latency", "run_throughput", "build_latency_argv", "build_throughput_argv",
]
