"""llmbench console entry point. Default (no subcommand, or `bench`) runs a benchmark
against one endpoint. Subcommands: parity, quality, calibrate, compare.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from . import metrics
from .arrivals import ClosedLoopPool
from .backends.base import Backend
from .backends.llamacpp import LlamaCppBackend
from .backends.vllm import VllmBackend
from .config import CmdParams, GROUP1_DEFAULTS, get_cmd_params_instances, parse_int_range, parse_pg_list, parse_str_list
from .env import capture_env
from .parity import run_parity_check
from .printers.common import PrintRow
from .printers.csv import render_csv
from .printers.json import render_json
from .printers.jsonl import render_jsonl
from .printers.markdown import render_llamabench_table, render_table
from .printers.sql import render_sql
from .prompts import new_run_salt
from .quality import run_quality_check
from .records import JsonlSink, SqliteSink
from .runner import CapacityError, run_instance
from .store import RunStore, compare_runs

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAPACITY_LIMIT = 2


def _add_group1_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", action="append", default=None, help="model id sent to the endpoint")
    p.add_argument("-p", "--n-prompt", default="512", help="prompt tokens (range syntax supported)")
    p.add_argument("-n", "--n-gen", default="128", help="generated tokens (range syntax supported)")
    p.add_argument("-pg", default=None, help="combined test, pp,tg[;pp,tg...]")
    p.add_argument("-d", "--n-depth", default="0", help="KV depth before the measured region")
    p.add_argument("-r", "--repetitions", type=int, default=5, help="reps per instance")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--warmup-fixed", type=int, default=None, help="llama-bench-comparable fixed warmup count")
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("-c", "--concurrency", default="1", help="closed-loop in-flight requests")
    p.add_argument("--request-rate", type=float, default=None)
    p.add_argument("--burstiness", type=float, default=1.0)
    p.add_argument("--shared-prefix", default="0", help="shared prefix tokens (cache test)")
    p.add_argument("-o", "--output", default="md", choices=["csv", "json", "jsonl", "md", "md-llamabench", "sql"])
    p.add_argument("-oe", "--output-err", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--progress", action="store_true")


def _add_group2_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-ngl", default="-1")
    p.add_argument("-fa", "--flash-attn", default="auto")
    p.add_argument("-ctk", default="f16")
    p.add_argument("-ctv", default="f16")
    p.add_argument("-b", "--batch", default="2048")
    p.add_argument("-ub", "--ubatch", default="512")
    p.add_argument("-t", "--threads", default="-1")
    p.add_argument("-np", "--n-parallel", default="-1", help="server slot count / max_num_seqs")
    p.add_argument("--n-ctx", default="0", help="server total context / max_model_len")


def _add_group3_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=float, default=300.0,
                    help="HTTP read timeout in seconds (default 300); raise this for large "
                         "prompts on slow/CPU backends rather than getting silent timeouts")
    p.add_argument("--backend", default="auto", choices=["auto", "llamacpp", "vllm"])
    p.add_argument("--measure", default="both", choices=["client", "server", "both"])
    p.add_argument("--server-mode", default="attach", choices=["attach", "manage"])
    p.add_argument("--server-cmd", default=None, help="manage mode: template, e.g. "
                    "'llama-server -m {model_path} --port {port} -ngl {ngl} -b {batch}'")
    p.add_argument("--server-model-path", default=None, help="manage mode: filesystem path for {model_path}")
    p.add_argument("--endpoint", default="completions", choices=["completions", "chat"])
    p.add_argument("--no-verify-tokens", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--tag", default=None)
    p.add_argument("--parity-mode", default=None, choices=[None, "weights", "native"])
    p.add_argument("--config", default=None, help="bench.yaml; CLI overrides YAML")


def build_bench_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmbench", description="Endpoint-driven inference benchmark harness")
    _add_group1_args(p)
    _add_group2_args(p)
    _add_group3_args(p)
    return p


def _params_from_args(args: argparse.Namespace) -> CmdParams:
    params = CmdParams()
    if args.config:
        from .config import load_yaml_config

        yaml_data = load_yaml_config(args.config)
        for k, v in yaml_data.items():
            if hasattr(params, k):
                setattr(params, k, v)

    if args.model:
        params.model = args.model
    params.n_prompt = parse_int_range(args.n_prompt)
    params.n_gen = parse_int_range(args.n_gen)
    params.pg = parse_pg_list(args.pg) if args.pg else []
    params.n_depth = parse_int_range(args.n_depth)
    params.reps = args.repetitions
    params.no_warmup = args.no_warmup
    params.warmup_fixed = args.warmup_fixed
    params.delay = args.delay
    params.concurrency = parse_int_range(args.concurrency)
    params.request_rate = args.request_rate
    params.burstiness = args.burstiness
    params.shared_prefix = parse_int_range(args.shared_prefix)
    params.output = args.output
    params.output_err = args.output_err
    params.verbose = args.verbose
    params.progress = args.progress

    params.ngl = parse_int_range(args.ngl, allow_negative=True)
    params.flash_attn = parse_str_list(args.flash_attn)
    params.ctk = parse_str_list(args.ctk)
    params.ctv = parse_str_list(args.ctv)
    params.batch = parse_int_range(args.batch)
    params.ubatch = parse_int_range(args.ubatch)
    params.threads = parse_int_range(args.threads, allow_negative=True)
    params.n_parallel = parse_int_range(args.n_parallel, allow_negative=True)
    params.n_ctx = parse_int_range(getattr(args, "n_ctx"))

    params.url = args.url
    params.api_key = args.api_key
    params.timeout = args.timeout
    params.backend = args.backend
    params.measure = args.measure
    params.server_mode = args.server_mode
    params.server_cmd = args.server_cmd
    params.server_model_path = args.server_model_path
    params.endpoint = args.endpoint
    params.no_verify_tokens = args.no_verify_tokens
    params.force = args.force
    params.out_dir = args.out_dir
    params.tag = args.tag
    params.parity_mode = args.parity_mode
    return params


async def _make_backend(url: str, backend_name: str, api_key: str | None, timeout_s: float = 300.0) -> Backend:
    if backend_name == "llamacpp":
        return LlamaCppBackend(url, api_key, timeout_s)
    if backend_name == "vllm":
        return VllmBackend(url, api_key, timeout_s)
    lc = LlamaCppBackend(url, api_key, timeout_s)
    vl = VllmBackend(url, api_key, timeout_s)
    is_lc, is_vl = await lc.detect(), await vl.detect()
    if is_lc and not is_vl:
        await vl.close()
        return lc
    if is_vl and not is_lc:
        await lc.close()
        return vl
    await lc.close()
    await vl.close()
    raise RuntimeError(
        f"--backend auto could not unambiguously detect a backend at {url} "
        f"(llamacpp={is_lc}, vllm={is_vl}); pass --backend explicitly."
    )


async def _run_instances_against(
    backend: Backend, instances: list, params: CmdParams, *, run_id: str, sink, run_salt: int,
    info, print_rows: list[PrintRow],
) -> int:
    exit_code = EXIT_OK
    if not params.model:
        params.model = [info.model or "unknown"]
    for inst in instances:
        if not inst.model:
            inst.model = params.model[0]
        try:
            records = await run_instance(backend, inst, params, run_id=run_id, sink=sink, run_salt=run_salt)
        except CapacityError as e:
            print(f"capacity: {inst.test_name()} did not fit -- {e}", file=sys.stderr)
            exit_code = max(exit_code, EXIT_CAPACITY_LIMIT)
            continue

        raw_dicts = [r.to_dict() for r in records]
        test_name = inst.test_name()

        if params.verbose and records:
            print(f"{test_name}: warmup converged after {records[0].warmup_iterations} iteration(s)", file=sys.stderr)

        if params.measure in ("client", "both"):
            row = metrics.aggregate_client(raw_dicts, test_name)
            print_rows.append(_to_print_row(row, inst, info, params))
        if params.measure in ("server", "both"):
            srow = metrics.aggregate_server(raw_dicts, test_name)
            if srow is not None:
                print_rows.append(_to_print_row(srow, inst, info, params))
    return exit_code


def _group2_key(inst) -> tuple:
    f = inst.group2_flags()
    return tuple(f[k] for k in sorted(f))


async def cmd_bench_manage(params: CmdParams) -> int:
    """--server-mode manage (spec §7): restart the server once per distinct Group-2
    combination, running all of that combination's Group-1 instances against one live
    server -- outer loop matches config.NESTING_ORDER (Group-2 outermost) so restarts are
    minimized, mirroring get_cmd_params_instances()'s own ordering rationale."""
    from urllib.parse import urlparse

    from .server.manage import ServerCapacityFailure, ServerStartupError, launch_server, wait_for_port_release, wait_for_ready

    if not params.server_cmd:
        print("error: --server-mode manage requires --server-cmd", file=sys.stderr)
        return EXIT_ERROR

    parsed = urlparse(params.url)
    port = parsed.port or 8080
    health_path = "/health"

    out_dir = Path(params.out_dir)
    jsonl_sink = JsonlSink(out_dir / f"manage-{new_run_salt()}.jsonl")
    sqlite_sink = SqliteSink(out_dir / "llmbench.db")

    class _MultiSink:
        def write(self, rec):
            jsonl_sink.write(rec)
            sqlite_sink.write(rec)

    sink = _MultiSink()
    run_salt = new_run_salt()
    print_rows: list[PrintRow] = []
    exit_code = EXIT_OK

    instances = get_cmd_params_instances(params)
    groups: dict[tuple, list] = {}
    for inst in instances:
        groups.setdefault(_group2_key(inst), []).append(inst)

    for tag_idx, (_, group_instances) in enumerate(groups.items()):
        flags = dict(group_instances[0].group2_flags())
        flags["port"] = port
        flags["model_path"] = params.server_model_path or ""
        tag = f"combo{tag_idx}"
        server = launch_server(params.server_cmd, flags, out_dir=out_dir, tag=tag)
        try:
            await wait_for_ready(server, health_url=f"{params.url}{health_path}")
        except ServerCapacityFailure as e:
            print(f"capacity: server combo {tag_idx} did not fit -- {e}", file=sys.stderr)
            exit_code = max(exit_code, EXIT_CAPACITY_LIMIT)
            continue
        except ServerStartupError as e:
            print(f"error: server combo {tag_idx} failed to start -- {e}", file=sys.stderr)
            exit_code = max(exit_code, EXIT_ERROR)
            continue

        run_id = f"manage-{tag}-{run_salt}"
        try:
            backend = await _make_backend(params.url, params.backend, params.api_key, params.timeout)
            info = await backend.info()
            exit_code = max(exit_code, await _run_instances_against(
                backend, group_instances, params, run_id=run_id, sink=sink, run_salt=run_salt,
                info=info, print_rows=print_rows,
            ))
            await backend.close()
        finally:
            server.terminate()
            await wait_for_port_release(port)

    jsonl_sink.close()
    sqlite_sink.close()
    _emit(print_rows, params)
    return exit_code


async def cmd_bench(params: CmdParams) -> int:
    params.validate_attach_mode()
    if params.server_mode == "manage":
        return await cmd_bench_manage(params)

    try:
        backend = await _make_backend(params.url, params.backend, params.api_key, params.timeout)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR

    info = await backend.info()
    env_snapshot = capture_env()
    instances = get_cmd_params_instances(params)

    out_dir = Path(params.out_dir)
    run_id = f"{backend.name}-{new_run_salt()}"
    jsonl_sink = JsonlSink(out_dir / f"{run_id}.jsonl")
    sqlite_sink = SqliteSink(out_dir / "llmbench.db")

    class _MultiSink:
        def write(self, rec):
            jsonl_sink.write(rec)
            sqlite_sink.write(rec)

    sink = _MultiSink()
    run_salt = new_run_salt()
    print_rows: list[PrintRow] = []
    exit_code = await _run_instances_against(
        backend, instances, params, run_id=run_id, sink=sink, run_salt=run_salt, info=info, print_rows=print_rows,
    )

    jsonl_sink.close()
    sqlite_sink.close()

    _emit(print_rows, params)
    return exit_code


def _stat_or_none(stats, attr: str) -> float | None:
    """ResultRow's Stats fields default to a zeroed Stats(), not None, when nothing was
    measured (e.g. TTFT on a test with no generation phase) -- surface that as None/N/A
    rather than a misleading 0.0 that looks like a real near-zero measurement."""
    if stats is None or stats.n == 0:
        return None
    return getattr(stats, attr)


def _to_print_row(row: metrics.ResultRow, inst, info, params: CmdParams) -> PrintRow:
    return PrintRow(
        model=inst.model, size_bytes=info.model_size_bytes, n_params=info.model_n_params,
        backend=info.backend, ngl=inst.ngl, src=row.src, test=row.test_name,
        tps_mean=row.tps_mean, tps_stddev=row.tps_stddev, url=params.url, tag=params.tag,
        concurrency=inst.concurrency, shared_prefix_n=inst.shared_prefix,
        threads=inst.threads, batch=inst.batch, ubatch=inst.ubatch, ctk=inst.ctk, ctv=inst.ctv,
        flash_attn=inst.flash_attn, n_parallel=inst.n_parallel, n_ctx=inst.n_ctx,
        ttft_ms_mean=_stat_or_none(row.ttft_ms, "mean"),
        ttft_ms_p50=_stat_or_none(row.ttft_ms, "median"),
        ttft_ms_p95=_stat_or_none(row.ttft_ms, "p95"),
        ttft_ms_p99=_stat_or_none(row.ttft_ms, "p99"),
        tpot_ms_mean=_stat_or_none(row.tpot_ms, "mean"),
        itl_ms_mean=_stat_or_none(row.itl_ms, "mean"),
        itl_ms_p50=_stat_or_none(row.itl_ms, "median"),
        itl_ms_p95=_stat_or_none(row.itl_ms, "p95"),
        itl_p99_ms=_stat_or_none(row.itl_ms, "p99"),
        e2e_ms_mean=_stat_or_none(row.e2e_ms, "mean"),
        e2e_ms_p50=_stat_or_none(row.e2e_ms, "median"),
        e2e_ms_p99=_stat_or_none(row.e2e_ms, "p99"),
        prefill_tps_mean=_stat_or_none(row.prefill_tps, "mean"),
        decode_tps_mean=_stat_or_none(row.decode_tps, "mean"),
        request_throughput=row.request_throughput,
        total_token_throughput=row.total_token_throughput,
        overhead_ms_mean=_stat_or_none(row.overhead_ms, "mean"),
        n_prompt_actual=round(row.n_prompt_actual_mean) if row.n_prompt_actual_mean is not None else None,
        cached_tokens=round(row.cached_tokens_mean) if row.cached_tokens_mean is not None else None,
        preemptions_delta=row.preemptions_delta_total,
        request_rate=params.request_rate, load_mode="open" if params.request_rate else "closed",
        flags=row.flags,
    )


def _emit(rows: list[PrintRow], params: CmdParams) -> None:
    fmt = params.output
    if fmt == "md":
        sys.stdout.write(render_table(rows, defaults=CmdParams()))
    elif fmt == "md-llamabench":
        sys.stdout.write(render_llamabench_table(rows, defaults=CmdParams()))
    elif fmt == "csv":
        sys.stdout.write(render_csv(rows))
    elif fmt == "json":
        sys.stdout.write(render_json(rows))
    elif fmt == "jsonl":
        sys.stdout.write(render_jsonl(rows))
    elif fmt == "sql":
        sys.stdout.write(render_sql(rows))


async def cmd_parity(args: argparse.Namespace) -> int:
    backend_a = await _make_backend(args.a, "auto", None, args.timeout)
    backend_b = await _make_backend(args.b, "auto", None, args.timeout)
    report = await run_parity_check(backend_a, backend_b, requested_sampling={"greedy_both": True})
    report.write(Path(args.out_dir) / "parity.json")
    print(report.headline())
    for a in report.axes:
        print(f"  {a.axis}: {a.verdict} (a={a.value_a}, b={a.value_b})")
    return EXIT_OK if report.all_matched else EXIT_ERROR


async def cmd_quality(args: argparse.Namespace) -> int:
    backend_a = await _make_backend(args.a, "auto", None, args.timeout)
    backend_b = await _make_backend(args.b, "auto", None, args.timeout)
    rng = random.Random(args.seed)
    prompts = [[rng.randrange(32000) for _ in range(args.n_prompt)] for _ in range(args.n_prompts)]
    report = await run_quality_check(
        backend_a, backend_b, prompts, model_a=args.model_a, model_b=args.model_b, max_tokens=args.max_tokens,
    )
    print(report.summary_line())
    if report.mean_kl is not None:
        print(f"mean KL (top-k): {report.mean_kl:.4f}")
    return EXIT_OK


async def cmd_calibrate(args: argparse.Namespace) -> int:
    """Runs llmbench (src=server) against llama-server and reports the delta vs a native
    llama-bench run the caller already captured -- course_correct.txt §5's replacement for
    the old Gate A. This command reports; it does not gate."""
    import json as _json
    import subprocess as _subprocess

    lb_out = _subprocess.run(
        [args.llama_bench_bin, "-m", args.model_path, "-p", str(args.n_prompt), "-n", str(args.n_gen),
         "-r", str(args.reps), "-o", "json"],
        capture_output=True, text=True, timeout=600,
    )
    if lb_out.returncode != 0:
        print(f"llama-bench failed: {lb_out.stderr}", file=sys.stderr)
        return EXIT_ERROR
    native = _json.loads(lb_out.stdout)

    params = CmdParams(
        model=[args.model], n_prompt=[args.n_prompt], n_gen=[args.n_gen], reps=args.reps,
        url=args.url, backend="llamacpp", measure="server", out_dir=args.out_dir,
        concurrency=[1],
    )
    backend = LlamaCppBackend(args.url)
    instances = get_cmd_params_instances(params)
    run_salt = new_run_salt()
    out_dir = Path(args.out_dir)
    sink = JsonlSink(out_dir / "calibrate.jsonl")
    for inst in instances:
        inst.model = args.model
        records = await run_instance(backend, inst, params, run_id="calibrate", sink=sink, run_salt=run_salt)
        raw = [r.to_dict() for r in records]
        srow = metrics.aggregate_server(raw, inst.test_name())
        # Native llama-bench's JSON has no combined "test" string -- match on the same
        # n_prompt/n_gen/n_depth fields it actually emits (docs/reference-notes.md §1).
        native_row = next(
            (r for r in native if r.get("n_prompt") == inst.n_prompt
             and r.get("n_gen") == inst.n_gen and r.get("n_depth") == inst.n_depth),
            None,
        )
        if srow and native_row:
            native_tps = native_row.get("avg_ts")
            if native_tps:
                tax_pct = (native_tps - srow.tps_mean) / native_tps * 100
                print(f"{inst.test_name()}: native={native_tps:.2f} t/s, llmbench(src=server)={srow.tps_mean:.2f} t/s, "
                      f"HTTP+scheduler tax={tax_pct:.1f}%")
    sink.close()
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    store = RunStore(Path(args.out_dir) / "llmbench.db")
    records_a = store.load_run(args.run_a)
    records_b = store.load_run(args.run_b)
    if not records_a or not records_b:
        print("error: one or both run_ids not found in the store", file=sys.stderr)
        return EXIT_ERROR
    findings = compare_runs(records_a, records_b, threshold_pct=args.threshold)
    if not findings:
        print(f"no regressions >= {args.threshold}% between {args.run_a} and {args.run_b}")
        return EXIT_OK
    for f in findings:
        print(f"REGRESSION [{f.src}] {f.test_name} {f.metric}: {f.value_a:.2f} -> {f.value_b:.2f} ({f.pct_change:+.1f}%)")
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    subcommands = {"parity", "quality", "calibrate", "compare", "sweep"}

    if argv and argv[0] in subcommands:
        sub, rest = argv[0], argv[1:]
        if sub == "sweep":
            # The orchestrated path: llmbench manages the servers, their CPU/NUMA placement,
            # multi-instance fleets and load balancing, rather than attaching to one endpoint.
            from .suite.cli import main as sweep_main

            return sweep_main(rest)
        if sub == "parity":
            p = argparse.ArgumentParser(prog="llmbench parity")
            p.add_argument("--a", required=True)
            p.add_argument("--b", required=True)
            p.add_argument("--out-dir", default="out")
            p.add_argument("--timeout", type=float, default=300.0)
            return asyncio.run(cmd_parity(p.parse_args(rest)))
        if sub == "quality":
            p = argparse.ArgumentParser(prog="llmbench quality")
            p.add_argument("--a", required=True)
            p.add_argument("--b", required=True)
            p.add_argument("--model-a", required=True)
            p.add_argument("--model-b", required=True)
            p.add_argument("--n-prompts", type=int, default=8)
            p.add_argument("--n-prompt", type=int, default=64)
            p.add_argument("--max-tokens", type=int, default=128)
            p.add_argument("--timeout", type=float, default=300.0)
            p.add_argument("--seed", type=int, default=42)
            return asyncio.run(cmd_quality(p.parse_args(rest)))
        if sub == "calibrate":
            p = argparse.ArgumentParser(prog="llmbench calibrate")
            p.add_argument("--llama-bench-bin", required=True)
            p.add_argument("--model-path", required=True, help="path passed to native llama-bench -m")
            p.add_argument("--model", required=True, help="model id to send to llama-server")
            p.add_argument("--url", default="http://127.0.0.1:8080")
            p.add_argument("--n-prompt", type=int, default=512)
            p.add_argument("--n-gen", type=int, default=128)
            p.add_argument("--reps", type=int, default=5)
            p.add_argument("--out-dir", default="out")
            return asyncio.run(cmd_calibrate(p.parse_args(rest)))
        if sub == "compare":
            p = argparse.ArgumentParser(prog="llmbench compare")
            p.add_argument("run_a")
            p.add_argument("run_b")
            p.add_argument("--out-dir", default="out")
            p.add_argument("--threshold", type=float, default=5.0)
            return cmd_compare(p.parse_args(rest))

    parser = build_bench_parser()
    args = parser.parse_args(argv)
    params = _params_from_args(args)
    return asyncio.run(cmd_bench(params))


if __name__ == "__main__":
    sys.exit(main())
