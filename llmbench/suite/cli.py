"""`llmbench sweep` -- the orchestrated, server-managing benchmark framework.

Subcommands:
  llmbench sweep run    --spec sweep.yaml [--dry-run]   plan, launch, measure, report
  llmbench sweep plan   --spec sweep.yaml               print the resolved plan and exit
  llmbench sweep report <out-dir> [-f html]             re-render reports from artifacts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .deploy import DeploymentError
from .execute import STATUS_CAPACITY, STATUS_ERROR, STATUS_OK, SweepRunner
from .lb.nginx import NginxUnavailable
from .objective import rank
from .plan import build_plan
from .report import FORMATS, ReportContext, load_context, render, write_reports
from .spec import SpecError, SuiteSpec
from .topology import AllocationError, Topology, TopologyError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PARTIAL = 2


class Progress:
    """Terminal progress. Goes to stderr so stdout stays pipeable."""

    def __init__(self, total: int, *, quiet: bool = False):
        self.total = total
        self.done = 0
        self.quiet = quiet
        self.t0 = time.monotonic()

    def __call__(self, kind: str, payload: dict) -> None:
        if self.quiet:
            return
        if kind == "deployment_start":
            d = payload["deployment"]
            axes = d["axes"]
            print(
                f"\n=== {d['id']}: {axes['backend']} × {axes['instances']} inst "
                f"({axes['cores_per_instance']} cores each, np={axes['n_parallel']}, "
                f"lb={axes['lb']}) ===",
                file=sys.stderr, flush=True,
            )
            for inst in d["instances"]:
                print(f"    i{inst['index']} :{inst['port']} cpus={inst['cpus']} "
                      f"ccds={inst['ccds']} membind={inst['membind']}",
                      file=sys.stderr, flush=True)
        elif kind == "trial_start":
            # Count units of planned work, not emitted rows: one online workload emits a
            # client row and a server row, and one offline plan can emit many rows (a single
            # llama-bench invocation returns both its pp and tg tests). Counting rows made the
            # progress overshoot its own total.
            self.done += 1
            elapsed = time.monotonic() - self.t0
            rate = self.done / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            print(f"  -> [{self.done}/{self.total}] {payload['trial_id']} {payload['test']}"
                  f"   (elapsed {elapsed / 60:.1f}m, eta {eta / 60:.1f}m)",
                  file=sys.stderr, flush=True)
        elif kind == "trial":
            if payload.get("src") in ("client", "native"):
                status = payload["status"]
                metric = payload.get("metrics", {}).get("tps_mean")
                detail = f"{metric:,.2f} t/s" if isinstance(metric, (int, float)) else status
                print(f"       {detail}", file=sys.stderr, flush=True)
        elif kind == "warning":
            print(f"  !! {payload.get('deployment', '')}: {payload['message']}",
                  file=sys.stderr, flush=True)


def _load_spec(args) -> SuiteSpec:
    spec = SuiteSpec.from_yaml(args.spec)
    if getattr(args, "out_dir", None):
        spec.out_dir = args.out_dir
    if getattr(args, "dry_run", False):
        spec.dry_run = True
    if getattr(args, "mode", None):
        spec.mode = args.mode
    if getattr(args, "name", None):
        spec.name = args.name
    return spec


def cmd_plan(args) -> int:
    spec = _load_spec(args)
    plan = build_plan(spec, Topology.detect())
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, default=str))
        return EXIT_OK

    print(f"sweep: {spec.name}    mode={spec.mode}")
    print(f"objective: {spec.objective.describe()}")
    print(f"budget: {spec.cpu.budget}  smt={spec.cpu.smt}  ccd_align={spec.cpu.ccd_align}")
    print(f"\n{plan.n_deployments} deployment(s), {plan.n_online_trials} online trial(s), "
          f"{plan.n_offline_trials} offline trial(s) -- {plan.n_trials} rows\n")
    for dep, workloads in plan.groups:
        axes = dep.axes()
        print(f"  {dep.id}  {axes['backend']:9s} inst={axes['instances']} "
              f"cores/inst={axes['cores_per_instance']:3d} np={axes['n_parallel']:3d} "
              f"ctx={axes['n_ctx']} lb={axes['lb']:6s} -> {dep.client_url}")
        for inst in dep.instances:
            print(f"        i{inst.index} :{inst.port} cpus={inst.cores.physcpubind:12s} "
                  f"ccds={inst.cores.ccds} membind={inst.cores.membind}")
        if dep.lb_placement:
            print(f"        nginx cpus={dep.lb_placement.physcpubind} "
                  f"({dep.lb_placement.source})")
        print(f"        {len(workloads)} workload(s): "
              f"{', '.join(w.test_name() + '/c' + str(w.concurrency) for w in workloads[:8])}"
              f"{' ...' if len(workloads) > 8 else ''}")
    if plan.offline:
        print(f"\n  offline ({len(plan.offline)}):")
        for o in plan.offline:
            print(f"    {o.id}  {o.backend:9s} {o.tool:22s} {o.test_name():24s} "
                  f"cores={o.cores.n_physical_cores} reps={o.reps}")
    if plan.warnings:
        print("\nwarnings:")
        for w in plan.warnings:
            print(f"  ! {w}")
    for dep, _ in plan.groups:
        for w in dep.warnings:
            print(f"  ! [{dep.id}] {w}")
    return EXIT_OK


async def _run(args) -> int:
    spec = _load_spec(args)
    plan = build_plan(spec, Topology.detect())

    print(f"sweep '{spec.name}': {plan.n_deployments} deployment(s), {plan.n_trials} trial(s)",
          file=sys.stderr)
    print(f"objective: {spec.objective.describe()}", file=sys.stderr)
    print(f"out-dir  : {spec.out_dir}", file=sys.stderr)
    for w in plan.warnings:
        print(f"  ! {w}", file=sys.stderr)

    progress = Progress(plan.n_online_trials + plan.n_offline_trials, quiet=args.quiet)
    runner = SweepRunner(plan, on_event=progress)
    try:
        results = await runner.run()
    except KeyboardInterrupt:
        print("\ninterrupted -- partial results are on disk; "
              f"re-render with `llmbench sweep report {spec.out_dir}`", file=sys.stderr)
        results = runner.results

    manifest = json.loads((Path(spec.out_dir) / "run.json").read_text())
    ctx = ReportContext(
        manifest=manifest, results=results,
        ranking=rank(results, spec.objective), plan=plan.to_dict(),
    )
    written = write_reports(ctx, spec.out_dir, formats=tuple(args.formats))

    print("\n" + "=" * 72, file=sys.stderr)
    print(ctx.ranking.headline(), file=sys.stderr)
    for note in ctx.ranking.notes:
        print(f"note: {note}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for name, path in written.items():
        print(f"  {name:6s} {path}", file=sys.stderr)

    counts = ctx.status_counts()
    if counts.get(STATUS_OK, 0) == 0:
        return EXIT_ERROR
    if counts.get(STATUS_ERROR, 0) or counts.get(STATUS_CAPACITY, 0):
        return EXIT_PARTIAL
    return EXIT_OK


def cmd_report(args) -> int:
    ctx = load_context(args.out_dir, from_records=args.from_records)
    if args.stdout:
        sys.stdout.write(render(ctx, args.format))
        return EXIT_OK
    written = write_reports(ctx, args.out_dir, formats=tuple(args.formats))
    print(ctx.ranking.headline())
    for name, path in written.items():
        print(f"  {name:6s} {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmbench sweep",
        description="Orchestrated benchmark sweeps: manages servers, CPU/NUMA placement, "
                    "multi-instance fleets and load balancing, then reports the best config.",
    )
    sub = p.add_subparsers(dest="cmd")

    def add_spec_args(sp):
        sp.add_argument("--spec", required=True, help="sweep YAML (see sweep.example.yaml)")
        sp.add_argument("--out-dir", default=None, help="override spec.out_dir")
        sp.add_argument("--mode", default=None, choices=["online", "offline", "both"])
        sp.add_argument("--name", default=None, help="override spec.name")

    run = sub.add_parser("run", help="execute the sweep")
    add_spec_args(run)
    run.add_argument("--dry-run", action="store_true",
                     help="resolve and record the plan without launching anything")
    run.add_argument("-f", "--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    run.add_argument("-q", "--quiet", action="store_true")

    pl = sub.add_parser("plan", help="print the resolved plan and exit")
    add_spec_args(pl)
    pl.add_argument("--json", action="store_true")

    rep = sub.add_parser("report", help="re-render reports from a finished run directory")
    rep.add_argument("out_dir")
    rep.add_argument("-f", "--format", default="md", choices=list(FORMATS),
                     help="format for --stdout")
    rep.add_argument("--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS),
                     help="formats to write to disk")
    rep.add_argument("--stdout", action="store_true", help="print --format instead of writing")
    rep.add_argument("--from-records", action="store_true",
                     help="recompute every statistic from records/ instead of replaying the "
                          "values stored in trials.jsonl -- use after changing metrics.py")
    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        if args.cmd == "plan":
            return cmd_plan(args)
        if args.cmd == "report":
            return cmd_report(args)
        if args.cmd == "run":
            return asyncio.run(_run(args))
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return EXIT_ERROR
    # Configuration problems, not harness bugs: an unsatisfiable core split, a host whose
    # topology cannot be read, a port already taken, a missing nginx. Each already carries a
    # message naming the fix, and a traceback on top of it only obscures that.
    except (AllocationError, TopologyError, DeploymentError, NginxUnavailable) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


__all__ = ["main", "build_parser"]
