"""`llmbench sweep` -- the orchestrated, server-managing benchmark framework.

Subcommands:
  llmbench sweep run     --spec sweep.yaml [--dry-run] [--resume]   plan, launch, measure, report
  llmbench sweep plan    --spec sweep.yaml                          print the resolved plan and exit
  llmbench sweep report  <out-dir> [-f html]                        re-render reports from artifacts
  llmbench sweep cleanup <out-dir> [--dry-run]                      stop servers a killed run left behind
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from . import procs
from .deploy import DeploymentError
from .execute import (
    RUN_INTERRUPTED, STATUS_CAPACITY, STATUS_ERROR, STATUS_OK, ResumeError, SweepRunner,
)
from .lb.nginx import NginxUnavailable
from .objective import rank
from .plan import SweepPlan, build_plan
from .report import FORMATS, ReportContext, load_context, render, write_reports
from .spec import SpecError, SuiteSpec
from .topology import AllocationError, Topology, TopologyError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PARTIAL = 2
EXIT_INTERRUPTED = 130

# Each of these ends the run in an orderly way: live servers torn down, reports written.
# SIGHUP is the one an overnight run actually gets, when the SSH session it was started from
# drops.
_STOP_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")


def _say(*parts) -> None:
    """Print to stderr, and survive a terminal that has gone away.

    After SIGHUP the controlling terminal may be gone, and a write to it raises EIO. That
    used to be fatal to whatever code printed -- including the teardown path, which is
    precisely what has to keep running then. events.jsonl is the durable copy anyway.
    """
    try:
        print(*parts, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


class Progress:
    """Terminal progress. Goes to stderr so stdout stays pipeable."""

    def __init__(self, total: int, *, quiet: bool = False):
        self.total = total
        self.done = 0
        self.initial = 0
        self.quiet = quiet
        self.t0 = time.monotonic()

    def start_from(self, done: int) -> None:
        """A resumed run starts part-way through; ETA is paced on this attempt's work only."""
        self.done = self.initial = done

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "warning":
            # Warnings are shown even with -q: they are what says a number is not to be trusted.
            _say(f"  !! {payload.get('deployment', '')}: {payload['message']}")
            return
        if self.quiet:
            return
        if kind == "deployment_start":
            d = payload["deployment"]
            axes = d["axes"]
            _say(
                f"\n=== {d['id']}: {axes['backend']} × {axes['instances']} inst "
                f"({axes['cores_per_instance']} cores each, np={axes['n_parallel']}, "
                f"lb={axes['lb']}) ==="
            )
            for inst in d["instances"]:
                _say(f"    i{inst['index']} :{inst['port']} cpus={inst['cpus']} "
                     f"ccds={inst['ccds']} membind={inst['membind']}")
        elif kind == "trial_start":
            # Count units of planned work, not emitted rows: one online workload emits a
            # client row and a server row, and one offline plan can emit many rows (a single
            # llama-bench invocation returns both its pp and tg tests). Counting rows made the
            # progress overshoot its own total.
            self.done += 1
            elapsed = time.monotonic() - self.t0
            rate = (self.done - self.initial) / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            _say(f"  -> [{self.done}/{self.total}] {payload['trial_id']} {payload['test']}"
                 f"   (elapsed {elapsed / 60:.1f}m, eta {eta / 60:.1f}m)")
        elif kind == "trial":
            if payload.get("src") in ("client", "native"):
                status = payload["status"]
                metric = payload.get("metrics", {}).get("tps_mean")
                detail = f"{metric:,.2f} t/s" if isinstance(metric, (int, float)) else status
                _say(f"       {detail}")
        elif kind == "resume":
            _say(f"resuming (attempt {payload['previous_attempt'] + 1}): "
                 f"{payload['units_already_done']}/{payload['units_planned']} unit(s) already "
                 f"complete, {payload['rows_superseded']} failed row(s) will be re-measured")


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
        print(f"  {dep.id}  {axes['backend']:20s} inst={axes['instances']} "
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
            print(f"    {o.id}  {o.backend:20s} {o.tool:22s} {o.test_name():24s} "
                  f"cores={o.cores.n_physical_cores} reps={o.reps}")
    if plan.warnings:
        print("\nwarnings:")
        for w in plan.warnings:
            print(f"  ! {w}")
    for dep, _ in plan.groups:
        for w in dep.warnings:
            print(f"  ! [{dep.id}] {w}")
    return EXIT_OK


def _install_stop_handlers(runner: SweepRunner, task: asyncio.Future) -> Callable[[], None]:
    """Turn SIGINT/SIGTERM/SIGHUP into an orderly stop, and return a function undoing it.

    The stop cancels the task driving the run; the cancellation unwinds through every
    `finally` that tears a fleet down, and the caller then writes the reports. Before this,
    SIGTERM and SIGHUP killed the interpreter with no `finally` run at all -- and because the
    servers live in their own sessions, they kept running. SIGINT went through asyncio.run,
    which cancels the main task rather than raising KeyboardInterrupt in it, so the partial
    report that `except KeyboardInterrupt` was meant to write never was.

    A repeated signal does not cancel again: a second cancellation would interrupt the
    teardown the first one started.
    """
    loop = asyncio.get_running_loop()
    installed: list[int] = []

    def stop(name: str) -> None:
        if runner.stop_reason is None:
            runner.request_stop(name)
            _say(f"\n{name} received: tearing down live servers, then writing reports")
            task.cancel()
        else:
            _say(f"{name} received again: still tearing down. If this process must die now, "
                 f"kill it and run `llmbench sweep cleanup {runner.out_dir}`")

    for name in _STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop, name)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass    # no loop signal support (Windows); SIGINT still reaches us via asyncio.run

    def restore() -> None:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    return restore


def _write_final_reports(args, spec: SuiteSpec, plan: SweepPlan,
                         runner: SweepRunner) -> ReportContext | None:
    """Reports for whatever the run produced, however it ended. Never raises: this runs in a
    `finally`, and a rendering bug must not replace the run's real outcome."""
    try:
        ctx = ReportContext(
            manifest=runner.manifest, results=runner.results,
            ranking=rank(runner.results, spec.objective), plan=plan.to_dict(),
        )
        written = write_reports(ctx, spec.out_dir, formats=tuple(args.formats))
    except Exception as e:  # noqa: BLE001
        _say(f"could not write reports ({type(e).__name__}: {e}); every row is in "
             f"{spec.out_dir}/trials.jsonl -- retry with `llmbench sweep report {spec.out_dir}`")
        return None

    _say("\n" + "=" * 72)
    if runner.status != "finished":
        _say(f"run {runner.status}: {runner.error}. Continue it with "
             f"`llmbench sweep run --spec {args.spec} --resume`"
             + (f" --out-dir {args.out_dir}" if args.out_dir else ""))
    _say(ctx.ranking.headline())
    for note in ctx.ranking.notes:
        _say(f"note: {note}")
    _say("=" * 72)
    for name, path in written.items():
        _say(f"  {name:6s} {path}")
    return ctx


async def _run(args) -> int:
    spec = _load_spec(args)
    plan = build_plan(spec, Topology.detect())

    _say(f"sweep '{spec.name}': {plan.n_deployments} deployment(s), {plan.n_trials} trial(s)")
    _say(f"objective: {spec.objective.describe()}")
    _say(f"out-dir  : {spec.out_dir}")
    for w in plan.warnings:
        _say(f"  ! {w}")

    progress = Progress(plan.n_online_trials + plan.n_offline_trials, quiet=args.quiet)
    runner = SweepRunner(plan, on_event=progress, resume=args.resume)
    progress.start_from(runner.n_units_done)

    task = asyncio.ensure_future(runner.run())
    restore = _install_stop_handlers(runner, task)
    ctx = None
    try:
        await task
    except asyncio.CancelledError:
        pass    # an orderly stop: the fleet is already down; report what was measured
    finally:
        restore()
        ctx = _write_final_reports(args, spec, plan, runner)

    if runner.status == RUN_INTERRUPTED:
        return EXIT_INTERRUPTED
    counts = ctx.status_counts() if ctx else {}
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


def cmd_cleanup(args) -> int:
    """Stop the processes a killed run left behind, and only those (see procs.py)."""
    ledger = Path(args.out_dir) / procs.PIDS_FILE
    if not ledger.exists():
        print(f"no {procs.PIDS_FILE} under {args.out_dir}; nothing to clean up")
        return EXIT_OK
    for line in procs.cleanup(ledger, dry_run=args.dry_run):
        print(line)
    if args.dry_run:
        return EXIT_OK
    return EXIT_ERROR if procs.inspect(ledger).leftovers else EXIT_OK


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
    run.add_argument("--resume", action="store_true",
                     help="continue the run recorded in out-dir: keep completed trials, redo "
                          "failed and unfinished ones. Refused if the spec now describes "
                          "different measurements")
    run.add_argument("-f", "--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    run.add_argument("-q", "--quiet", action="store_true", help="progress off; warnings still shown")

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

    cl = sub.add_parser("cleanup", help="stop servers left running by a run that was killed")
    cl.add_argument("out_dir")
    cl.add_argument("--dry-run", action="store_true", help="list what would be stopped")
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
        if args.cmd == "cleanup":
            return cmd_cleanup(args)
        if args.cmd == "run":
            return asyncio.run(_run(args))
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return EXIT_ERROR
    # Configuration problems, not harness bugs: an unsatisfiable core split, a host whose
    # topology cannot be read, a port already taken, a missing nginx, a run that cannot be
    # resumed. Each already carries a message naming the fix, and a traceback on top of it
    # only obscures that.
    except (AllocationError, TopologyError, DeploymentError, NginxUnavailable, ResumeError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    return EXIT_ERROR


__all__ = ["main", "build_parser"]
