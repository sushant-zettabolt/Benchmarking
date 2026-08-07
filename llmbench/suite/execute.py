"""Execute a SweepPlan: bring fleets up, drive workloads, run offline tools, persist everything.

Measurement itself is delegated, deliberately. Online trials go through the existing
`runner.run_instance` + `metrics.aggregate_*` path -- the same code the single-endpoint CLI
uses -- so a sweep row and a one-off `llmbench` row are computed by identical arithmetic and
`tests/test_metrics_no_backend_branch.py` still mechanically guarantees no backend-specific
branching. This module owns orchestration and persistence, not statistics.

Raw records are always written before anything is aggregated, so a sweep that dies at trial
90 of 126 leaves the first 89 trials fully re-analysable from disk.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import metrics
from ..config import CmdParams, Instance
from ..env import capture_env
from ..prompts import new_run_salt
from ..records import JsonlSink, SqliteSink
from ..runner import CapacityError, run_instance
from .contention import ContentionMonitor
from .deploy import CapacityFailure, DeploymentError, launch_deployment
from .lb import make_backend
from .lb.fanout import FanoutBackend
from .offline import run_offline
from .plan import DeploymentPlan, OfflinePlan, SweepPlan, WorkloadPlan
from .spec import SuiteSpec

STATUS_OK = "ok"
STATUS_CAPACITY = "capacity"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

# Foreign CPU load above this share of a deployment's allocated cpus marks a trial as
# contaminated. 15% is roughly where a measurement stops being reproducible on this class of
# machine; below it, normal system noise dominates.
CONTENTION_WARN_PCT = 15.0


@dataclass
class TrialResult:
    """One measured row. `metrics` is flat and JSON-safe so reports can consume it directly."""

    trial_id: str
    kind: str                     # online | offline
    status: str
    run_id: str = ""              # which sweep produced this row (see _rotate_stale_artifacts)
    src: str = "client"
    backend: str = ""
    model: str = ""
    test: str = ""
    deployment_id: str = ""
    workload_id: str = ""
    axes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    comparable_across_backends: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    started_at_utc: str = ""
    duration_s: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _stat(stats, attr: str) -> float | None:
    """Stats defaults to a zeroed instance when nothing was measured; surface that as None
    rather than a 0.0 that reads like a real near-zero measurement."""
    if stats is None or getattr(stats, "n", 0) == 0:
        return None
    return getattr(stats, attr, None)


def result_row_to_metrics(row: metrics.ResultRow) -> dict[str, Any]:
    """Flatten a ResultRow into the metric namespace objectives and reports address by name."""
    return {
        "tps_mean": row.tps_mean,
        "tps_stddev": row.tps_stddev,
        "n_reps": row.n_reps,
        "n_reps_valid": row.n_reps_valid,
        "ttft_ms_mean": _stat(row.ttft_ms, "mean"),
        "ttft_ms_p50": _stat(row.ttft_ms, "median"),
        "ttft_ms_p95": _stat(row.ttft_ms, "p95"),
        "ttft_ms_p99": _stat(row.ttft_ms, "p99"),
        "tpot_ms_mean": _stat(row.tpot_ms, "mean"),
        "itl_ms_mean": _stat(row.itl_ms, "mean"),
        "itl_ms_p50": _stat(row.itl_ms, "median"),
        "itl_ms_p95": _stat(row.itl_ms, "p95"),
        "itl_ms_p99": _stat(row.itl_ms, "p99"),
        "tokens_per_chunk": row.tokens_per_chunk,
        "e2e_ms_mean": _stat(row.e2e_ms, "mean"),
        "e2e_ms_p50": _stat(row.e2e_ms, "median"),
        "e2e_ms_p99": _stat(row.e2e_ms, "p99"),
        "prefill_tps_mean": _stat(row.prefill_tps, "mean"),
        "decode_tps_mean": _stat(row.decode_tps, "mean"),
        "request_throughput": row.request_throughput,
        "total_token_throughput": row.total_token_throughput,
        "overhead_ms_mean": _stat(row.overhead_ms, "mean"),
        "n_prompt_actual_mean": row.n_prompt_actual_mean,
        "cached_tokens_mean": row.cached_tokens_mean,
        "preemptions_delta_total": row.preemptions_delta_total,
        "flags": list(row.flags),
    }


def offline_result_to_metrics(res) -> dict[str, Any]:
    return {
        "pp_tps": res.pp_tps,
        "tg_tps": res.tg_tps,
        "total_tps": res.total_tps,
        # Aliased onto the online metric name so one objective can rank both kinds of row.
        "total_token_throughput": res.total_tps,
        "tps_mean": res.total_tps,
        "latency_s": res.latency_s,
        "e2e_ms_mean": (res.latency_s * 1000.0) if res.latency_s is not None else None,
        "n_threads": res.n_threads,
        "batch_size": res.batch_size,
        "rep": res.rep,
    }


class SweepRunner:
    """Owns the output directory and drives a plan to completion."""

    def __init__(self, plan: SweepPlan, *, on_event: Callable[[str, dict], None] | None = None):
        self.plan = plan
        self.spec: SuiteSpec = plan.spec
        self.out_dir = Path(self.spec.out_dir)
        self.on_event = on_event or (lambda kind, payload: None)

        self.results: list[TrialResult] = []
        self.run_id = f"{self.spec.name}-{new_run_salt()}"
        self.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._t0 = time.monotonic()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "records").mkdir(exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        self.rotated: list[str] = self._rotate_stale_artifacts()
        self._trials_path = self.out_dir / "trials.jsonl"
        self._trials_f = open(self._trials_path, "a")
        self._sqlite = SqliteSink(self.out_dir / "llmbench.db")
        self.deployment_records: list[dict[str, Any]] = []

    # -- persistence --

    def _rotate_stale_artifacts(self) -> list[str]:
        """Move a previous run's row files aside before this run starts writing.

        Rows are appended as they are measured, so that an interrupted sweep still leaves
        every completed trial on disk. The cost of that choice is that pointing a second run
        at the same `out_dir` would append its rows to the first run's, and the report -- which
        rebuilds itself from `trials.jsonl` and the *new* `run.json` -- would silently present
        two different sweeps as one table. Renaming is enough to prevent it, and keeps the old
        data: `llmbench sweep report` reads `trials.jsonl`, so the archived file is inert
        until someone deliberately looks at it.
        """
        run_json = self.out_dir / "run.json"
        suffix = "previous"
        try:
            suffix = json.loads(run_json.read_text()).get("run_id") or suffix
        except (OSError, ValueError):
            pass
        moved = []
        for name in ("trials.jsonl", "offline.jsonl"):
            path = self.out_dir / name
            if not (path.exists() and path.stat().st_size):
                continue
            archive = self.out_dir / f"{path.stem}.{suffix}{path.suffix}"
            n = 1
            while archive.exists():
                archive = self.out_dir / f"{path.stem}.{suffix}.{n}{path.suffix}"
                n += 1
            path.rename(archive)
            moved.append(archive.name)
        if moved:
            self.on_event("warning", {
                "deployment": "",
                "message": f"{self.out_dir} already held results from an earlier run; "
                           f"archived as {', '.join(moved)} so this run's report describes "
                           f"only this run",
            })
        return moved

    def _emit(self, result: TrialResult) -> None:
        """Append-as-you-go: a sweep killed at trial N still has N usable rows on disk."""
        result.run_id = self.run_id
        self.results.append(result)
        self._trials_f.write(json.dumps(result.to_dict()) + "\n")
        self._trials_f.flush()
        self.on_event("trial", result.to_dict())

    def close(self) -> None:
        self._trials_f.close()
        self._sqlite.close()

    # -- online --

    def _make_params(self, dep: DeploymentPlan, w: WorkloadPlan) -> CmdParams:
        return CmdParams(
            model=[dep.backend_spec.served_model_name or dep.backend_spec.model],
            reps=w.reps,
            warmup_fixed=self.spec.workload.warmup_fixed,
            no_warmup=self.spec.workload.no_warmup,
            request_rate=w.request_rate,
            force=self.spec.workload.force,
            url=dep.client_url,
            timeout=self.spec.request_timeout_s,
            backend=dep.backend,
            measure="both",
            endpoint=self.spec.endpoint,
            out_dir=str(self.out_dir),
            tag=dep.id,
        )

    def _make_instance(self, dep: DeploymentPlan, w: WorkloadPlan) -> Instance:
        return Instance(
            model=dep.backend_spec.served_model_name or dep.backend_spec.model,
            ngl=-1, batch=dep.batch, ubatch=dep.ubatch, ctk="f16", ctv="f16",
            # The thread count actually passed to the server, not the count of granted cpus:
            # `deployment.threads_per_instance` overrides the latter, and recording the wrong
            # one would put a number in every raw record that no server was ever launched with.
            flash_attn="auto", threads=_threads_of(dep),
            n_parallel=dep.n_parallel, n_ctx=dep.n_ctx,
            n_depth=w.n_depth, concurrency=w.concurrency, shared_prefix=w.shared_prefix,
            n_prompt=w.n_prompt, n_gen=w.n_gen, is_pg=w.is_pg,
        )

    async def _run_deployment(self, dep: DeploymentPlan, workloads: list[WorkloadPlan]) -> None:
        self.on_event("deployment_start", {"deployment": dep.to_dict()})

        if self.spec.dry_run:
            self._skip_group(dep, workloads, STATUS_SKIPPED, "dry-run")
            return

        try:
            live = await launch_deployment(dep, self.spec, out_dir=self.out_dir)
        except CapacityFailure as e:
            # "Did not fit" is a result about this configuration, not a harness failure.
            self._skip_group(dep, workloads, STATUS_CAPACITY, str(e))
            return
        except (DeploymentError, OSError) as e:
            self._skip_group(dep, workloads, STATUS_ERROR, str(e))
            if not self.spec.continue_on_error:
                raise
            return

        self.deployment_records.append({**dep.to_dict(), "live": live.to_dict()})
        self._write_json(self.out_dir / "deployments.json", self.deployment_records)

        misplaced = [a for a in live.affinity if a.get("threads_on_other_cpus")]
        if misplaced:
            self.on_event("warning", {
                "deployment": dep.id,
                "message": f"{len(misplaced)} instance(s) have threads pinned outside their "
                           f"allocation: {misplaced[0].get('error')}",
            })
        unverified = [
            a for a in live.affinity
            if not a.get("verified") and not a.get("threads_on_other_cpus")
        ]
        if unverified:
            self.on_event("warning", {
                "deployment": dep.id,
                "message": f"{len(unverified)} instance(s) could not be read from /proc; "
                           f"placement unverified (see deployments.json)",
            })

        backend = make_backend(dep, self.spec)
        jsonl_sink = JsonlSink(self.out_dir / "records" / f"{dep.id}.jsonl")

        class _MultiSink:
            def __init__(self, *sinks):
                self._sinks = sinks

            def write(self, rec):
                for s in self._sinks:
                    s.write(rec)

        sink = _MultiSink(jsonl_sink, self._sqlite)

        try:
            # Version banner only -- it lands in provenance and nothing depends on it. A
            # backend that will not answer /props must not cost the sweep every remaining
            # deployment, which is what letting this propagate used to do (the surrounding
            # `finally` tears down, but `continue_on_error` never got a say).
            try:
                info = await backend.info()
            except Exception as e:  # noqa: BLE001
                info = None
                self.on_event("warning", {
                    "deployment": dep.id,
                    "message": f"could not read backend version info ({type(e).__name__}: {e}); "
                               f"measuring anyway, provenance will lack backend_version",
                })
            for w in workloads:
                await self._run_workload(dep, w, backend, sink, info, live)
        finally:
            jsonl_sink.close()
            try:
                await backend.close()
            except Exception:  # noqa: BLE001 -- teardown must not mask a real failure
                pass
            for message in await live.teardown(settle_s=self.spec.settle_s):
                self.on_event("warning", {"deployment": dep.id, "message": message})
            self.on_event("deployment_end", {"deployment_id": dep.id})

    async def _run_workload(self, dep, w, backend, sink, info, live) -> None:
        trial_id = f"{dep.id}/{w.id}"
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t0 = time.monotonic()
        axes = {**dep.axes(), **_workload_axes(w)}
        self.on_event("trial_start", {"trial_id": trial_id, "test": w.test_name(), "axes": axes})

        params = self._make_params(dep, w)
        inst = self._make_instance(dep, w)
        run_id = f"{self.run_id}-{dep.id}-{w.id}"

        if isinstance(backend, FanoutBackend):
            backend.dispatch_counts = [0] * len(backend.members)

        # Pinning stops our servers leaving their cores; it cannot stop anyone else using
        # them. On a shared box that difference silently halves throughput, so measure it.
        monitor = ContentionMonitor(
            cpus=[c for i in dep.instances for c in i.cores.cpus],
            pids=[p.pid for p in live.processes],
        )
        monitor.start()

        try:
            records = await run_instance(
                backend, inst, params, run_id=run_id, sink=sink, run_salt=new_run_salt(),
            )
        except CapacityError as e:
            self._emit(TrialResult(
                trial_id=trial_id, kind="online", status=STATUS_CAPACITY,
                backend=dep.backend, model=dep.backend_spec.model_label(), test=w.test_name(),
                deployment_id=dep.id, workload_id=w.id, axes=axes, error=str(e),
                started_at_utc=started, duration_s=time.monotonic() - t0,
            ))
            return
        except Exception as e:  # noqa: BLE001 -- one bad trial must not kill the sweep
            self._emit(TrialResult(
                trial_id=trial_id, kind="online", status=STATUS_ERROR,
                backend=dep.backend, model=dep.backend_spec.model_label(), test=w.test_name(),
                deployment_id=dep.id, workload_id=w.id, axes=axes,
                error=f"{type(e).__name__}: {e}",
                provenance={"traceback": traceback.format_exc()},
                started_at_utc=started, duration_s=time.monotonic() - t0,
            ))
            if not self.spec.continue_on_error:
                raise
            return

        raw = [r.to_dict() for r in records]
        duration = time.monotonic() - t0
        contention = monitor.sample()
        contention_warning = contention.warning(CONTENTION_WARN_PCT)
        if contention_warning:
            self.on_event("warning", {"deployment": dep.id, "message": contention_warning})

        # A request that failed still produces a record, and `aggregate_client` still produces
        # a row from it -- one whose tps_mean is 0.0 because there was nothing to average.
        # Emitting that as `ok` would put a plausible-looking zero into the report and into
        # the ranking, so the row's status has to reflect what actually happened to the
        # requests, not merely that the harness got through the trial without raising.
        n_errors = sum(1 for r in raw if r.get("error"))
        first_error = next((r.get("error") for r in raw if r.get("error")), None)
        all_failed = bool(raw) and n_errors == len(raw)
        partial_warning = None
        if n_errors and not all_failed:
            partial_warning = (
                f"{n_errors} of {len(raw)} request(s) failed during this trial (first: "
                f"{first_error}); the statistics below describe only the {len(raw) - n_errors} "
                f"that succeeded"
            )
            self.on_event("warning", {"deployment": dep.id, "message": partial_warning})

        provenance = {
            "contention": contention.to_dict(),
            "run_id": run_id,
            "client_url": dep.client_url,
            "n_records": len(raw),
            "n_errors": n_errors,
            "warmup_iterations": raw[0].get("warmup_iterations") if raw else None,
            "backend_version": getattr(info, "version", None),
            "records_file": f"records/{dep.id}.jsonl",
        }
        if isinstance(backend, FanoutBackend):
            provenance["lb_distribution"] = backend.distribution()

        warnings = list(dep.warnings)
        for extra in (contention_warning, partial_warning):
            if extra:
                warnings.append(extra)

        for src in ("client", "server"):
            row = (metrics.aggregate_client(raw, w.test_name()) if src == "client"
                   else metrics.aggregate_server(raw, w.test_name()))
            if row is None:
                continue
            self._emit(TrialResult(
                trial_id=trial_id, kind="online",
                status=STATUS_ERROR if (all_failed or not raw) else STATUS_OK, src=src,
                backend=dep.backend, model=dep.backend_spec.model_label(), test=w.test_name(),
                deployment_id=dep.id, workload_id=w.id, axes=axes,
                metrics=result_row_to_metrics(row),
                comparable_across_backends=(src == "client"),
                error=(f"all {len(raw)} request(s) failed: {first_error}" if all_failed else
                       ("the trial produced no records at all" if not raw else None)),
                warnings=warnings,
                started_at_utc=started, duration_s=duration,
                provenance=provenance,
            ))

    def _skip_group(self, dep, workloads, status: str, error: str) -> None:
        if status != STATUS_SKIPPED:
            self.on_event("warning", {"deployment": dep.id, "message": error})
        for w in workloads:
            axes = {**dep.axes(), **_workload_axes(w)}
            # Emit trial_start for skipped work too. The progress counter counts planned units
            # of work, so staying silent here would leave it permanently short of its own
            # total whenever a deployment failed to launch.
            self.on_event("trial_start", {
                "trial_id": f"{dep.id}/{w.id}", "test": w.test_name(), "axes": axes,
            })
            self._emit(TrialResult(
                trial_id=f"{dep.id}/{w.id}", kind="online", status=status,
                backend=dep.backend, model=dep.backend_spec.model_label(), test=w.test_name(),
                deployment_id=dep.id, workload_id=w.id,
                axes=axes, error=error, warnings=list(dep.warnings),
            ))

    # -- offline --

    def _run_offline(self, o: OfflinePlan) -> None:
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        axes = {
            "backend": o.backend, "tool": o.tool, "batch_size": o.batch_size,
            "cores_per_instance": o.cores.n_physical_cores,
            "threads_per_instance": o.cores.n_threads, "instances": 1, "lb": "none",
        }
        self.on_event("trial_start", {"trial_id": o.id, "test": o.test_name(), "axes": axes})

        if self.spec.dry_run:
            self._emit(TrialResult(
                trial_id=o.id, kind="offline", status=STATUS_SKIPPED, backend=o.backend,
                model=o.backend_spec.model_label(), test=o.test_name(), axes=axes,
                error="dry-run", comparable_across_backends=False,
            ))
            return

        try:
            results = run_offline(
                o, self.spec.cpu, log_dir=self.out_dir / "logs" / "offline",
                timeout_s=self.spec.startup_timeout_s + self.spec.request_timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            self._emit(TrialResult(
                trial_id=o.id, kind="offline", status=STATUS_ERROR, backend=o.backend,
                model=o.backend_spec.model_label(), test=o.test_name(), axes=axes,
                error=f"{type(e).__name__}: {e}", comparable_across_backends=False,
                provenance={"traceback": traceback.format_exc()}, started_at_utc=started,
            ))
            if not self.spec.continue_on_error:
                raise
            return

        with open(self.out_dir / "offline.jsonl", "a") as f:
            for res in results:
                f.write(json.dumps(res.to_dict()) + "\n")

        for res in results:
            self._emit(TrialResult(
                trial_id=f"{o.id}/r{res.rep}", kind="offline",
                status=STATUS_ERROR if res.error else STATUS_OK,
                src="native", backend=res.backend, model=o.backend_spec.model_label(),
                test=res.test, axes={**axes, "tool": res.tool},
                metrics=offline_result_to_metrics(res),
                comparable_across_backends=False, error=res.error,
                started_at_utc=started, duration_s=res.duration_s,
                provenance={
                    "argv": res.argv, "env": res.env, "log": res.stdout_path,
                    "returncode": res.returncode,
                    "reps_mechanism": res.raw.get("reps_mechanism"),
                },
            ))

    # -- driver --

    async def run(self) -> list[TrialResult]:
        self._write_json(self.out_dir / "plan.json", self.plan.to_dict())
        self._write_manifest(status="running")
        try:
            for dep, workloads in self.plan.groups:
                await self._run_deployment(dep, workloads)
            for o in self.plan.offline:
                # Offline tools own the whole core budget, so they must not overlap a live
                # fleet. They run only after every deployment has been torn down.
                self._run_offline(o)
        finally:
            self._write_manifest(status="finished")
            self.close()
        return self.results

    # -- manifest --

    def _write_json(self, path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, default=str))

    def _write_manifest(self, *, status: str) -> None:
        env = capture_env()
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        self._write_json(self.out_dir / "run.json", {
            "run_id": self.run_id,
            "name": self.spec.name,
            "status": status,
            "started_at_utc": self.started_at,
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "mode": self.spec.mode,
            "objective": {
                "metric": self.spec.objective.metric,
                "goal": self.spec.objective.goal,
                "src": self.spec.objective.src,
                "constraints": [dataclasses.asdict(c) for c in self.spec.objective.constraints],
                "described": self.spec.objective.describe(),
            },
            "spec": self.spec.to_dict(),
            "env": env.to_dict(),
            "topology": self.plan.topology.to_dict(),
            "counts": {
                "planned_online": self.plan.n_online_trials,
                "planned_offline": self.plan.n_offline_trials,
                "emitted_rows": len(self.results),
                "by_status": counts,
            },
            "warnings": list(self.plan.warnings),
            "artifacts": {
                "plan": "plan.json",
                "trials": "trials.jsonl",
                "offline": "offline.jsonl",
                "deployments": "deployments.json",
                "raw_records": "records/",
                "sqlite": "llmbench.db",
                "logs": "logs/",
            },
        })


def _threads_of(dep: DeploymentPlan) -> int:
    """The `-t` a server in this deployment is launched with. Mirrors deploy.build_server_command."""
    if dep.threads_per_instance:
        return dep.threads_per_instance
    return dep.instances[0].cores.n_threads if dep.instances else -1


def _workload_axes(w: WorkloadPlan) -> dict[str, Any]:
    return {
        "n_prompt": w.n_prompt, "n_gen": w.n_gen, "n_depth": w.n_depth,
        "concurrency": w.concurrency, "shared_prefix": w.shared_prefix,
        "request_rate": w.request_rate,
    }


async def execute(plan: SweepPlan, *, on_event=None) -> tuple[list[TrialResult], SweepRunner]:
    runner = SweepRunner(plan, on_event=on_event)
    results = await runner.run()
    return results, runner


__all__ = ["SweepRunner", "TrialResult", "execute", "result_row_to_metrics",
           "STATUS_OK", "STATUS_CAPACITY", "STATUS_ERROR", "STATUS_SKIPPED"]
