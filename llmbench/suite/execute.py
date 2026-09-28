"""Execute a SweepPlan: bring fleets up, drive workloads, run offline tools, persist everything.

Measurement itself is delegated, deliberately. Online trials go through the existing
`runner.run_instance` + `metrics.aggregate_*` path -- the same code the single-endpoint CLI
uses -- so a sweep row and a one-off `llmbench` row are computed by identical arithmetic and
`tests/test_metrics_no_backend_branch.py` still mechanically guarantees no backend-specific
branching. This module owns orchestration and persistence, not statistics.

Raw records are always written before anything is aggregated, so a sweep that dies at trial
90 of 126 leaves the first 89 trials fully re-analysable from disk -- and `--resume` picks up
at trial 90 instead of starting again.
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import socket
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
from . import procs
from .contention import ContentionMonitor
from .deploy import CapacityFailure, DeploymentError, launch_deployment
from .hostinfo import capture_host, capture_software
from .lb import make_backend
from .lb.fanout import FanoutBackend
from .offline import run_offline
from .plan import DeploymentPlan, OfflinePlan, SweepPlan, WorkloadPlan
from .spec import SuiteSpec

STATUS_OK = "ok"
STATUS_CAPACITY = "capacity"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

# A unit of planned work (one workload on one deployment, or one offline plan) is complete
# when every row it produced has one of these. `capacity` is a result ("did not fit"), so
# resuming does not retry it; `error` and `skipped` are not, so resuming does.
_DONE_STATUSES = (STATUS_OK, STATUS_CAPACITY)

# What run.json's `status` can say. `running` on a directory nobody is writing to means the
# sweep was killed too hard to record anything (SIGKILL, power loss) -- see pids.json.
RUN_RUNNING = "running"
RUN_FINISHED = "finished"
RUN_INTERRUPTED = "interrupted"
RUN_FAILED = "failed"

# Foreign CPU load above this share of a deployment's allocated cpus marks a trial as
# contaminated. 15% is roughly where a measurement stops being reproducible on this class of
# machine; below it, normal system noise dominates.
CONTENTION_WARN_PCT = 15.0


class ResumeError(RuntimeError):
    """`--resume` was asked for but the out_dir does not hold a resumable run of this plan."""


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
    # Every measured request of an online trial, one dict per rep (metrics.per_request_values),
    # so a 3-rep trial keeps all three values and not only their mean. Client rows only.
    reps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def unit_id(self) -> str:
        """The planned unit of work this row belongs to.

        An online trial id *is* its unit (`d000/w0003`). An offline plan emits one row per
        rep (`o0001/r2`), or a single `o0001` row when the tool failed outright.
        """
        return self.trial_id if self.kind == "online" else self.trial_id.split("/", 1)[0]


def read_trials(path: Path) -> tuple[list[TrialResult], int]:
    """Read trials.jsonl defensively. Returns (rows, n_unreadable).

    Two things this must survive, because both happen to real runs. A row is flushed after
    every trial, so a hard kill can leave the last line half-written -- losing one truncated
    line must not cost the report the ninety complete ones above it. And a directory can hold
    rows written by a different version of the harness, whose extra or missing keys would
    otherwise make the constructor raise; unknown keys are dropped and absent ones default.
    """
    known = {f.name for f in dataclasses.fields(TrialResult)}
    rows: list[TrialResult] = []
    skipped = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            rows.append(TrialResult(**{k: v for k, v in obj.items() if k in known}))
        except (ValueError, TypeError):
            skipped += 1
    return rows, skipped


def completed_units(rows: list[TrialResult]) -> set[str]:
    """Units whose every row is a result (ok or capacity) rather than a failure to measure."""
    status_by_unit: dict[str, set[str]] = {}
    for r in rows:
        status_by_unit.setdefault(r.unit_id, set()).add(r.status)
    return {u for u, statuses in status_by_unit.items() if statuses <= set(_DONE_STATUSES)}


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


class _MultiSink:
    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, rec):
        for s in self._sinks:
            s.write(rec)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _event_summary(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """What events.jsonl keeps of an event. Full rows already live in trials.jsonl and full
    deployments in deployments.json; the log is for reconstructing *when* things happened."""
    if kind == "trial":
        return {
            "trial_id": payload.get("trial_id"), "src": payload.get("src"),
            "status": payload.get("status"), "error": payload.get("error"),
            "duration_s": payload.get("duration_s"),
            "tps_mean": (payload.get("metrics") or {}).get("tps_mean"),
            "n_warnings": len(payload.get("warnings") or []),
        }
    if kind == "deployment_start":
        d = payload.get("deployment") or {}
        return {"deployment_id": d.get("id"), "axes": d.get("axes")}
    if kind == "trial_start":
        return {"trial_id": payload.get("trial_id"), "test": payload.get("test")}
    return dict(payload)


class SweepRunner:
    """Owns the output directory and drives a plan to completion."""

    def __init__(self, plan: SweepPlan, *, on_event: Callable[[str, dict], None] | None = None,
                 resume: bool = False):
        self.plan = plan
        self.spec: SuiteSpec = plan.spec
        self.out_dir = Path(self.spec.out_dir)
        self.on_event = on_event or (lambda kind, payload: None)

        self.results: list[TrialResult] = []
        self.done_units: set[str] = set()
        self.fingerprint = plan.fingerprint()
        self.started_at = _utc_now()
        self.first_started_at = self.started_at
        self.attempt = 1
        self.status = "pending"
        self.error: str | None = None
        self.error_traceback: str | None = None
        self.stop_reason: str | None = None
        self.manifest: dict[str, Any] = {}
        self.host: dict[str, Any] = {}
        self.software: dict[str, Any] = {}
        self.deployment_records: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        self._closed = False

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "records").mkdir(exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        self._refuse_if_previous_run_is_alive()

        self._trials_path = self.out_dir / "trials.jsonl"
        prior = self._read_json(self.out_dir / "run.json")
        self.run_id = f"{self.spec.name}-{new_run_salt()}"
        self._events_f = open(self.out_dir / "events.jsonl", "a")
        try:
            if resume:
                self.rotated = self._prepare_resume(prior)
            else:
                self.rotated = self._rotate_stale_artifacts(prior)
        except BaseException:
            self._events_f.close()
            raise
        self._trials_f = open(self._trials_path, "a")
        self._sqlite = SqliteSink(self.out_dir / "llmbench.db")
        self._procs = procs.activate(self.out_dir / procs.PIDS_FILE, self.run_id)

    # -- start-up checks --

    def _refuse_if_previous_run_is_alive(self) -> None:
        """Two sweeps in one out_dir would interleave rows; a dead one's servers would hold
        the ports. Either way, say so here rather than as a port pre-flight failure later."""
        state = procs.inspect(self.out_dir / procs.PIDS_FILE)
        if state.owner_alive:
            raise DeploymentError(
                f"another sweep (run {state.run_id}, pid {state.owner_pid}) is still running "
                f"against {self.out_dir}. Stop it, or use a different out_dir."
            )
        if state.leftovers:
            raise DeploymentError(
                f"{len(state.leftovers)} process(es) started by an earlier run "
                f"({state.run_id}) in {self.out_dir} are still alive: "
                f"{'; '.join(lo.describe() for lo in state.leftovers)}. That run was killed "
                f"before it could tear down. Stop them with "
                f"`llmbench sweep cleanup {self.out_dir}`."
            )

    # -- persistence --

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _archive(path: Path, tag: str) -> Path:
        archive = path.with_name(f"{path.stem}.{tag}{path.suffix}")
        n = 1
        while archive.exists():
            archive = path.with_name(f"{path.stem}.{tag}.{n}{path.suffix}")
            n += 1
        path.rename(archive)
        return archive

    def _rotate_stale_artifacts(self, prior: dict[str, Any]) -> list[str]:
        """Move a previous run's row files aside before this run starts writing.

        Rows are appended as they are measured, so that an interrupted sweep still leaves
        every completed trial on disk. The cost of that choice is that pointing a second run
        at the same `out_dir` would append its rows to the first run's, and the report -- which
        rebuilds itself from `trials.jsonl` and the *new* `run.json` -- would silently present
        two different sweeps as one table. Renaming is enough to prevent it, and keeps the old
        data: `llmbench sweep report` reads `trials.jsonl`, so the archived file is inert
        until someone deliberately looks at it.

        To continue an interrupted run instead of replacing it, use `--resume`.
        """
        suffix = prior.get("run_id") or "previous"
        moved = []
        for name in ("trials.jsonl", "offline.jsonl", "deployments.json"):
            path = self.out_dir / name
            if not (path.exists() and path.stat().st_size):
                continue
            moved.append(self._archive(path, suffix).name)
        if moved:
            self._event("warning", {
                "deployment": "",
                "message": f"{self.out_dir} already held results from an earlier run; "
                           f"archived as {', '.join(moved)} so this run's report describes "
                           f"only this run (pass --resume to continue that run instead)",
            })
        return moved

    def _prepare_resume(self, prior: dict[str, Any]) -> list[str]:
        """Continue the run recorded in out_dir: keep its finished units, redo the rest.

        Kept: every row of a unit that completed (`ok` or `capacity`). Redone: units with an
        `error` or `skipped` row, and units that never emitted a row because the sweep died
        inside them. The previous trials.jsonl is archived whole before being rewritten with
        only the kept rows, so an error row that is about to be superseded is still on disk
        -- but no longer in the table next to the row that replaced it.
        """
        if not prior.get("run_id"):
            raise ResumeError(f"nothing to resume: {self.out_dir}/run.json is missing or unreadable")
        if prior.get("plan_fingerprint") != self.fingerprint:
            raise ResumeError(
                f"cannot resume {prior['run_id']}: the spec no longer describes the same "
                f"measurements (plan fingerprint {prior.get('plan_fingerprint') or 'absent'} "
                f"on disk, {self.fingerprint} now). Only timeouts, settle_s, "
                f"continue_on_error, the objective and the name may change between attempts. "
                f"Start a fresh run in a new out_dir instead."
            )
        self.run_id = prior["run_id"]
        self.attempt = int(prior.get("attempt") or 1) + 1
        self.first_started_at = prior.get("first_started_at_utc") or prior.get("started_at_utc") \
            or self.started_at

        rows: list[TrialResult] = []
        n_unreadable = 0
        moved: list[str] = []
        if self._trials_path.exists():
            rows, n_unreadable = read_trials(self._trials_path)
            moved.append(self._archive(self._trials_path, f"{self.run_id}.attempt{self.attempt - 1}").name)
        self.done_units = completed_units(rows)
        kept = [r for r in rows if r.unit_id in self.done_units]
        procs.atomic_write_text(
            self._trials_path, "".join(json.dumps(r.to_dict()) + "\n" for r in kept),
        )
        self.results = kept

        records = self._read_json_list(self.out_dir / "deployments.json")
        self.deployment_records = records

        planned = self._planned_units()
        self._event("resume", {
            "previous_status": prior.get("status"),
            "previous_attempt": self.attempt - 1,
            "units_planned": len(planned),
            "units_already_done": len(self.done_units & planned),
            "rows_kept": len(kept),
            "rows_superseded": len(rows) - len(kept),
            "unreadable_lines": n_unreadable,
            "archived": moved,
        })
        return moved

    @staticmethod
    def _read_json_list(path: Path) -> list[Any]:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        return data if isinstance(data, list) else []

    def _planned_units(self) -> set[str]:
        units = {f"{d.id}/{w.id}" for d, ws in self.plan.groups for w in ws}
        units |= {o.id for o in self.plan.offline}
        return units

    @property
    def n_units_done(self) -> int:
        """Planned units already complete before this attempt started (non-zero on resume)."""
        return len(self.done_units & self._planned_units())

    def _emit(self, result: TrialResult) -> None:
        """Append-as-you-go: a sweep killed at trial N still has N usable rows on disk."""
        result.run_id = self.run_id
        self.results.append(result)
        self._trials_f.write(json.dumps(result.to_dict()) + "\n")
        self._trials_f.flush()
        self._event("trial", result.to_dict())

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        """Record an event in events.jsonl, then hand it to the caller's callback.

        The file is the durable copy: terminal output is gone once an overnight session
        closes, and several warnings (a port that never freed, threads pinned outside their
        allocation) are attached to no row. Neither half may take the sweep down -- a full
        disk or a vanished terminal is not a reason to abandon live servers mid-measurement.
        """
        line = {
            "ts_utc": _utc_now(), "elapsed_s": round(time.monotonic() - self._t0, 2),
            "run_id": getattr(self, "run_id", ""), "attempt": self.attempt, "kind": kind,
            **_event_summary(kind, payload),
        }
        f = getattr(self, "_events_f", None)
        if f is not None and not f.closed:
            try:
                f.write(json.dumps(line, default=str) + "\n")
                f.flush()
            except OSError:
                pass
        try:
            self.on_event(kind, payload)
        except OSError:
            pass

    def request_stop(self, reason: str) -> None:
        """Record why the run is stopping. The caller then cancels the task driving run()."""
        if self.stop_reason is None:
            self.stop_reason = reason
            self._event("stop_requested", {"reason": reason})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._trials_f.close()
        self._sqlite.close()
        procs.deactivate(self._procs)
        self._events_f.close()

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
            backend=dep.backend_spec.type,
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
        self._event("deployment_start", {"deployment": dep.to_dict()})

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

        # From here the fleet is running, so *everything* up to teardown sits inside the
        # try: a failure writing deployments.json (disk full) or building the client used to
        # happen before it, and left every server of the deployment running.
        backend = None
        jsonl_sink = None
        try:
            self._event("deployment_launched", {
                "deployment_id": dep.id, "launch_seconds": round(live.launch_seconds, 2),
                "pids": [p.pid for p in live.processes],
            })
            self.deployment_records.append({**dep.to_dict(), "live": live.to_dict()})
            self._write_json(self.out_dir / "deployments.json", self.deployment_records)

            misplaced = [a for a in live.affinity if a.get("threads_on_other_cpus")]
            if misplaced:
                self._event("warning", {
                    "deployment": dep.id,
                    "message": f"{len(misplaced)} instance(s) have threads pinned outside their "
                               f"allocation: {misplaced[0].get('error')}",
                })
            unverified = [
                a for a in live.affinity
                if not a.get("verified") and not a.get("threads_on_other_cpus")
            ]
            if unverified:
                self._event("warning", {
                    "deployment": dep.id,
                    "message": f"{len(unverified)} instance(s) could not be read from /proc; "
                               f"placement unverified (see deployments.json)",
                })

            backend = make_backend(dep, self.spec)
            jsonl_sink = JsonlSink(self.out_dir / "records" / f"{dep.id}.jsonl")
            sink = _MultiSink(jsonl_sink, self._sqlite)

            # Version banner only -- it lands in provenance and nothing depends on it. A
            # backend that will not answer /props must not cost the sweep every remaining
            # deployment, which is what letting this propagate used to do (the surrounding
            # `finally` tears down, but `continue_on_error` never got a say).
            try:
                info = await backend.info()
            except Exception as e:  # noqa: BLE001
                info = None
                self._event("warning", {
                    "deployment": dep.id,
                    "message": f"could not read backend version info ({type(e).__name__}: {e}); "
                               f"measuring anyway, provenance will lack backend_version",
                })
            for w in workloads:
                await self._run_workload(dep, w, backend, sink, info, live)
        finally:
            if jsonl_sink is not None:
                jsonl_sink.close()
            if backend is not None:
                try:
                    await backend.close()
                except Exception:  # noqa: BLE001 -- teardown must not mask a real failure
                    pass
            for message in await live.teardown(settle_s=self.spec.settle_s):
                self._event("warning", {"deployment": dep.id, "message": message})
            self._event("deployment_end", {"deployment_id": dep.id})

    def _trial_run_id(self, dep: DeploymentPlan, w: WorkloadPlan) -> str:
        """The run_id written into every raw record of one trial.

        Attempt-qualified after a resume: a trial that errored in attempt 1 still has its
        failed-request records in records/, and `report --from-records` groups records by
        this id, so reusing it would pool the old failures with the new measurement.
        """
        if self.attempt == 1:
            return f"{self.run_id}-{dep.id}-{w.id}"
        return f"{self.run_id}-a{self.attempt}-{dep.id}-{w.id}"

    async def _run_workload(self, dep, w, backend, sink, info, live) -> None:
        trial_id = f"{dep.id}/{w.id}"
        started = _utc_now()
        t0 = time.monotonic()
        axes = {**dep.axes(), **_workload_axes(w)}
        self._event("trial_start", {"trial_id": trial_id, "test": w.test_name(), "axes": axes})

        params = self._make_params(dep, w)
        inst = self._make_instance(dep, w)
        run_id = self._trial_run_id(dep, w)

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
            self._event("warning", {"deployment": dep.id, "message": contention_warning})

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
            self._event("warning", {"deployment": dep.id, "message": partial_warning})

        # Carried on every row, not only in deployments.json, so the ranking can refuse a row
        # whose fleet was not core-isolated without having to join against another file.
        stray = sum(int(a.get("threads_on_other_cpus") or 0) for a in live.affinity)
        placement = {
            "verified": all(a.get("verified") for a in live.affinity),
            "threads_on_other_cpus": stray,
        }
        placement_warning = None
        if stray:
            placement_warning = (
                f"{stray} server thread(s) were pinned outside this deployment's allocation; "
                f"it was not core-isolated and these numbers should not be compared against "
                f"correctly placed ones"
            )

        provenance = {
            "contention": contention.to_dict(),
            "placement": placement,
            "run_id": run_id,
            "client_url": dep.client_url,
            "n_records": len(raw),
            "n_errors": n_errors,
            "warmup_iterations": raw[0].get("warmup_iterations") if raw else None,
            "backend_version": getattr(info, "version", None),
            "records_file": f"records/{dep.id}.jsonl",
            # The exact command each server of this deployment was started with: argv (numactl
            # prefix included) and the environment variables set on top of the sweep's own.
            "server_commands": [
                {"name": p.name, "argv": list(p.argv), "env": dict(p.env_overrides)}
                for p in live.processes
            ],
        }
        if isinstance(backend, FanoutBackend):
            provenance["lb_distribution"] = backend.distribution()

        warnings = list(dep.warnings)
        for extra in (contention_warning, partial_warning, placement_warning):
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
                reps=metrics.per_request_values(raw) if src == "client" else [],
            ))

    def _skip_group(self, dep, workloads, status: str, error: str) -> None:
        if status != STATUS_SKIPPED:
            self._event("warning", {"deployment": dep.id, "message": error})
        for w in workloads:
            axes = {**dep.axes(), **_workload_axes(w)}
            # Emit trial_start for skipped work too. The progress counter counts planned units
            # of work, so staying silent here would leave it permanently short of its own
            # total whenever a deployment failed to launch.
            self._event("trial_start", {
                "trial_id": f"{dep.id}/{w.id}", "test": w.test_name(), "axes": axes,
            })
            self._emit(TrialResult(
                trial_id=f"{dep.id}/{w.id}", kind="online", status=status,
                backend=dep.backend, model=dep.backend_spec.model_label(), test=w.test_name(),
                deployment_id=dep.id, workload_id=w.id,
                axes=axes, error=error, warnings=list(dep.warnings),
            ))

    # -- offline --

    async def _run_offline(self, o: OfflinePlan) -> None:
        started = _utc_now()
        axes = {
            "backend": o.backend, "tool": o.tool, "batch_size": o.batch_size,
            "cores_per_instance": o.cores.n_physical_cores,
            "threads_per_instance": o.cores.n_threads, "instances": 1, "lb": "none",
        }
        self._event("trial_start", {"trial_id": o.id, "test": o.test_name(), "axes": axes})

        if self.spec.dry_run:
            self._emit(TrialResult(
                trial_id=o.id, kind="offline", status=STATUS_SKIPPED, backend=o.backend,
                model=o.backend_spec.model_label(), test=o.test_name(), axes=axes,
                error="dry-run", comparable_across_backends=False,
            ))
            return

        # In a worker thread so the event loop -- and with it the signal handlers -- stays
        # responsive. A SIGTERM during a blocking `communicate()` on the loop thread would
        # otherwise wait for the whole tool run before the sweep noticed it.
        work = asyncio.ensure_future(asyncio.to_thread(
            run_offline, o, self.spec.cpu, log_dir=self.out_dir / "logs" / "offline",
            timeout_s=self.spec.startup_timeout_s + self.spec.request_timeout_s,
        ))
        try:
            results = await asyncio.shield(work)
        except asyncio.CancelledError:
            # The thread cannot be cancelled. Stopping the registry kills the tool it is
            # waiting on and makes any further launch (the next llama-batched-bench rep)
            # raise RunStopping, so it ends promptly -- and it must end *before* close()
            # deactivates the registry, or that next rep would start unsupervised.
            self._procs.stop()
            try:
                await work
            except BaseException:  # noqa: BLE001 -- RunStopping, or the killed tool's error
                pass
            raise
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
        self.status = RUN_RUNNING
        self._write_json(self.out_dir / "plan.json", self.plan.to_dict())
        self.host = capture_host()
        # Skipped on a dry run: `vllm --version` alone can take tens of seconds.
        self.software = {} if self.spec.dry_run else capture_software(self.spec.backends)
        self._write_manifest()
        self._event("run_start", {
            "pid": os.getpid(), "attempt": self.attempt, "plan_fingerprint": self.fingerprint,
            "units_planned": len(self._planned_units()), "units_already_done": self.n_units_done,
        })
        try:
            for dep, workloads in self.plan.groups:
                todo = [w for w in workloads if f"{dep.id}/{w.id}" not in self.done_units]
                if not todo:
                    # Resumed and already complete: not relaunching a fleet is the whole point.
                    self._event("deployment_skipped", {
                        "deployment_id": dep.id, "reason": "already complete (resumed run)",
                    })
                    continue
                await self._run_deployment(dep, todo)
            for o in self.plan.offline:
                if o.id in self.done_units:
                    continue
                # Offline tools own the whole core budget, so they must not overlap a live
                # fleet. They run only after every deployment has been torn down.
                await self._run_offline(o)
            self.status = RUN_FINISHED
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.status = RUN_INTERRUPTED
            self.error = f"stopped by {self.stop_reason}" if self.stop_reason else "cancelled"
            raise
        except BaseException as e:
            self.status = RUN_FAILED
            self.error = f"{type(e).__name__}: {e}"
            self.error_traceback = traceback.format_exc()
            raise
        finally:
            # Last line of defence. Every normal and error path has already torn its fleet
            # down; anything still registered here is a leak, and must not outlive the run.
            leaked = self._procs.stop()
            if leaked:
                self._event("warning", {
                    "deployment": "",
                    "message": f"{len(leaked)} process(es) were still running at the end of "
                               f"the run and have been stopped: {', '.join(leaked)}",
                })
            self._write_manifest()
            self._event("run_end", {
                "status": self.status, "error": self.error, "rows": len(self.results),
            })
            self.close()
        return self.results

    # -- manifest --

    def _write_json(self, path: Path, obj: Any) -> None:
        procs.atomic_write_text(path, json.dumps(obj, indent=2, default=str))

    def _write_manifest(self) -> None:
        """run.json. Written at start and at the end, whatever the end was.

        `status` says how the run ended -- finished, interrupted (and by what), or failed
        (with the exception) -- rather than `finished` for all three, which is what it used to
        record even for a run that crashed on its first deployment.
        """
        env = capture_env()
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        ended = self.status not in (RUN_RUNNING, "pending")
        self.manifest = {
            "run_id": self.run_id,
            "name": self.spec.name,
            "status": self.status,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "traceback": self.error_traceback,
            "attempt": self.attempt,
            "plan_fingerprint": self.fingerprint,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "host_info": self.host,
            "software": self.software,
            "first_started_at_utc": self.first_started_at,
            "started_at_utc": self.started_at,
            "ended_at_utc": _utc_now() if ended else None,
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "mode": self.spec.mode,
            "objective": {
                "metric": self.spec.objective.metric,
                "goal": self.spec.objective.goal,
                "src": self.spec.objective.src,
                "constraints": [dataclasses.asdict(c) for c in self.spec.objective.constraints],
                "max_error_pct": self.spec.objective.max_error_pct,
                "rank_flagged": self.spec.objective.rank_flagged,
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
                "events": "events.jsonl",
                "pids": procs.PIDS_FILE,
                "raw_records": "records/",
                "per_rep_values": "report_reps.csv",
                "sqlite": "llmbench.db",
                "logs": "logs/",
            },
        }
        try:
            self._write_json(self.out_dir / "run.json", self.manifest)
        except OSError as e:
            # The in-memory manifest still reaches the reports; a full disk here must not
            # replace the real outcome of the run with an OSError.
            self._event("warning", {"deployment": "", "message": f"could not write run.json: {e}"})


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


__all__ = ["SweepRunner", "TrialResult", "ResumeError", "execute", "read_trials",
           "completed_units", "result_row_to_metrics",
           "STATUS_OK", "STATUS_CAPACITY", "STATUS_ERROR", "STATUS_SKIPPED",
           "RUN_FINISHED", "RUN_INTERRUPTED", "RUN_FAILED"]
