"""What an unattended sweep leaves behind when it ends badly, and how it picks up again.

An overnight run meets every failure mode eventually: a dropped SSH session (SIGHUP), a
`kill`, the OOM killer, a harness exception at 3am. These pin the properties that make that
survivable: the manifest says how the run ended, no fleet outlives the run, every event is
on disk, a killed run's servers can be found and stopped, a resumed run redoes only what did
not finish -- and a row measured under contention never becomes the answer.

Tests that need real processes and /proc are Linux-only; the rest drive SweepRunner against
the stub backend from test_suite_execute.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from llmbench.suite import cli, procs
from llmbench.suite import execute as execute_mod
from llmbench.suite.deploy import DeploymentError, spawn
from llmbench.suite.execute import (
    STATUS_CAPACITY, STATUS_ERROR, STATUS_OK, STATUS_SKIPPED, ResumeError, SweepRunner,
    TrialResult, completed_units,
)
from llmbench.suite.objective import rank
from llmbench.suite.plan import build_plan
from llmbench.suite.spec import ObjectiveSpec
from tests.test_suite_execute import StubBackend, StubLive, make_spec
from tests.test_suite_topology import make_topology

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="needs /proc and POSIX process groups",
)

TWO_BY_TWO = {
    "deployment": {"backend": ["llamacpp"], "instances": [1, 2]},
    "workload": {"n_prompt": [64, 128], "n_gen": [0], "reps": 3, "no_warmup": True},
}


def two_by_two(tmp_path, **overrides):
    return make_spec(tmp_path, **{**TWO_BY_TWO, **overrides})


def patch_fleet(monkeypatch, backend_for=lambda dep: StubBackend()):
    """Stub out launch and client construction; record what was launched."""
    launched: list[str] = []
    lives: list[StubLive] = []

    async def fake_launch(dep, spec, *, out_dir):
        launched.append(dep.id)
        live = StubLive()
        lives.append(live)
        return live

    monkeypatch.setattr(execute_mod, "launch_deployment", fake_launch)
    monkeypatch.setattr(execute_mod, "make_backend", lambda dep, spec: backend_for(dep))
    return launched, lives


def run_sweep(spec, *, resume: bool = False) -> SweepRunner:
    runner = SweepRunner(build_plan(spec, make_topology()), resume=resume)
    asyncio.run(runner.run())
    return runner


def manifest(tmp_path) -> dict:
    return json.loads((tmp_path / "run.json").read_text())


def events(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def trials(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]


# --- run.json says how the run ended ---


def test_a_run_that_completes_is_recorded_as_finished(tmp_path, monkeypatch):
    patch_fleet(monkeypatch)
    runner = run_sweep(make_spec(tmp_path))

    m = manifest(tmp_path)
    assert m["status"] == "finished" and m["error"] is None
    assert m["ended_at_utc"] and m["plan_fingerprint"] == runner.fingerprint
    assert not list(tmp_path.glob(".*.tmp")), "an atomic write left its temp file behind"


def test_a_run_that_raises_is_recorded_as_failed_not_finished(tmp_path, monkeypatch):
    """run.json used to say `finished` in its `finally` whatever had happened -- including
    for a run that crashed on its first deployment."""
    class Exploding(StubBackend):
        async def capacity(self):
            raise RuntimeError("backend fell over")

    _, lives = patch_fleet(monkeypatch, lambda dep: Exploding())
    with pytest.raises(RuntimeError):
        run_sweep(make_spec(tmp_path, continue_on_error=False))

    m = manifest(tmp_path)
    assert m["status"] == "failed"
    assert "backend fell over" in m["error"] and "Traceback" in m["traceback"]
    assert lives[0].torn_down


def test_a_stopped_run_is_interrupted_its_fleet_is_down_and_its_rows_are_kept(
        tmp_path, monkeypatch):
    async def scenario():
        blocked = asyncio.Event()

        class HangsOnSecondWorkload(StubBackend):
            async def complete_stream(self, **kw):
                if self.sent >= 3:           # reps=3: the first workload completes
                    blocked.set()
                    await asyncio.sleep(3600)
                async for chunk in super().complete_stream(**kw):
                    yield chunk

        backend = HangsOnSecondWorkload()
        _, lives = patch_fleet(monkeypatch, lambda dep: backend)

        runner = SweepRunner(build_plan(two_by_two(tmp_path), make_topology()))
        task = asyncio.ensure_future(runner.run())
        await asyncio.wait_for(blocked.wait(), timeout=10)
        runner.request_stop("SIGTERM")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return lives

    lives = asyncio.run(scenario())

    m = manifest(tmp_path)
    assert m["status"] == "interrupted" and m["stop_reason"] == "SIGTERM"
    assert lives and all(live.torn_down for live in lives)
    measured = {r["trial_id"] for r in trials(tmp_path)}
    assert measured == {"d000/w0000"}, "the completed trial must survive the interruption"
    kinds = [e["kind"] for e in events(tmp_path)]
    assert "stop_requested" in kinds and kinds[-1] == "run_end"


def test_a_failure_between_launch_and_measurement_still_tears_the_fleet_down(
        tmp_path, monkeypatch):
    """Writing deployments.json and building the client used to happen *before* the
    try/finally that tears down, so a failure there left every server running."""
    _, lives = patch_fleet(monkeypatch)

    def broken_backend(dep, spec):
        raise ValueError("no backend client")

    monkeypatch.setattr(execute_mod, "make_backend", broken_backend)
    with pytest.raises(ValueError):
        run_sweep(make_spec(tmp_path))
    assert lives[0].torn_down


# --- events.jsonl ---


def test_every_stage_of_the_run_is_in_the_event_log(tmp_path, monkeypatch):
    patch_fleet(monkeypatch)
    runner = run_sweep(make_spec(tmp_path))

    log = events(tmp_path)
    kinds = [e["kind"] for e in log]
    for expected in ("run_start", "deployment_start", "deployment_launched", "trial_start",
                     "trial", "deployment_end", "run_end"):
        assert expected in kinds, f"{expected} missing from events.jsonl: {kinds}"
    assert all(e["run_id"] == runner.run_id and e["ts_utc"] for e in log)
    assert log[-1]["status"] == "finished"


def test_a_warning_attached_to_no_row_still_reaches_the_event_log(tmp_path, monkeypatch):
    """A port that never freed after teardown is on no row; it used to exist only on the
    terminal, which an overnight run's user never sees."""
    class StuckPort(StubLive):
        async def teardown(self, *, settle_s=0.0):
            self.torn_down = True
            return ["port 18000 was still bound 30s after teardown of d000"]

    async def fake_launch(dep, spec, *, out_dir):
        return StuckPort()

    monkeypatch.setattr(execute_mod, "launch_deployment", fake_launch)
    monkeypatch.setattr(execute_mod, "make_backend", lambda dep, spec: StubBackend())
    run_sweep(make_spec(tmp_path))

    warnings = [e["message"] for e in events(tmp_path) if e["kind"] == "warning"]
    assert any("still bound" in w for w in warnings)


# --- resume ---


def test_resume_redoes_only_what_did_not_finish(tmp_path, monkeypatch):
    # Attempt 1: d000 measures cleanly; every request to d001 fails.
    patch_fleet(monkeypatch, lambda dep: StubBackend(fail_after=0 if dep.id == "d001" else -1))
    first = run_sweep(two_by_two(tmp_path))
    statuses = {r["trial_id"]: r["status"] for r in trials(tmp_path)}
    assert statuses == {"d000/w0000": STATUS_OK, "d000/w0001": STATUS_OK,
                        "d001/w0000": STATUS_ERROR, "d001/w0001": STATUS_ERROR}

    # Attempt 2: healthy. d000 is complete and must not even be relaunched.
    launched, _ = patch_fleet(monkeypatch)
    second = run_sweep(two_by_two(tmp_path), resume=True)

    assert launched == ["d001"]
    assert second.run_id == first.run_id and manifest(tmp_path)["attempt"] == 2
    rows = trials(tmp_path)
    assert {r["status"] for r in rows} == {STATUS_OK}, "a superseded error row is still listed"
    assert {r["trial_id"] for r in rows} == set(statuses)
    archived = tmp_path / f"trials.{first.run_id}.attempt1.jsonl"
    # One client row per trial: the stub reports no server timings, so no src=server rows.
    assert archived.exists() and len(archived.read_text().splitlines()) == 4

    # Raw records of the retry must not share a run_id with the failed attempt's records,
    # or `report --from-records` would pool the old failures into the new measurement.
    retry_ids = {json.loads(line)["run_id"]
                 for line in (tmp_path / "records" / "d001.jsonl").read_text().splitlines()}
    assert any("-a2-" in rid for rid in retry_ids)
    assert any("-a2-" not in rid for rid in retry_ids)


def test_resume_refuses_a_spec_that_measures_something_else(tmp_path, monkeypatch):
    patch_fleet(monkeypatch)
    run_sweep(make_spec(tmp_path))

    changed = make_spec(tmp_path, workload={"n_prompt": [64], "n_gen": [0], "reps": 9,
                                            "no_warmup": True})
    with pytest.raises(ResumeError, match="fingerprint"):
        SweepRunner(build_plan(changed, make_topology()), resume=True)


def test_resume_accepts_a_raised_timeout(tmp_path, monkeypatch):
    """The commonest reason to resume: the model was slower to load than startup_timeout_s."""
    patch_fleet(monkeypatch)
    run_sweep(make_spec(tmp_path))
    runner = run_sweep(make_spec(tmp_path, startup_timeout_s=3600), resume=True)
    assert runner.attempt == 2


def test_resume_needs_a_run_to_resume(tmp_path):
    with pytest.raises(ResumeError, match="nothing to resume"):
        SweepRunner(build_plan(make_spec(tmp_path), make_topology()), resume=True)


def test_a_unit_is_complete_only_when_every_row_it_produced_is_a_result():
    def row(tid, status, kind="online", src="client"):
        return TrialResult(trial_id=tid, kind=kind, status=status, src=src)

    rows = [
        row("d000/w0000", STATUS_OK), row("d000/w0000", STATUS_OK, src="server"),
        row("d000/w0001", STATUS_ERROR),
        row("d000/w0002", STATUS_CAPACITY),               # "did not fit" is a result
        row("d000/w0003", STATUS_SKIPPED),                 # dry-run measured nothing
        row("o0000/r0", STATUS_OK, "offline"), row("o0000/r1", STATUS_ERROR, "offline"),
        row("o0001/r0", STATUS_OK, "offline"),
        row("o0002", STATUS_ERROR, "offline"),             # the tool failed outright
    ]
    assert completed_units(rows) == {"d000/w0000", "d000/w0002", "o0001"}


def test_the_fingerprint_tracks_measurements_not_housekeeping(tmp_path):
    def fp(**overrides):
        return build_plan(make_spec(tmp_path, **overrides), make_topology()).fingerprint()

    base = fp()
    assert fp() == base
    assert fp(request_timeout_s=1, settle_s=0, continue_on_error=False, name="x") == base
    assert fp(objective={"metric": "ttft_ms_p99", "goal": "min"}) == base
    assert fp(cpu={"budget": "96-99"}) != base
    assert fp(workload={"n_prompt": [65], "n_gen": [0], "reps": 3, "no_warmup": True}) != base


# --- the ranking refuses contaminated rows ---


def scored(tid, tput, **provenance) -> TrialResult:
    return TrialResult(
        trial_id=tid, kind="online", status=STATUS_OK, backend="llamacpp", test="pp512",
        deployment_id=tid.split("/")[0], axes={"backend": "llamacpp", "instances": 1},
        metrics={"total_token_throughput": tput}, provenance=provenance,
    )


OBJ = ObjectiveSpec(metric="total_token_throughput", goal="max")


def test_a_row_measured_under_contention_cannot_win():
    report = rank([
        scored("d000/w0", 900.0, contention={"foreign_pct": 40.0}),
        scored("d001/w0", 500.0, contention={"foreign_pct": 2.0}),
    ], OBJ)
    assert report.best.trial_id == "d001/w0"
    assert [c.trial_id for c in report.excluded] == ["d000/w0"]
    assert any("excluded from the ranking" in n and "foreign CPU" in n for n in report.notes)


def test_a_row_that_lost_requests_cannot_win_unless_the_loss_is_tolerated():
    rows = [scored("d000/w0", 900.0, n_records=40, n_errors=1), scored("d001/w0", 500.0)]
    assert rank(rows, OBJ).best.trial_id == "d001/w0"

    lenient = ObjectiveSpec(metric="total_token_throughput", goal="max", max_error_pct=5.0)
    assert rank(rows, lenient).best.trial_id == "d000/w0"


def test_a_row_whose_threads_left_their_cores_cannot_win():
    report = rank([
        scored("d000/w0", 900.0, placement={"verified": False, "threads_on_other_cpus": 3}),
        scored("d001/w0", 500.0, placement={"verified": True, "threads_on_other_cpus": 0}),
    ], OBJ)
    assert report.best.trial_id == "d001/w0"


def test_flagged_rows_can_be_ranked_deliberately():
    obj = ObjectiveSpec(metric="total_token_throughput", goal="max", rank_flagged=True)
    report = rank([scored("d000/w0", 900.0, contention={"foreign_pct": 40.0}),
                   scored("d001/w0", 500.0)], obj)
    assert report.best.trial_id == "d000/w0" and not report.excluded
    assert report.best.issues


def test_when_every_row_is_flagged_the_report_says_why_nothing_won():
    report = rank([scored("d000/w0", 900.0, contention={"foreign_pct": 40.0})], OBJ)
    assert report.best is None
    assert any("every successful row was excluded" in n for n in report.notes)


# --- the process ledger ---


def test_nothing_launches_once_the_run_is_stopping(tmp_path):
    """A llama-batched-bench sweep runs its reps as separate invocations from a worker thread;
    killing the current one is not enough if the thread then starts the next."""
    registry = procs.activate(tmp_path / procs.PIDS_FILE, "t")
    try:
        registry.stop()
        with pytest.raises(procs.RunStopping):
            procs.start(["true"], name="next-rep")
    finally:
        procs.deactivate(registry)


def test_cleanup_of_a_directory_with_no_ledger_is_a_no_op(tmp_path):
    assert cli.cmd_cleanup(argparse.Namespace(out_dir=str(tmp_path), dry_run=False)) == 0


@linux_only
def test_the_ledger_records_a_launch_and_forgets_it_after_teardown(tmp_path):
    ledger = tmp_path / procs.PIDS_FILE
    registry = procs.activate(ledger, "t")
    try:
        mp = spawn(["sh", "-c", "sleep 30"], name="srv", log_path=tmp_path / "srv.log",
                   env_overrides={}, port=18123)
        entry = json.loads(ledger.read_text())["processes"][0]
        assert entry["pid"] == mp.pid and entry["port"] == 18123
        assert entry["start_ticks"] == procs.start_ticks(mp.pid)

        mp.terminate(timeout=5.0)
        assert json.loads(ledger.read_text())["processes"] == []
    finally:
        procs.deactivate(registry)


@linux_only
def test_cleanup_stops_what_a_killed_run_left_behind_including_grandchildren(tmp_path):
    ledger = tmp_path / procs.PIDS_FILE
    pid_file = tmp_path / "child.pid"
    registry = procs.activate(ledger, "t")
    mp = spawn(["sh", "-c", f"sleep 120 & echo $! > {pid_file}; wait"], name="srv",
               log_path=tmp_path / "srv.log", env_overrides={})
    procs.deactivate(registry)          # as if the sweep had died without tearing down
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pid_file.read_text().strip())

    assert {lo.pid for lo in procs.inspect(ledger).leftovers} == {mp.pid}
    actions = procs.cleanup(ledger, timeout_s=5.0)
    mp.proc.wait(timeout=5.0)           # reap our own zombie; init would, for a dead sweep

    assert any("stopping srv" in a for a in actions)
    assert procs.start_ticks(child, live_only=True) is None, "the grandchild was orphaned"
    assert not procs.inspect(ledger).leftovers


@linux_only
def test_cleanup_never_signals_a_reused_pid(tmp_path):
    """Same PID, different start time: somebody else's process now. Must be left alone."""
    stranger = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        ledger = tmp_path / procs.PIDS_FILE
        ledger.write_text(json.dumps({"run_id": "old", "owner": {"active": False},
                                      "processes": [{
                                          "pid": stranger.pid, "pgid": stranger.pid,
                                          "name": "srv", "port": None,
                                          "start_ticks": procs.start_ticks(stranger.pid) - 1,
                                      }]}))
        assert not procs.inspect(ledger).leftovers
        procs.cleanup(ledger, timeout_s=1.0)
        assert stranger.poll() is None, "cleanup killed a process that was not ours"
    finally:
        stranger.kill()
        stranger.wait()


@linux_only
def test_a_new_run_refuses_to_start_over_a_dead_run_s_servers(tmp_path):
    orphan = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        (tmp_path / procs.PIDS_FILE).write_text(json.dumps({
            "run_id": "old", "owner": {"active": False},
            "processes": [{"pid": orphan.pid, "pgid": orphan.pid, "name": "d000-llamacpp-i0",
                           "port": 18080, "start_ticks": procs.start_ticks(orphan.pid)}],
        }))
        with pytest.raises(DeploymentError, match="sweep cleanup"):
            SweepRunner(build_plan(make_spec(tmp_path), make_topology()))
    finally:
        orphan.kill()
        orphan.wait()


# --- the CLI turns a signal into an orderly stop ---


@linux_only
def test_sigterm_tears_down_writes_reports_and_exits_130(tmp_path, monkeypatch):
    """SIGTERM used to kill the interpreter with no `finally` run: no teardown, no report,
    run.json still saying `running`."""
    spec_path = tmp_path / "sweep.yaml"
    spec_path.write_text(json.dumps({            # JSON is valid YAML
        "name": "t", "out_dir": str(tmp_path / "out"),
        "cpu": {"budget": "96-103"}, "lb": {"kind": "client"},
        "backends": {"llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server"}},
        **TWO_BY_TWO,
    }))

    class SignalsItselfOnSecondWorkload(StubBackend):
        async def complete_stream(self, **kw):
            if self.sent >= 3:               # reps=3: the first workload completes
                os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.sleep(3600)
            async for chunk in super().complete_stream(**kw):
                yield chunk

    backend = SignalsItselfOnSecondWorkload()
    _, lives = patch_fleet(monkeypatch, lambda dep: backend)
    monkeypatch.setattr(cli.Topology, "detect", staticmethod(make_topology))

    args = argparse.Namespace(spec=str(spec_path), out_dir=None, dry_run=False, mode=None,
                              name=None, resume=False, formats=["md", "json"], quiet=True)
    code = asyncio.run(cli._run(args))

    out = tmp_path / "out"
    assert code == cli.EXIT_INTERRUPTED
    assert lives and all(live.torn_down for live in lives)
    m = json.loads((out / "run.json").read_text())
    assert m["status"] == "interrupted" and m["stop_reason"] == "SIGTERM"
    assert (out / "report.md").exists() and (out / "best.json").exists()
