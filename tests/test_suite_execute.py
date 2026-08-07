"""How the sweep runner records what happened to a trial, and to an out_dir.

These exercise SweepRunner directly against a fake Backend rather than a live server. The
properties under test are about bookkeeping -- what status a row gets, which warnings ride
along with it, which file it lands in -- and none of them depend on real inference.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


from llmbench.backends.base import Backend, Capacity, ServerInfo, StreamChunk
from llmbench.suite.execute import STATUS_ERROR, STATUS_OK, STATUS_SKIPPED, SweepRunner
from llmbench.suite.plan import build_plan
from llmbench.suite.spec import SuiteSpec
from tests.test_suite_topology import make_topology


class StubBackend(Backend):
    """Answers every probe; `fail_after` requests start erroring."""

    name = "llamacpp"

    def __init__(self, *, fail_after: int = -1):
        super().__init__("http://127.0.0.1:1")
        self.fail_after = fail_after
        self.sent = 0

    async def detect(self): return True
    async def info(self): return ServerInfo(backend="llamacpp", model="m", version="test")
    async def capacity(self):
        return Capacity(per_request_ctx=4096, max_concurrent=8, kv_bytes=1000)
    async def tokenize(self, text): return [1]
    async def detokenize(self, ids): return "x"
    async def supports_token_id_prompts(self): return True
    async def metrics_snapshot(self): return {}
    async def close(self): pass

    async def complete_stream(self, **kw):
        self.sent += 1
        if 0 <= self.fail_after < self.sent:
            raise ConnectionError("upstream refused the connection")
        yield StreamChunk(text="a")
        yield StreamChunk(text="b", usage={"prompt_tokens": 64, "completion_tokens": 2})


class ListSink:
    def __init__(self): self.records: list[Any] = []
    def write(self, rec): self.records.append(rec)


class StubLive:
    """Stands in for a LiveDeployment, so a deployment can be driven without launching one."""

    def __init__(self):
        self.processes: list = []
        self.lb_process = None
        self.affinity: list[dict] = []
        self.torn_down = False

    def to_dict(self): return {"processes": []}

    async def teardown(self, *, settle_s: float = 0.0) -> list[str]:
        self.torn_down = True
        return []


def make_spec(tmp_path, **overrides) -> SuiteSpec:
    data = {
        "name": "t",
        "out_dir": str(tmp_path),
        "cpu": {"budget": "96-103"},
        "lb": {"kind": "client"},
        "backends": {"llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server"}},
        "deployment": {"backend": ["llamacpp"], "instances": [1]},
        "workload": {"n_prompt": [64], "n_gen": [0], "reps": 3, "no_warmup": True},
    }
    data.update(overrides)
    return SuiteSpec.from_dict(data)


def drive(tmp_path, backend: StubBackend):
    """Run the single planned workload and return the rows it emitted."""
    spec = make_spec(tmp_path)
    plan = build_plan(spec, make_topology())
    runner = SweepRunner(plan)
    dep, workloads = plan.groups[0]
    try:
        asyncio.run(runner._run_workload(
            dep, workloads[0], backend, ListSink(), None, StubLive(),
        ))
    finally:
        runner.close()
    return runner.results


# --- trial status reflects what happened to the requests ---


def test_a_trial_whose_every_request_failed_is_not_reported_as_a_success(tmp_path):
    """A failed request still produces a record, and aggregate_client still produces a row
    from it -- one whose tps_mean is 0.0 because there was nothing to average. Emitting that
    as `ok` puts a plausible-looking zero into the report and into the ranking, where it is
    indistinguishable from a configuration that really is that slow."""
    rows = drive(tmp_path, StubBackend(fail_after=0))

    assert rows and all(r.status == STATUS_ERROR for r in rows)
    assert "all 3 request(s) failed" in rows[0].error
    assert "ConnectionError" in rows[0].error
    assert rows[0].provenance["n_errors"] == 3


def test_a_partially_failed_trial_is_measured_but_says_how_many_were_lost(tmp_path):
    """Some failures are a real property of the configuration under test, so the surviving
    requests are still worth aggregating -- but the row must not imply it measured all of
    them."""
    rows = drive(tmp_path, StubBackend(fail_after=2))

    assert rows[0].status == STATUS_OK
    assert any("of 3 request(s) failed" in w for w in rows[0].warnings)
    assert rows[0].provenance["n_errors"] == 1


def test_a_clean_trial_is_ok_and_carries_no_failure_warning(tmp_path):
    rows = drive(tmp_path, StubBackend())

    assert rows[0].status == STATUS_OK
    assert rows[0].error is None
    assert not any("failed" in w for w in rows[0].warnings)
    assert rows[0].metrics["tps_mean"] > 0


def test_every_row_records_which_run_produced_it(tmp_path):
    rows = drive(tmp_path, StubBackend())
    assert rows[0].run_id.startswith("t-")


# --- out_dir hygiene ---


def test_a_second_run_does_not_append_to_the_first_run_s_rows(tmp_path):
    """Rows are appended as they are measured so an interrupted sweep keeps its completed
    trials. The cost is that a second run pointed at the same out_dir would append to the
    first run's file, and the report -- rebuilt from trials.jsonl plus the *new* run.json --
    would present two different sweeps as one table."""
    drive(tmp_path, StubBackend())
    first = (tmp_path / "trials.jsonl").read_text().splitlines()
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "earlier-run"}))

    drive(tmp_path, StubBackend())
    second = (tmp_path / "trials.jsonl").read_text().splitlines()

    assert len(second) == len(first)
    archived = (tmp_path / "trials.earlier-run.jsonl")
    assert archived.exists() and archived.read_text().splitlines() == first


# --- progress accounting ---


def test_skipped_deployments_still_count_as_planned_work(tmp_path):
    """The progress counter counts planned units of work. Staying silent when a deployment
    fails to launch left it permanently short of its own total."""
    spec = make_spec(tmp_path, dry_run=True)
    plan = build_plan(spec, make_topology())
    events: list[tuple[str, dict]] = []
    runner = SweepRunner(plan, on_event=lambda k, p: events.append((k, p)))
    try:
        asyncio.run(runner.run())
    finally:
        pass

    starts = [p for k, p in events if k == "trial_start"]
    assert len(starts) == plan.n_online_trials + plan.n_offline_trials
    assert all(r.status == STATUS_SKIPPED for r in runner.results)


def drive_deployment(tmp_path, backend: StubBackend, monkeypatch) -> SweepRunner:
    """Run the full deployment path with the launch and teardown stubbed out."""
    from llmbench.suite import execute as execute_mod

    live = StubLive()
    async def fake_launch(dep, spec, *, out_dir): return live
    monkeypatch.setattr(execute_mod, "launch_deployment", fake_launch)
    monkeypatch.setattr(execute_mod, "make_backend", lambda dep, spec: backend)

    spec = make_spec(tmp_path)
    runner = SweepRunner(build_plan(spec, make_topology()), on_event=lambda k, p: None)
    asyncio.run(runner.run())
    assert live.torn_down, "the deployment must be torn down whatever happened inside it"
    return runner


def test_a_backend_that_will_not_report_its_version_does_not_abort_the_sweep(
        tmp_path, monkeypatch):
    """`info()` is a version banner and nothing depends on it. Letting it propagate skipped
    every remaining workload in the deployment and, with continue_on_error off, the rest of
    the sweep -- and `continue_on_error` never got a say, because the failure happened outside
    the per-trial handler."""
    class NoInfo(StubBackend):
        async def info(self):
            raise RuntimeError("/props not available")

    runner = drive_deployment(tmp_path, NoInfo(), monkeypatch)

    assert runner.results, "the workload must still have been measured"
    assert all(r.status == STATUS_OK for r in runner.results)
    assert runner.results[0].provenance["backend_version"] is None


def test_a_healthy_deployment_records_the_backend_version(tmp_path, monkeypatch):
    runner = drive_deployment(tmp_path, StubBackend(), monkeypatch)
    assert runner.results[0].provenance["backend_version"] == "test"


def test_the_recorded_thread_count_is_the_one_the_server_is_launched_with(tmp_path):
    """These two are computed in different modules and used to disagree: the launcher honours
    `threads_per_instance` for `-t`, the record used the count of granted cpus. Every raw
    record then carried a thread count no server was ever started with."""
    from llmbench.suite.deploy import build_server_command
    from llmbench.suite.execute import _threads_of

    for requested, expected in ((None, 8), (4, 4)):
        deployment = {"backend": ["llamacpp"], "instances": [1]}
        if requested is not None:
            deployment["threads_per_instance"] = [requested]
        spec = make_spec(tmp_path, cpu={"budget": "96-103"}, deployment=deployment)
        dep, _ = build_plan(spec, make_topology()).groups[0]

        argv, _env = build_server_command(dep, dep.instances[0], spec.cpu)
        launched = int(argv[argv.index("-t") + 1])
        assert launched == expected
        assert _threads_of(dep) == launched


# --- one backend at a time ---


def multi_backend_spec(tmp_path) -> SuiteSpec:
    return SuiteSpec.from_dict({
        "name": "t", "out_dir": str(tmp_path), "mode": "both",
        "cpu": {"budget": "96-103"}, "lb": {"kind": "client"},
        "backends": {
            "llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server",
                         "offline_bin": "/b/llama-bench"},
            "vllm": {"model": "/m/hf", "server_bin": "/b/vllm", "offline_bin": "/b/vllm"},
        },
        "deployment": {"backend": ["llamacpp", "vllm"], "instances": [1, 2]},
        "workload": {"n_prompt": [64], "n_gen": [0], "reps": 2, "no_warmup": True},
        "offline": {"batch_size": [1], "n_prompt": [64], "n_gen": [32], "reps": 1},
    })


def record_execution_order(tmp_path, monkeypatch) -> list[tuple[str, str]]:
    """Drive a two-backend sweep, logging every launch, teardown and offline invocation."""
    from llmbench.suite import execute as execute_mod

    events: list[tuple[str, str]] = []

    class TrackingLive(StubLive):
        def __init__(self, dep_id):
            super().__init__()
            self.dep_id = dep_id

        async def teardown(self, *, settle_s=0.0, port_timeout_s=0.0):
            events.append(("teardown", self.dep_id))
            return []

    async def fake_launch(dep, spec, *, out_dir):
        events.append(("launch", dep.id))
        return TrackingLive(dep.id)

    def fake_offline(o, cpu, *, log_dir, timeout_s):
        events.append(("offline", o.id))
        return []

    monkeypatch.setattr(execute_mod, "launch_deployment", fake_launch)
    monkeypatch.setattr(execute_mod, "make_backend", lambda dep, spec: StubBackend())
    monkeypatch.setattr(execute_mod, "run_offline", fake_offline)

    spec = multi_backend_spec(tmp_path)
    runner = SweepRunner(build_plan(spec, make_topology()), on_event=lambda k, p: None)
    asyncio.run(runner.run())
    return events


def test_only_one_backend_is_ever_live_at_a_time(tmp_path, monkeypatch):
    """Two backends resident at once contend for cores, memory bandwidth and page cache, and
    each one's numbers then describe the pair rather than either. `backend` is a deployment
    axis and deployments are strictly sequential -- but nothing enforced that, so this pins it
    against a future refactor that decides launching the next fleet early would save time."""
    events = record_execution_order(tmp_path, monkeypatch)

    live = 0
    for kind, _ in events:
        if kind == "launch":
            live += 1
        elif kind == "teardown":
            live -= 1
        assert live <= 1, f"two deployments were live simultaneously: {events}"
    assert live == 0, "a deployment was left running at the end of the sweep"


def test_each_deployment_is_torn_down_before_the_next_one_launches(tmp_path, monkeypatch):
    events = record_execution_order(tmp_path, monkeypatch)
    deployments = [e for e in events if e[0] in ("launch", "teardown")]

    assert [k for k, _ in deployments] == ["launch", "teardown"] * (len(deployments) // 2)
    for (_, launched), (_, torn) in zip(deployments[::2], deployments[1::2]):
        assert launched == torn, f"{launched} was not the deployment torn down next"


def test_offline_tools_run_only_after_every_fleet_is_down(tmp_path, monkeypatch):
    """The native tools take the whole core budget as one process. Overlapping them with a
    live server would have each measuring the other."""
    events = record_execution_order(tmp_path, monkeypatch)

    first_offline = next(i for i, (k, _) in enumerate(events) if k == "offline")
    assert not any(k == "launch" for k, _ in events[first_offline:])
    assert events[first_offline - 1][0] == "teardown"


def test_a_deployment_is_torn_down_even_when_its_workload_explodes(tmp_path, monkeypatch):
    """The dangerous failure is a fleet left running: it holds its cores and its memory for
    the rest of the sweep, so every later row measures two backends instead of one."""
    from llmbench.suite import execute as execute_mod

    live = StubLive()

    async def fake_launch(dep, spec, *, out_dir): return live

    class Exploding(StubBackend):
        async def capacity(self):
            raise RuntimeError("backend fell over")

    monkeypatch.setattr(execute_mod, "launch_deployment", fake_launch)
    monkeypatch.setattr(execute_mod, "make_backend", lambda dep, spec: Exploding())

    spec = make_spec(tmp_path)
    runner = SweepRunner(build_plan(spec, make_topology()), on_event=lambda k, p: None)
    asyncio.run(runner.run())

    assert live.torn_down, "the fleet was left running after the workload failed"
    assert all(r.status == STATUS_ERROR for r in runner.results)
