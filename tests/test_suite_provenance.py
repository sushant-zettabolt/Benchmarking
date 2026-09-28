"""What a result file keeps about how it was produced: every rep, the server command, the host.

And one measurement fix that rides along: vLLM's server-side numbers come from a /metrics
delta, whose baseline has to be taken after the warm-up or the warm-up is counted as a rep.
"""
from __future__ import annotations

import asyncio
import csv
import io

from llmbench import metrics
from llmbench.backends.base import StreamChunk
from llmbench.suite.execute import SweepRunner
from llmbench.suite.plan import build_plan
from llmbench.suite.report import ReportContext, rank, render, render_reps_csv
from llmbench.suite.spec import ObjectiveSpec
from tests.test_suite_execute import ListSink, StubBackend, StubLive, make_spec
from tests.test_suite_topology import make_topology


class FakeProcess:
    def __init__(self, name, argv, env):
        self.name, self.argv, self.env_overrides = name, argv, env
        self.pid = 4242


class LlamaLikeBackend(StubBackend):
    """Reports per-request server timings, as llama-server does."""

    async def complete_stream(self, **kw):
        self.sent += 1
        yield StreamChunk(text="a")
        yield StreamChunk(text="b", usage={"prompt_tokens": 64, "completion_tokens": 2},
                          server_timings={"prompt_n": 64, "prompt_ms": 10.0 * self.sent,
                                          "predicted_n": 2, "predicted_ms": 4.0})


class VllmLikeBackend(StubBackend):
    """Exposes only Prometheus counters. The first request (the warm-up) is ten times slower
    to prefill than the rest, as a cold first request is."""

    name = "vllm"
    supports_vllm_metrics = True

    def __init__(self):
        super().__init__()
        self.done = 0
        self.prefill_s = 0.0

    async def metrics_snapshot(self):
        return {"vllm:request_prefill_time_seconds_sum": self.prefill_s,
                "vllm:request_prefill_time_seconds_count": self.done,
                "vllm:prompt_tokens_total": 64 * self.done}

    async def complete_stream(self, **kw):
        async for chunk in super().complete_stream(**kw):
            yield chunk
        self.done += 1
        self.prefill_s += 10.0 if self.done == 1 else 1.0


def drive(tmp_path, backend, *, warmup: int = 0, processes=()):
    workload = {"n_prompt": [64], "n_gen": [0], "reps": 3}
    workload.update({"warmup_fixed": warmup} if warmup else {"no_warmup": True})
    plan = build_plan(make_spec(tmp_path, workload=workload), make_topology())
    runner = SweepRunner(plan)
    live = StubLive()
    live.processes = list(processes)
    sink = ListSink()
    dep, workloads = plan.groups[0]
    try:
        asyncio.run(runner._run_workload(dep, workloads[0], backend, sink, None, live))
    finally:
        runner.close()
    return runner.results, [r.to_dict() for r in sink.records]


# --- every rep survives, not only the mean ---


def test_each_rep_of_a_trial_is_kept_and_they_average_to_the_reported_mean(tmp_path):
    rows, _ = drive(tmp_path, LlamaLikeBackend())
    client = next(r for r in rows if r.src == "client")

    assert [rep["rep"] for rep in client.reps] == [0, 1, 2]
    tps = [rep["tps"] for rep in client.reps]
    assert all(t > 0 for t in tps)
    assert abs(sum(tps) / 3 - client.metrics["tps_mean"]) < 1e-9


def test_per_request_server_timings_are_kept_per_rep(tmp_path):
    rows, _ = drive(tmp_path, LlamaLikeBackend())
    reps = next(r for r in rows if r.src == "client").reps
    # prompt_ms was 10, 20, 30 ms for 64 tokens: each rep keeps its own value.
    assert [round(rep["server_prefill_tps"]) for rep in reps] == [6400, 3200, 2133]


def test_server_rows_carry_no_rep_list(tmp_path):
    rows, _ = drive(tmp_path, LlamaLikeBackend())
    assert all(not r.reps for r in rows if r.src == "server")


# --- vLLM: the /metrics baseline is taken after the warm-up ---


def test_vllm_server_timings_exclude_the_warm_up_request(tmp_path):
    """The baseline used to be taken before the warm-up, so the slow first request was
    averaged into the measured reps: (10 + 1 + 1 + 1) / 4 s instead of 1 s. That made vLLM's
    server t/s read low and its overhead_ms (client time minus server time) negative."""
    _, records = drive(tmp_path, VllmLikeBackend(), warmup=1)
    last = records[-1]
    assert last["server_prompt_ms"] == 1000.0
    assert metrics.SERVER_TIMINGS_TRIAL_MEAN in last["flags"]


def test_a_trial_wide_server_average_is_not_reported_as_one_rep_s_value(tmp_path):
    rows, _ = drive(tmp_path, VllmLikeBackend(), warmup=1)
    reps = next(r for r in rows if r.src == "client").reps
    assert len(reps) == 3
    assert all(rep["server_tps"] is None and rep["server_prefill_tps"] is None for rep in reps)


# --- the server command and the host go into the reports ---


def _context(rows, manifest=None) -> ReportContext:
    objective = ObjectiveSpec(metric="tps_mean", goal="max", src="client")
    return ReportContext(manifest=manifest or {}, results=rows, ranking=rank(rows, objective),
                         plan={})


def test_the_exact_server_command_reaches_every_report(tmp_path):
    proc = FakeProcess("d000-llamacpp-i0",
                       ["numactl", "--physcpubind=96-103", "--", "/b/llama-server", "-fa", "on"],
                       {"LD_PRELOAD": "/lib/libomp.so.5", "OMP_NUM_THREADS": "8"})
    rows, _ = drive(tmp_path, LlamaLikeBackend(), processes=[proc])
    line = ("LD_PRELOAD=/lib/libomp.so.5 OMP_NUM_THREADS=8 numactl --physcpubind=96-103 -- "
            "/b/llama-server -fa on")

    assert rows[0].provenance["server_commands"][0]["argv"] == proc.argv
    ctx = _context(rows)
    assert line in render(ctx, "md")
    assert line in render(ctx, "html")
    table = list(csv.DictReader(io.StringIO(render(ctx, "csv"))))
    assert all(row["server cmd"] == line for row in table)


def test_report_reps_csv_has_one_line_per_measured_request(tmp_path):
    rows, _ = drive(tmp_path, LlamaLikeBackend(), warmup=1)
    table = list(csv.DictReader(io.StringIO(render_reps_csv(_context(rows)))))
    assert [r["rep"] for r in table] == ["0", "1", "2"]      # the warm-up is not a rep
    assert all(float(r["t/s"]) > 0 and float(r["prefill t/s"]) > 0 for r in table)


def test_the_host_and_software_are_listed_in_the_reports(tmp_path):
    rows, _ = drive(tmp_path, LlamaLikeBackend())
    manifest = {
        "host_info": {"hostname": "turin-pod-5", "cpu_model": "AMD EPYC 9755",
                      "kubernetes": {"in_pod": True, "pod": "turin-pod-5", "namespace": "zendnn"},
                      "cpus_allowed": "160-191", "mems_allowed": "5",
                      "numa_nodes": [{"node": 4, "cpus": "128-159"},
                                     {"node": 5, "cpus": "160-191", "mem_total_gib": 283.4}]},
        "software": {"/b/llama-server": {"type": "llamacpp", "version": "version: 1 (abc123)"}},
    }
    md = render(_context(rows, manifest), "md")
    assert "turin-pod-5 (namespace zendnn)" in md
    assert "160-191" in md and "AMD EPYC 9755" in md
    assert "numa node 5" in md and "numa node 4" not in md      # only the nodes it may use
    assert "version: 1 (abc123)" in md
