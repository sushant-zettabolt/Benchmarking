"""Per-core CPU time series (llmbench/suite/coresampler.py) and its wiring into the runner."""
from __future__ import annotations

import csv
import os
import time

from llmbench.suite.coresampler import CoreSampler
from tests.test_suite_execute import StubBackend, drive_deployment


def _fields(user=0, system=0, idle=0, iowait=0, irq=0, softirq=0, steal=0, nice=0):
    # /proc/stat order: user nice system idle iowait irq softirq steal
    return (user, nice, system, idle, iowait, irq, softirq, steal)


def test_each_core_gets_its_own_busy_percentage_and_the_split_is_aggregated():
    s = CoreSampler([0, 1], pids=[], path="/dev/null")
    prev = {0: _fields(), 1: _fields()}
    cur = {0: _fields(user=20, idle=5),              # 80% busy
           1: _fields(system=5, idle=15, iowait=5)}  # 20% busy; iowait counts as idle
    row = s._row(prev, cur, 0, 250_000_000, our_ticks=0, harness_ticks=0)

    assert row["per_cpu"] == {0: 80.0, 1: 20.0}
    assert row["busy_pct"] == 50.0                  # 25 busy ticks of 50
    assert row["user_pct"] == 40.0 and row["system_pct"] == 10.0 and row["iowait_pct"] == 10.0


def test_load_is_split_into_servers_harness_and_foreign():
    s = CoreSampler([0, 1], pids=[], path="/dev/null")
    prev = {0: _fields(), 1: _fields()}
    cur = {0: _fields(user=100), 1: _fields(user=50, idle=50)}   # 150 busy ticks in 1 s
    row = s._row(prev, cur, 0, 1_000_000_000, our_ticks=100, harness_ticks=20)

    assert round(row["busy_cores"], 2) == 1.5
    assert round(row["our_cores"], 2) == 1.0
    assert round(row["harness_cores"], 2) == 0.2
    assert round(row["foreign_cores"], 2) == 0.3


def test_rows_are_labelled_with_the_phase_of_the_trial_they_fall_in():
    windows = [(0, 100, 200), (1, 300, 400)]
    assert CoreSampler._phase(50, "", windows) == "idle"
    assert CoreSampler._phase(50, "d000/w0000", windows) == "warmup"
    assert CoreSampler._phase(150, "d000/w0000", windows) == "rep0"
    assert CoreSampler._phase(250, "d000/w0000", windows) == "between"
    assert CoreSampler._phase(350, "d000/w0000", windows) == "rep1"
    assert CoreSampler._phase(50, "d000/w0000", []) == "warmup"   # the trial measured nothing


def test_a_real_sampling_run_writes_one_column_per_core(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))[:4]
    path = tmp_path / "cores" / "d000.csv"
    s = CoreSampler(cpus, pids=[os.getpid()], path=path, interval_s=0.05)
    s.start()
    time.sleep(0.15)                                 # idle before the trial
    s.begin_trial("d000/w0000", "pp16")
    t_send = time.perf_counter_ns()
    end = time.monotonic() + 0.3
    while time.monotonic() < end:                    # a "request" that burns cpu
        pass
    s.end_trial([(0, t_send, time.perf_counter_ns())])
    time.sleep(0.1)
    s.stop()

    rows = list(csv.DictReader(path.open()))
    assert rows, "no samples were written"
    assert all(f"cpu{c}" in rows[0] for c in cpus)
    phases = {r["phase"] for r in rows}
    assert "rep0" in phases and "idle" in phases
    assert all(r["trial"] == "d000/w0000" for r in rows if r["phase"] == "rep0")
    assert all(r["test"] == "" for r in rows if r["phase"] == "idle")


class FakeSampler:
    instances: list["FakeSampler"] = []

    def __init__(self, cpus, pids, path, *, interval_s):
        self.cpus, self.path, self.interval_s = list(cpus), path, interval_s
        self.calls: list[tuple] = []
        FakeSampler.instances.append(self)

    def start(self): self.calls.append(("start",))
    def begin_trial(self, trial_id, test): self.calls.append(("begin", trial_id, test))
    def end_trial(self, windows): self.calls.append(("end", len(windows)))
    def stop(self): self.calls.append(("stop",))


def test_the_runner_samples_each_deployment_and_labels_every_trial(tmp_path, monkeypatch):
    from llmbench.suite import execute as execute_mod

    FakeSampler.instances = []
    monkeypatch.setattr(execute_mod, "CoreSampler", FakeSampler)
    runner = drive_deployment(tmp_path, StubBackend(), monkeypatch)

    (sampler,) = FakeSampler.instances
    assert sampler.path == tmp_path / "cores" / "d000.csv"
    assert sampler.interval_s == 0.25
    assert sampler.cpus == list(range(96, 104))
    assert [c[0] for c in sampler.calls] == ["start", "begin", "end", "stop"]
    assert sampler.calls[2] == ("end", 3)             # the three measured reps' windows
    assert runner.results[0].provenance["core_timeseries"] == "cores/d000.csv"


def test_the_sampler_can_be_turned_off(tmp_path, monkeypatch):
    from llmbench.suite import execute as execute_mod
    from tests import test_suite_execute as te

    FakeSampler.instances = []
    monkeypatch.setattr(execute_mod, "CoreSampler", FakeSampler)
    real = te.make_spec
    monkeypatch.setattr(te, "make_spec",
                        lambda tmp, **kw: real(tmp, core_sample_interval_s=0, **kw))
    runner = drive_deployment(tmp_path, StubBackend(), monkeypatch)

    assert FakeSampler.instances == []
    assert runner.results[0].provenance["core_timeseries"] is None
