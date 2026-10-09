"""Measurement conditions the placement checks cannot see (2026-09-30, from the Turin Qwen3.6 work):
where a server's memory really is (deploy.verify_memory_placement), the cores' clock
(CoreSampler mhz, provenance.clock_mhz) and real-text prompts (workload.prompt_text)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmbench.prompts import text_prompt_tokens
from llmbench.suite import coresampler as coresampler_mod
from llmbench.suite import deploy as deploy_mod
from llmbench.suite.coresampler import CoreSampler
from llmbench.suite.execute import STATUS_OK, TrialResult
from llmbench.suite.objective import rank
from llmbench.suite.plan import build_plan
from llmbench.suite.report.common import used_columns
from llmbench.suite.spec import ObjectiveSpec, SpecError
from tests.test_suite_execute import make_spec
from tests.test_suite_topology import make_topology

NUMA_MAPS = """\
7f0000000000 default file=/m/x.gguf mapped=1000 mapmax=2 N1=200 N5=800 kernelpagesize_kB=4
7f1000000000 default anon=10 dirty=10 active=0 N1=10 kernelpagesize_kB=2048
7f2000000000 default stack anon=3 dirty=3 N1=3 kernelpagesize_kB=4
7f3000000000 default file=/lib/libc.so.6
"""


def test_numa_maps_are_summed_per_node_with_each_mappings_page_size():
    assert deploy_mod.parse_numa_maps(NUMA_MAPS) == {1: 200 * 4 + 10 * 2048 + 3 * 4, 5: 800 * 4}


def _proc(membind, per_node_kb, monkeypatch):
    monkeypatch.setattr(deploy_mod, "_numa_resident_kb", lambda pid: per_node_kb)
    from llmbench.suite import contention
    monkeypatch.setattr(contention, "_descendant_pids", lambda pid: [pid])
    return SimpleNamespace(alive=True, pid=4242, cores=SimpleNamespace(membind=membind))


def test_a_server_reading_its_weights_from_another_node_is_flagged(monkeypatch):
    # the Turin case: 70 GB of mmap'd GGUF page cache on node 5, the server bound to node 1
    out = deploy_mod.verify_memory_placement(_proc([1], {1: 5 << 20, 5: 70 << 20}, monkeypatch))
    assert out["verified"] is False
    assert out["local_fraction"] == pytest.approx(5 / 75, abs=1e-3)
    assert out["per_node_gb"] == {"1": 5.0, "5": 70.0}
    assert "--load-mode" in out["error"]


def test_local_memory_passes(monkeypatch):
    out = deploy_mod.verify_memory_placement(_proc([1], {1: 70 << 20, 0: 1 << 20}, monkeypatch))
    assert out["verified"] is True and "error" not in out


def test_no_membind_means_nothing_to_check(monkeypatch):
    out = deploy_mod.verify_memory_placement(_proc([], {0: 1 << 20, 5: 1 << 20}, monkeypatch))
    assert out["verified"] is True and "local_fraction" not in out


def _row(tid, tput, **prov):
    return TrialResult(
        trial_id=tid, kind="online", status=STATUS_OK, backend="llamacpp", test="pp512",
        deployment_id=tid.split("/")[0], axes={"backend": "llamacpp", "instances": 1},
        metrics={"total_token_throughput": tput}, provenance=prov,
    )


def test_a_row_with_remote_weights_cannot_win():
    report = rank([
        _row("d000/w0", 900.0, placement={"memory_local_fraction": 0.1}),
        _row("d001/w0", 500.0, placement={"memory_local_fraction": 0.99}),
    ], ObjectiveSpec(metric="total_token_throughput", goal="max"))
    assert report.best.trial_id == "d001/w0"
    assert [c.trial_id for c in report.excluded] == ["d000/w0"]


def test_clock_and_memory_locality_become_report_columns_only_when_present():
    with_prov = _row("d000/w0", 1.0, clock_mhz=3100, placement={"memory_local_fraction": 0.98})
    headers = [h for h, _, _ in used_columns([with_prov])]
    assert "MHz" in headers and "mem local" in headers
    assert "MHz" not in [h for h, _, _ in used_columns([_row("d000/w0", 1.0)])]


def test_the_sampler_records_the_clock_and_averages_it_over_the_measured_requests(monkeypatch):
    clocks = iter([2000.0, 3000.0, 4000.0])
    monkeypatch.setattr(coresampler_mod, "_read_avg_mhz", lambda cpus: next(clocks))
    s = CoreSampler([0], pids=[], path="/dev/null")
    s._write = lambda rows, rep_windows: None
    s.begin_trial("d000/w0000", "pp16")
    for t0, t1 in ((0, 100), (100, 200), (200, 300)):     # warm-up row, then two in rep 0
        row = s._row({0: (0,) * 8}, {0: (1, 0, 0, 1, 0, 0, 0, 0)}, t0, t1, 0, 0)
        row["trial"], row["test"] = s._label
        s._buf.append(row)
    assert [r["mhz"] for r in s._buf] == [2000.0, 3000.0, 4000.0]
    assert s.end_trial([(0, 120, 300)]) == pytest.approx(3500.0)


def test_text_prompts_are_unique_per_request_and_wrap_the_text():
    text = list(range(1000, 1010))                         # a 10-token "text"
    a = text_prompt_tokens(n_prompt=25, text_ids=text, shared_prefix_n=0, run_salt=7,
                           request_idx=1, vocab_size=50000, bos_token_id=1)
    b = text_prompt_tokens(n_prompt=25, text_ids=text, shared_prefix_n=0, run_salt=7,
                           request_idx=2, vocab_size=50000, bos_token_id=1)
    assert len(a) == len(b) == 25 and a[0] == b[0] == 1
    assert a[1:3] != b[1:3]                                # salt markers differ per request
    assert set(a[3:]) <= set(text)                         # the rest is text, wrapped
    shared = text_prompt_tokens(n_prompt=8, text_ids=text, shared_prefix_n=4, run_salt=7,
                                request_idx=3, vocab_size=50000)
    assert shared[:4] == text[:4]


def test_prompt_text_enters_the_fingerprint_only_when_set(tmp_path):
    def fp(**workload):
        wl = {"n_prompt": [64], "n_gen": [0], "reps": 3, "no_warmup": True, **workload}
        return build_plan(make_spec(tmp_path, workload=wl), make_topology()).fingerprint()

    f1, f2 = tmp_path / "a.txt", tmp_path / "b.txt"
    f1.write_text("one text"); f2.write_text("another text")
    base = fp()
    assert fp(prompt_text="") == base                     # unset: plans from before keep theirs
    assert fp(prompt_text=str(f1)) != base
    assert fp(prompt_text=str(f1)) != fp(prompt_text=str(f2))
    with pytest.raises(SpecError):
        fp(prompt_text=str(tmp_path / "missing.txt"))
