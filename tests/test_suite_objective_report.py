"""Objective evaluation, ranking, and report rendering."""
from __future__ import annotations

import json

import pytest

from llmbench.suite.execute import STATUS_ERROR, STATUS_OK, TrialResult
from llmbench.suite.objective import evaluate_constraints, rank
from llmbench.suite.report import ReportContext, render, write_reports
from llmbench.suite.spec import Constraint, ObjectiveSpec


def trial(*, tid, backend="llamacpp", test="pp512", instances=1, cores=96, tput=100.0,
          ttft=1000.0, src="client", status=STATUS_OK, kind="online", **metrics) -> TrialResult:
    return TrialResult(
        trial_id=tid, kind=kind, status=status, src=src, backend=backend, model="m",
        test=test, deployment_id=tid.split("/")[0], workload_id="w0",
        axes={"backend": backend, "instances": instances, "cores_per_instance": cores,
              "n_parallel": 1, "lb": "none"},
        metrics={"total_token_throughput": tput, "ttft_ms_p99": ttft,
                 "tps_mean": tput, **metrics},
    )


OBJ = ObjectiveSpec(metric="total_token_throughput", goal="max", src="client",
                    constraints=[Constraint(metric="ttft_ms_p99", max=2000)])


# --- constraints ---


def test_missing_metric_fails_the_constraint():
    """An unmeasured SLO is not a met SLO. Treating None as passing would promote exactly the
    configs whose latency data is missing because they fell over."""
    verdicts = evaluate_constraints({}, [Constraint(metric="ttft_ms_p99", max=2000)])
    assert verdicts[0].satisfied is False
    assert "was not measured" in verdicts[0].reason


def test_constraint_bounds():
    c = Constraint(metric="x", max=10, min=2)
    assert c.satisfied_by(5)
    assert not c.satisfied_by(11)
    assert not c.satisfied_by(1)
    assert c.describe() == "x >= 2 and x <= 10"


# --- ranking ---


def test_best_is_the_highest_feasible_not_the_highest_overall():
    results = [
        trial(tid="d0/w0", tput=500.0, ttft=9000.0),   # fastest, but violates the SLO
        trial(tid="d1/w0", tput=300.0, ttft=1500.0),   # feasible
    ]
    report = rank(results, OBJ)
    assert report.best.trial_id == "d1/w0"
    assert report.n_feasible == 1 and report.n_infeasible == 1


def test_no_feasible_config_still_reports_the_tradeoff():
    results = [trial(tid="d0/w0", tput=500.0, ttft=9000.0)]
    report = rank(results, OBJ)
    assert report.n_feasible == 0
    assert any("violated at least one constraint" in n for n in report.notes)
    assert report.best is not None          # falls back so the trade-off stays visible
    assert report.pareto


def test_objective_src_filters_out_diagnostic_server_rows():
    results = [
        trial(tid="d0/w0", src="client", tput=100.0, ttft=100.0),
        trial(tid="d0/w0", src="server", tput=999.0, ttft=100.0),
    ]
    report = rank(results, OBJ)
    assert len(report.candidates) == 1
    assert report.best.value == 100.0


def test_failed_trials_are_not_ranked():
    results = [
        trial(tid="d0/w0", tput=100.0, ttft=100.0),
        trial(tid="d1/w0", status=STATUS_ERROR, tput=9999.0, ttft=1.0),
    ]
    assert len(rank(results, OBJ).candidates) == 1


def test_best_per_test_exposes_the_crossover():
    """A single global winner hides that different configs win different workloads -- the
    llama.cpp/vLLM crossover on this project is exactly this shape."""
    results = [
        trial(tid="d0/w0", backend="llamacpp", test="pp16", tput=285.0, ttft=100.0),
        trial(tid="d1/w0", backend="vllm", test="pp16", tput=251.0, ttft=100.0),
        trial(tid="d0/w1", backend="llamacpp", test="pp1024", tput=606.0, ttft=100.0),
        trial(tid="d1/w1", backend="vllm", test="pp1024", tput=1027.0, ttft=100.0),
    ]
    report = rank(results, OBJ)
    assert report.best_per_test["pp16"].backend == "llamacpp"
    assert report.best_per_test["pp1024"].backend == "vllm"


def test_overall_score_normalises_so_big_workloads_do_not_dominate():
    """vllm wins pp1024 by 421 t/s and loses pp16 by 34. Without per-workload normalisation
    the large-magnitude workload would decide the overall ranking on its own."""
    results = [
        trial(tid="d0/w0", backend="llamacpp", test="pp16", tput=285.0, ttft=100.0),
        trial(tid="d0/w1", backend="llamacpp", test="pp1024", tput=606.0, ttft=100.0),
        trial(tid="d1/w0", backend="vllm", instances=2, test="pp16", tput=251.0, ttft=100.0),
        trial(tid="d1/w1", backend="vllm", instances=2, test="pp1024", tput=1027.0, ttft=100.0),
    ]
    report = rank(results, OBJ)
    scores = {s.backend: s.normalised_score for s in report.config_scores}
    # llamacpp: 1.0 on pp16, 606/1027 on pp1024 -> ~0.795
    # vllm:     251/285 on pp16, 1.0 on pp1024  -> ~0.940
    assert scores["vllm"] == pytest.approx((251 / 285 + 1.0) / 2, rel=1e-6)
    assert scores["llamacpp"] == pytest.approx((1.0 + 606 / 1027) / 2, rel=1e-6)
    assert report.best_overall.backend == "vllm"


def test_goal_min_inverts_the_ranking():
    obj = ObjectiveSpec(metric="ttft_ms_p99", goal="min", src="client", constraints=[])
    results = [trial(tid="d0/w0", ttft=500.0), trial(tid="d1/w0", ttft=100.0)]
    assert rank(results, obj).best.trial_id == "d1/w0"


def test_offline_rows_are_ranked_separately_from_online():
    results = [
        trial(tid="d0/w0", kind="online", src="client", tput=100.0, ttft=100.0),
        trial(tid="o0/r0", kind="offline", src="native", tput=9999.0, ttft=100.0),
    ]
    report = rank(results, OBJ)
    # Both are candidates, but the online pool is what gets scored -- an offline number must
    # never beat an online one, they are different measurement boundaries.
    assert report.best.kind == "online"


def test_offline_only_run_says_so():
    results = [trial(tid="o0/r0", kind="offline", src="native", tput=500.0, ttft=100.0)]
    report = rank(results, OBJ)
    assert any("not comparable across backends" in n for n in report.notes)


def test_unknown_metric_name_is_reported_not_silently_empty():
    obj = ObjectiveSpec(metric="no_such_metric", goal="max", src="client")
    report = rank([trial(tid="d0/w0")], obj)
    assert report.best is None
    assert any("no_such_metric" in n for n in report.notes)


def test_pareto_front_keeps_non_dominated_points():
    results = [
        trial(tid="d0/w0", tput=500.0, ttft=1900.0),   # fast, slow ttft
        trial(tid="d1/w0", tput=300.0, ttft=200.0),    # slower, fast ttft
        trial(tid="d2/w0", tput=250.0, ttft=1800.0),   # dominated by both
    ]
    ids = {c.trial_id for c in rank(results, OBJ).pareto}
    assert ids == {"d0/w0", "d1/w0"}


# --- reports ---


@pytest.fixture
def ctx():
    results = [
        trial(tid="d0/w0", test="pp16", tput=285.0, ttft=100.0),
        trial(tid="d0/w1", test="tg64", tput=21.0, ttft=100.0),
        trial(tid="d1/w0", backend="vllm", instances=2, cores=48, test="pp16",
              tput=251.0, ttft=100.0),
        trial(tid="d2/w0", status=STATUS_ERROR, test="pp16"),
        trial(tid="o0/r0", kind="offline", src="native", test="pp512+tg128 @ b4", tput=92.1,
              ttft=None),
    ]
    manifest = {
        "name": "t", "run_id": "t-1", "mode": "both", "elapsed_s": 12.3,
        "started_at_utc": "2026-08-07T00:00:00+00:00",
        "env": {"hostname": "h"},
        "artifacts": {"trials": "trials.jsonl"},
    }
    plan = {"topology": {"model_name": "EPYC", "n_physical": 192, "n_ccds": 24,
                         "cores_per_ccd": 8, "smt_enabled": True}, "warnings": ["w1"]}
    return ReportContext(manifest=manifest, results=results,
                         ranking=rank(results, OBJ), plan=plan)


@pytest.mark.parametrize("fmt", ["md", "csv", "html", "json"])
def test_every_format_renders(ctx, fmt):
    out = render(ctx, fmt)
    assert out and len(out) > 200


def test_markdown_leads_with_the_answer_and_flags_offline(ctx):
    md = render(ctx, "md")
    assert "## Result" in md
    assert md.index("## Result") < md.index("## Online results")
    assert "do not share a measurement boundary" in md
    assert "w1" in md                      # plan warning surfaced


def test_html_is_self_contained(ctx):
    """A strict-CSP or offline viewer must still render it."""
    html = render(ctx, "html")
    for forbidden in ("http://cdn", "https://cdn", "<script src=", "<link rel=\"stylesheet\""):
        assert forbidden not in html
    assert "<style>" in html and "Not comparable across backends" in html


def test_csv_keeps_raw_precision_for_resorting(ctx):
    csv = render(ctx, "csv")
    assert "285.0" in csv
    header = csv.splitlines()[0]
    assert "backend" in header and "test" in header


def test_missing_metrics_render_blank_never_zero(ctx):
    """A 0 in a latency column reads as an extraordinary result rather than missing data."""
    csv = render(ctx, "csv")
    offline_line = next(l for l in csv.splitlines() if "b4" in l)
    assert ",0," not in offline_line
    assert ",," in offline_line


def test_write_reports_emits_every_artifact(ctx, tmp_path):
    written = write_reports(ctx, tmp_path)
    assert set(written) == {"html", "md", "csv", "json", "best"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["best"]["trial_id"] == "d0/w0"
    assert "best_per_test" in best


def test_report_context_roundtrips_from_disk(ctx, tmp_path):
    """`llmbench sweep report <dir>` must be able to rebuild everything from artifacts."""
    from llmbench.suite.report import load_context

    (tmp_path / "trials.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for r in ctx.results))
    (tmp_path / "run.json").write_text(json.dumps({
        **ctx.manifest,
        "objective": {"metric": "total_token_throughput", "goal": "max", "src": "client",
                      "constraints": [{"metric": "ttft_ms_p99", "max": 2000}]},
    }))
    (tmp_path / "plan.json").write_text(json.dumps(ctx.plan))

    reloaded = load_context(tmp_path)
    assert len(reloaded.results) == len(ctx.results)
    assert reloaded.ranking.best.trial_id == ctx.ranking.best.trial_id


# --- report loading is defensive, because the files it reads are written by a live run ---


def test_a_truncated_final_row_does_not_cost_the_report_the_complete_ones(tmp_path):
    """Rows are flushed one at a time so an interrupted sweep keeps its completed trials. The
    corollary is that a hard kill can leave the last line half-written -- losing that line
    must not take the ninety good lines above it with it."""
    from llmbench.suite.report import load_context

    rows = [trial(tid=f"d000/w{i}", tput=100.0 + i) for i in range(3)]
    text = "\n".join(json.dumps(r.to_dict()) for r in rows)
    (tmp_path / "trials.jsonl").write_text(text + '\n{"trial_id": "d000/w3", "ki')
    (tmp_path / "run.json").write_text(json.dumps({"objective": {}}))

    ctx = load_context(tmp_path)
    assert len(ctx.results) == 3
    assert any("could not be parsed" in w for w in ctx.all_warnings())


def test_rows_from_a_different_harness_version_still_load(tmp_path):
    """An out_dir outlives the code that wrote it. An unknown key must not make the whole
    report unrenderable."""
    from llmbench.suite.report import load_context

    row = trial(tid="d000/w0").to_dict()
    row["some_future_field"] = {"added": "later"}
    row.pop("duration_s")
    (tmp_path / "trials.jsonl").write_text(json.dumps(row))
    (tmp_path / "run.json").write_text(json.dumps({"objective": {}}))

    ctx = load_context(tmp_path)
    assert len(ctx.results) == 1
    assert ctx.results[0].trial_id == "d000/w0"


def test_constraint_verdicts_carry_their_pass_fail_class(ctx):
    """The colouring used to be bolted on by replacing '>pass<' in the finished table, which
    produced a second class= attribute on a cell that already had one. Browsers keep the
    first, so the verdict column silently rendered unstyled."""
    html = render(ctx, "html")
    assert 'class="fail">FAIL<' in html or 'class="pass">pass<' in html
    assert 'class="" class=' not in html
