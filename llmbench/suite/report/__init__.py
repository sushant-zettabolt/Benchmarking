"""Report generation. HTML, Markdown and CSV all render the same rows, from the same context.

Reports are always regenerable from the artifacts on disk (`trials.jsonl`, `run.json`,
`plan.json`), so `llmbench sweep report <dir>` can rebuild them for a run that has already
finished -- or one that was interrupted.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ..execute import TrialResult
from ..objective import RankingReport, rank
from ..spec import Constraint, ObjectiveSpec
from .common import ReportContext
from .csv import render_csv
from .html import render_html
from .markdown import render_markdown

FORMATS = ("html", "md", "csv", "json")


def _load_trials(path: Path) -> tuple[list[TrialResult], int]:
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
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            rows.append(TrialResult(**{k: v for k, v in obj.items() if k in known}))
        except (ValueError, TypeError):
            skipped += 1
    return rows, skipped


def reaggregate(out_dir: Path, results: list[TrialResult]) -> tuple[list[TrialResult], int]:
    """Recompute every online row's statistics from the raw per-request records.

    `trials.jsonl` stores each row's metrics already flattened, so re-rendering a report
    normally replays those numbers rather than recomputing them. That is fine for a formatting
    change and wrong for anything else: a fix or an addition in `metrics.py` would silently not
    apply to a finished run, even though the per-request records it needs are sitting in
    `records/`. This walks those records instead, so an old run picks up new arithmetic.

    Rows are matched by `provenance.run_id`, which execute.py sets per (deployment, workload)
    -- the same value written into every RawRecord.run_id. A row whose records are missing is
    left exactly as it was rather than blanked.
    """
    from ... import metrics as metrics_mod
    from ..execute import result_row_to_metrics

    by_run: dict[str, list[dict]] = {}
    for path in sorted((out_dir / "records").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            by_run.setdefault(rec.get("run_id", ""), []).append(rec)

    updated = 0
    for row in results:
        if row.kind != "online":
            continue
        raw = by_run.get((row.provenance or {}).get("run_id", ""))
        if not raw:
            continue
        recomputed = (metrics_mod.aggregate_client(raw, row.test) if row.src == "client"
                      else metrics_mod.aggregate_server(raw, row.test))
        if recomputed is None:
            continue
        row.metrics = result_row_to_metrics(recomputed)
        updated += 1
    return results, updated


def load_context(out_dir: str | Path, *, from_records: bool = False) -> ReportContext:
    """Rebuild a report context from a run directory."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "run.json"
    trials_path = out_dir / "trials.jsonl"
    if not trials_path.exists():
        raise FileNotFoundError(f"no trials.jsonl under {out_dir}")

    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    plan: dict[str, Any] = {}
    plan_path = out_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())

    results, skipped = _load_trials(trials_path)

    obj_data = manifest.get("objective") or {}
    objective = ObjectiveSpec(
        metric=obj_data.get("metric", "total_token_throughput"),
        goal=obj_data.get("goal", "max"),
        src=obj_data.get("src", "client"),
        constraints=[
            Constraint(metric=c["metric"], max=c.get("max"), min=c.get("min"))
            for c in obj_data.get("constraints", [])
        ],
    )
    n_reaggregated = 0
    if from_records:
        results, n_reaggregated = reaggregate(out_dir, results)

    ctx = ReportContext(
        manifest=manifest, results=results, ranking=rank(results, objective), plan=plan,
    )
    if from_records:
        plan.setdefault("warnings", []).append(
            f"statistics for {n_reaggregated} online row(s) were recomputed from the raw "
            f"per-request records in records/ rather than replayed from trials.jsonl"
            if n_reaggregated else
            "--from-records was requested but no raw records matched any row; the numbers "
            "shown are the ones stored in trials.jsonl"
        )
    if skipped:
        plan.setdefault("warnings", []).append(
            f"{skipped} line(s) in {trials_path.name} could not be parsed and are missing from "
            f"this report; the usual cause is a run killed mid-write"
        )
    return ctx


def render(ctx: ReportContext, fmt: str) -> str:
    if fmt == "html":
        return render_html(ctx)
    if fmt == "md":
        return render_markdown(ctx)
    if fmt == "csv":
        return render_csv(ctx)
    if fmt == "json":
        return json.dumps({
            "manifest": ctx.manifest,
            "ranking": ctx.ranking.to_dict(),
            "results": [r.to_dict() for r in ctx.results],
        }, indent=2, default=str)
    raise ValueError(f"unknown report format {fmt!r}; known: {list(FORMATS)}")


def write_reports(ctx: ReportContext, out_dir: str | Path,
                  formats: tuple[str, ...] = FORMATS) -> dict[str, Path]:
    """Write every requested format plus `best.json`, the machine-readable answer."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        ext = {"html": "html", "md": "md", "csv": "csv", "json": "json"}[fmt]
        path = out_dir / f"report.{ext}"
        path.write_text(render(ctx, fmt))
        written[fmt] = path

    best_path = out_dir / "best.json"
    best_path.write_text(json.dumps({
        "objective": ctx.ranking.objective,
        "headline": ctx.ranking.headline(),
        "best": ctx.ranking.best.to_dict() if ctx.ranking.best else None,
        "best_overall": (
            ctx.ranking.best_overall.to_dict() if ctx.ranking.best_overall else None
        ),
        "best_per_test": {k: v.to_dict() for k, v in ctx.ranking.best_per_test.items()},
        "notes": ctx.ranking.notes,
    }, indent=2, default=str))
    written["best"] = best_path
    return written


__all__ = ["load_context", "reaggregate", "render", "write_reports", "ReportContext", "FORMATS",
           "RankingReport", "rank"]
