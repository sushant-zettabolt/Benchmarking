"""Markdown report: the answer first, then the evidence, then the caveats."""
from __future__ import annotations


from .common import ReportContext, rows_for, used_columns


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    numeric = [
        all(_is_num(row[i]) for row in rows if row[i]) for i in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        cells = [
            v.rjust(widths[i]) if numeric[i] else v.ljust(widths[i])
            for i, v in enumerate(values)
        ]
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join(
        ("-" * (widths[i] - 1) + ":") if numeric[i] else ("-" * widths[i])
        for i in range(len(headers))
    ) + " |"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)]) + "\n"


def _is_num(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def render_markdown(ctx: ReportContext) -> str:
    m = ctx.manifest
    rank = ctx.ranking
    out: list[str] = []
    a = out.append

    a(f"# {m.get('name', 'sweep')}\n")
    a(f"- run id: `{m.get('run_id')}`")
    a(f"- started: {m.get('started_at_utc')}  (elapsed {m.get('elapsed_s')}s)")
    a(f"- mode: `{m.get('mode')}`")
    a(f"- host: {m.get('env', {}).get('hostname')} — {ctx.plan.get('topology', {}).get('model_name')}")
    topo = ctx.plan.get("topology", {})
    a(f"- topology: {topo.get('n_physical')} physical cores, {topo.get('n_ccds')} CCDs "
      f"of {topo.get('cores_per_ccd')}, SMT {'on' if topo.get('smt_enabled') else 'off'}")
    counts = ctx.status_counts()
    a(f"- rows: {len(ctx.results)} ({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})\n")

    # ---- the answer ----
    a("## Result\n")
    a(f"**Objective:** {rank.objective}\n")
    a(f"**{rank.headline()}**\n")

    if rank.best is not None:
        a("### Best configuration\n")
        for k, v in rank.best.axes.items():
            if v is not None:
                a(f"- `{k}`: {v}")
        a("")
        if rank.best.verdicts:
            a("Constraint check for this configuration:\n")
            a(_table(["constraint", "value", "verdict"], [
                [v.described, ("" if v.value is None else f"{v.value:.4g}"),
                 "pass" if v.satisfied else "FAIL"]
                for v in rank.best.verdicts
            ]))

    if rank.best_per_test:
        a("### Best configuration per workload\n")
        a("A single winner hides real trade-offs — these are the per-workload answers.\n")
        a(_table(["test", "best config", rank.metric, "feasible"], [
            [test, c.config_label, f"{c.value:,.2f}" if c.value is not None else "",
             "yes" if c.feasible else "no"]
            for test, c in sorted(rank.best_per_test.items())
        ]))

    if rank.config_scores:
        a("### Configurations ranked across the workload mix\n")
        a("`score` normalises each workload to that workload's winner, then averages, so a "
          "high-throughput workload cannot outvote a low-throughput one by scale alone. "
          "1.000 means it won every workload.\n")
        a(_table(["#", "config", "score", "tests", "feasible", "all feasible"], [
            [str(i + 1), s.config_label, f"{s.normalised_score:.3f}", str(s.n_tests),
             str(s.n_feasible), "yes" if s.fully_feasible else "no"]
            for i, s in enumerate(rank.config_scores[:25])
        ]))

    if rank.pareto:
        a("### Pareto front\n")
        a("Configurations not beaten on every axis simultaneously.\n")
        a(_table(["config", "test", rank.metric] + [c.metric for c in rank.pareto[0].verdicts], [
            [c.config_label, c.test,
             f"{c.value:,.2f}" if c.value is not None else ""]
            + [("" if v.value is None else f"{v.value:.4g}") for v in c.verdicts]
            for c in rank.pareto[:20]
        ]))

    for note in rank.notes:
        a(f"> **Note:** {note}\n")

    # ---- evidence ----
    online = [r for r in ctx.online if r.status == "ok"]
    if online:
        a("## Online results\n")
        a("`src=client` is wall-clock and cross-backend comparable; `src=server` is "
          "backend-reported and is a diagnostic, not a comparison.\n")
        cols = used_columns(online)
        a(_table([h for h, _, _ in cols], rows_for(online, cols)))

    offline = [r for r in ctx.offline if r.status == "ok"]
    if offline:
        a("## Offline results (native tools)\n")
        a("> These come from each backend's own benchmark tool and **do not share a "
          "measurement boundary**. `llama-bench` times `llama_decode()` with no scheduler; "
          "`vllm bench latency` runs the full engine. Compare each backend's offline number "
          "to its own online number — that difference is its HTTP+scheduler tax — not to the "
          "other backend's.\n")
        cols = used_columns(offline)
        a(_table([h for h, _, _ in cols], rows_for(offline, cols)))

    # ---- caveats ----
    failures = ctx.failures
    if failures:
        a("## Failed and skipped trials\n")
        a(_table(["trial", "backend", "test", "status", "error"], [
            [r.trial_id, r.backend, r.test, r.status,
             (r.error or "").replace("\n", " ")[:160]]
            for r in failures[:60]
        ]))
        if len(failures) > 60:
            a(f"_...and {len(failures) - 60} more; see `trials.jsonl`._\n")

    warnings = ctx.all_warnings()
    if warnings:
        a("## Warnings\n")
        for w in warnings:
            a(f"- {w}")
        a("")

    a("## Artifacts\n")
    for label, path in (m.get("artifacts") or {}).items():
        a(f"- `{path}` — {label.replace('_', ' ')}")
    a("")
    return "\n".join(out)


__all__ = ["render_markdown"]
