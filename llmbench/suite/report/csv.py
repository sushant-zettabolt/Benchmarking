"""CSV report: every trial row, one line each, machine-readable."""
from __future__ import annotations

import csv
import io

from .common import REP_COLUMNS, ReportContext, cell, shell_command, used_columns


def render_csv(ctx: ReportContext) -> str:
    """Full-fidelity CSV.

    Unlike the markdown and HTML views this keeps *raw* values rather than display-formatted
    ones -- a spreadsheet should get 1027.1534... to sort and re-aggregate on, not "1,027.15".
    The last column is the exact server command line(s) of the row's deployment, which is too
    wide for the markdown and HTML tables (they list the commands in their own section).
    """
    columns = used_columns(ctx.results)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([header for header, _, _ in columns] + ["server cmd"])
    for r in ctx.results:
        cmds = (r.provenance or {}).get("server_commands") or []
        w.writerow([_raw(cell(r, s, k)) for _, s, k in columns]
                   + [" ;; ".join(shell_command(c) for c in cmds)])
    return buf.getvalue()


def render_reps_csv(ctx: ReportContext) -> str:
    """report_reps.csv: one line per measured request of every online trial.

    report.csv carries each trial's mean and stddev; this keeps the individual repetitions
    behind them (warm-up requests are not measured and are not included).
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["status", "backend", "model", "test", "conc", "trial"]
               + [header for header, _ in REP_COLUMNS])
    for r in ctx.results:
        if r.kind != "online" or r.src != "client":
            continue
        for rep in r.reps:
            w.writerow([r.status, r.backend, r.model, r.test, _raw(r.axes.get("concurrency")),
                        r.trial_id] + [_raw(rep.get(key)) for _, key in REP_COLUMNS])
    return buf.getvalue()


def _raw(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


__all__ = ["render_csv", "render_reps_csv"]
