"""CSV report: every trial row, one line each, machine-readable."""
from __future__ import annotations

import csv
import io

from .common import ReportContext, cell, used_columns


def render_csv(ctx: ReportContext) -> str:
    """Full-fidelity CSV.

    Unlike the markdown and HTML views this keeps *raw* values rather than display-formatted
    ones -- a spreadsheet should get 1027.1534... to sort and re-aggregate on, not "1,027.15".
    """
    columns = used_columns(ctx.results)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([header for header, _, _ in columns])
    for r in ctx.results:
        w.writerow([_raw(cell(r, s, k)) for _, s, k in columns])
    return buf.getvalue()


def _raw(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


__all__ = ["render_csv"]
