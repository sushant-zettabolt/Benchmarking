"""Markdown printer. `-o md` (default): parity-annotated table -- prepends the parity
report header when a ParityReport is supplied (course_correct.txt §2.10/§5), and every
flagged row gets a footnote so a flagged row never looks clean (spec §10).
`-o md-llamabench`: the plain llama-bench-shaped compatibility view, no parity annotations
(course_correct.txt §5/§6 -- renamed from the old default, demoted to a view).
"""
from __future__ import annotations

from ..config import CmdParams
from ..parity import ParityReport
from .common import PrintRow, format_params, format_size, format_ts

BASE_COLUMNS = ["model", "size", "params", "backend"]

# candidate extra columns: (header, attr, default_attr_on_CmdParams_group2 or None)
CANDIDATE_COLUMNS = [
    ("ngl", "ngl"), ("threads", "threads"), ("batch", "batch"), ("ubatch", "ubatch"),
    ("ctk", "ctk"), ("ctv", "ctv"), ("fa", "flash_attn"), ("np", "n_parallel"), ("ctx", "n_ctx"),
]


def _varying_columns(rows: list[PrintRow], defaults: CmdParams | None) -> list[str]:
    """llama-bench.cpp's rule: show a column if values differ across the run OR differ from
    the compiled default -- checked globally, not per-row."""
    out = []
    for header, attr in CANDIDATE_COLUMNS:
        values = {getattr(r, attr) for r in rows}
        if len(values) > 1:
            out.append(header)
            continue
        if defaults is not None and values:
            default_list = getattr(defaults, {
                "ngl": "ngl", "threads": "threads", "batch": "batch", "ubatch": "ubatch",
                "ctk": "ctk", "ctv": "ctv", "flash_attn": "flash_attn",
                "n_parallel": "n_parallel", "n_ctx": "n_ctx",
            }[attr], [])
            only_value = next(iter(values))
            if default_list and only_value != default_list[0]:
                out.append(header)
    return out


def _col_value(row: PrintRow, header: str):
    mapping = {
        "ngl": row.ngl, "threads": row.threads, "batch": row.batch, "ubatch": row.ubatch,
        "ctk": row.ctk, "ctv": row.ctv, "fa": row.flash_attn, "np": row.n_parallel, "ctx": row.n_ctx,
    }
    return mapping[header]


def render_parity_header(report: ParityReport) -> str:
    lines = []
    if report.all_matched:
        lines.append("**PARITY: all axes matched.**")
    else:
        lines.append("**PARITY WARNING -- NOT A VALID COMPARISON**")
        lines.append("")
        lines.append(report.headline())
    lines.append("")
    lines.append("| axis | verdict | value (a) | value (b) |")
    lines.append("| --- | --- | ---: | ---: |")
    for a in report.axes:
        marker = {"matched": "OK", "mismatched": "MISMATCH", "unverifiable": "?"}[a.verdict]
        lines.append(f"| {a.axis} | {marker} | {a.value_a} | {a.value_b} |")
    lines.append("")
    return "\n".join(lines)


def render_table(rows: list[PrintRow], *, defaults: CmdParams | None = None, parity: ParityReport | None = None) -> str:
    out_parts = []
    if parity is not None:
        out_parts.append(render_parity_header(parity))

    extra = _varying_columns(rows, defaults)
    show_c = any(r.concurrency > 1 for r in rows)
    show_rate = any(r.request_rate for r in rows)
    columns = BASE_COLUMNS + extra + (["c"] if show_c else []) + (["rate"] if show_rate else []) + ["src", "test", "t/s"]

    header = "| " + " | ".join(columns) + " |"
    right_justified = {"size", "params"} | set(extra) | {"c", "rate", "t/s"}
    sep_cells = []
    for c in columns:
        sep_cells.append("---:" if c in right_justified else "---")
    sep = "| " + " | ".join(sep_cells) + " |"

    body = []
    footnotes = []
    for row in rows:
        cells = [row.model, format_size(row.size_bytes), format_params(row.n_params), row.backend]
        for h in extra:
            cells.append(str(_col_value(row, h)))
        if show_c:
            cells.append(str(row.concurrency))
        if show_rate:
            cells.append(f"{row.request_rate:.2f}" if row.request_rate else "-")
        cells.append(row.src)
        test = row.test
        if row.flags:
            test += " *"
            footnotes.append(f"* `{row.test}` [{row.src}]: {', '.join(row.flags)}")
        cells.append(test)
        cells.append(format_ts(row.tps_mean, row.tps_stddev))
        body.append("| " + " | ".join(cells) + " |")

    out_parts.append("\n".join([header, sep] + body))
    if footnotes:
        out_parts.append("")
        out_parts.append("\n".join(footnotes))
    return "\n".join(out_parts) + "\n"


def render_llamabench_table(rows: list[PrintRow], *, defaults: CmdParams | None = None) -> str:
    """-o md-llamabench: compatibility view, src=server rows only (closest analogue to
    native llama-bench), no parity annotations, no footnotes beyond what llama-bench itself
    would show (course_correct.txt §6: demoted to a view, not a gate)."""
    server_rows = [r for r in rows if r.src == "server"] or rows
    return render_table(server_rows, defaults=defaults, parity=None)
