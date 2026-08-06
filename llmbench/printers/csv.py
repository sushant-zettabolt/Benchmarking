"""Every field double-quote-wrapped, embedded quotes doubled, even numeric fields
(llama-bench.cpp's escape_csv, docs/reference-notes.md §1)."""
from __future__ import annotations

from .common import PrintRow, field_order, row_to_dict


def _escape_csv(value) -> str:
    s = "" if value is None else str(value)
    return '"' + s.replace('"', '""') + '"'


def render_csv(rows: list[PrintRow]) -> str:
    fields = field_order()
    lines = [",".join(_escape_csv(f) for f in fields)]
    for row in rows:
        d = row_to_dict(row)
        lines.append(",".join(_escape_csv(d.get(f)) for f in fields))
    return "\n".join(lines) + "\n"
