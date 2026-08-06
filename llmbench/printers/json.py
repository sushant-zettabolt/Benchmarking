"""Top-level array; also emits samples_ts (per-rep raw t/s) -- the only stdout location for
per-repetition raw data, matching llama-bench.cpp's JSON printer (docs/reference-notes.md §1).
"""
from __future__ import annotations

import json as _json

from .common import PrintRow, field_order, row_to_dict


def render_json(rows: list[PrintRow]) -> str:
    out = []
    for row in rows:
        d = row_to_dict(row)
        d["samples_ts"] = row.samples_ts
        out.append(d)
    return _json.dumps(out, indent=2) + "\n"
