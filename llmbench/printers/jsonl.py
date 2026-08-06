"""Same fields/typing as json.py, one object per line, no enclosing array (streaming/tail-
friendly, matching llama-bench.cpp's JSONL printer)."""
from __future__ import annotations

import json as _json

from .common import PrintRow, row_to_dict


def render_jsonl(rows: list[PrintRow]) -> str:
    lines = []
    for row in rows:
        d = row_to_dict(row)
        d["samples_ts"] = row.samples_ts
        lines.append(_json.dumps(d))
    return "\n".join(lines) + "\n"
