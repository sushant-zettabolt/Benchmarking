"""CREATE TABLE + INSERT, mirroring llama-bench's field set (docs/reference-notes.md §1).

Deliberate departure: llama-bench's own SQL printer quotes every value as a string literal
regardless of declared column type (llama-bench.cpp:2100) -- a quirk of its C++ formatting
helper, not a behavior worth reproducing. llmbench emits properly typed SQL literals.
"""
from __future__ import annotations

from .common import PrintRow, field_order, row_to_dict

_SQL_TYPES = {
    "model": "TEXT", "size_bytes": "INTEGER", "n_params": "INTEGER", "backend": "TEXT",
    "ngl": "INTEGER", "threads": "INTEGER", "batch": "INTEGER", "ubatch": "INTEGER",
    "ctk": "TEXT", "ctv": "TEXT", "flash_attn": "TEXT", "n_parallel": "INTEGER", "n_ctx": "INTEGER",
    "src": "TEXT", "test": "TEXT", "tps_mean": "REAL", "tps_stddev": "REAL",
    "ttft_ms_mean": "REAL", "tpot_ms_mean": "REAL", "itl_p99_ms": "REAL", "overhead_ms_mean": "REAL",
    "n_prompt_actual": "INTEGER", "cached_tokens": "INTEGER", "preemptions_delta": "INTEGER",
    "load_mode": "TEXT", "request_rate": "REAL", "concurrency": "INTEGER", "shared_prefix_n": "INTEGER",
    "url": "TEXT", "tag": "TEXT",
}

TABLE_NAME = "llmbench_results"


def _sql_literal(value, sql_type: str) -> str:
    if value is None:
        return "NULL"
    if sql_type == "TEXT":
        return "'" + str(value).replace("'", "''") + "'"
    return str(value)


def render_sql(rows: list[PrintRow]) -> str:
    fields = field_order()
    cols_ddl = ",\n    ".join(f"{f} {_SQL_TYPES[f]}" for f in fields)
    out = [f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (\n    {cols_ddl}\n);"]
    for row in rows:
        d = row_to_dict(row)
        values = ", ".join(_sql_literal(d.get(f), _SQL_TYPES[f]) for f in fields)
        out.append(f"INSERT INTO {TABLE_NAME} ({', '.join(fields)}) VALUES ({values});")
    return "\n".join(out) + "\n"
