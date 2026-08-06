"""RawRecord: one per request, raw timestamps only. Never aggregate here (non-negotiable #2)."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawRecord:
    schema: int = 1
    run_id: str = ""
    instance_id: str = ""
    rep_idx: int = 0
    ts_utc: str = ""

    backend: str = ""
    backend_version: str = ""
    url: str = ""
    tag: str | None = None

    model: str = ""
    model_size_bytes: int | None = None
    model_n_params: int | None = None

    server_flags_json: str = "{}"
    load_mode: str = "closed"  # closed | open
    request_rate: float | None = None
    burstiness: float | None = None

    n_prompt_target: int = 0
    n_prompt_actual: int | None = None
    n_gen_target: int = 0
    n_gen_actual: int | None = None
    n_depth: int = 0
    concurrency: int = 1
    shared_prefix_n: int = 0

    t_send_ns: int = 0
    t_first_token_ns: int | None = None
    t_last_token_ns: int | None = None
    t_end_ns: int = 0
    itl_ns: list[int] = field(default_factory=list)

    server_cache_n: int | None = None
    server_prompt_n: int | None = None
    server_prompt_ms: float | None = None
    server_predicted_n: int | None = None
    server_predicted_ms: float | None = None

    cached_tokens: int | None = None
    preemptions_delta: int | None = None
    slots_busy_at_send: int | None = None

    http_status: int | None = None
    error: str | None = None
    flags: list[str] = field(default_factory=list)
    warmup_iterations: int = 0  # course_correct.txt §2.9: itself a reportable finding

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS raw_records (
    run_id TEXT, instance_id TEXT, rep_idx INTEGER, ts_utc TEXT,
    backend TEXT, backend_version TEXT, url TEXT, tag TEXT,
    model TEXT, model_size_bytes INTEGER, model_n_params INTEGER,
    server_flags_json TEXT, load_mode TEXT, request_rate REAL, burstiness REAL,
    n_prompt_target INTEGER, n_prompt_actual INTEGER, n_gen_target INTEGER, n_gen_actual INTEGER,
    n_depth INTEGER, concurrency INTEGER, shared_prefix_n INTEGER,
    t_send_ns INTEGER, t_first_token_ns INTEGER, t_last_token_ns INTEGER, t_end_ns INTEGER,
    itl_ns_json TEXT,
    server_cache_n INTEGER, server_prompt_n INTEGER, server_prompt_ms REAL,
    server_predicted_n INTEGER, server_predicted_ms REAL,
    cached_tokens INTEGER, preemptions_delta INTEGER, slots_busy_at_send INTEGER,
    http_status INTEGER, error TEXT, flags_json TEXT, warmup_iterations INTEGER
);
"""


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, rec: RawRecord) -> None:
        self._f.write(json.dumps(rec.to_dict()) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class SqliteSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(SQLITE_DDL)
        self._conn.commit()

    def write(self, rec: RawRecord) -> None:
        d = rec.to_dict()
        d["itl_ns_json"] = json.dumps(d.pop("itl_ns"))
        d["flags_json"] = json.dumps(d.pop("flags"))
        d.pop("schema", None)
        cols = ", ".join(d.keys())
        qs = ", ".join("?" for _ in d)
        self._conn.execute(f"INSERT INTO raw_records ({cols}) VALUES ({qs})", list(d.values()))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def read_jsonl(path: str | Path) -> list[RawRecord]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("schema", None)
            out.append(RawRecord(**d))
    return out
