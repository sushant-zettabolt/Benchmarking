"""SQLite-backed run storage and `llmbench compare` (milestone 5)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import metrics
from .records import SQLITE_DDL


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SQLITE_DDL)
        self._conn.commit()

    def list_runs(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT run_id FROM raw_records ORDER BY run_id").fetchall()
        return [r["run_id"] for r in rows]

    def load_run(self, run_id: str) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM raw_records WHERE run_id = ?", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["itl_ns"] = json.loads(d.pop("itl_ns_json") or "[]")
            d["flags"] = json.loads(d.pop("flags_json") or "[]")
            out.append(d)
        return out

    def close(self) -> None:
        self._conn.close()


@dataclass
class RegressionFinding:
    test_name: str
    src: str
    metric: str
    value_a: float
    value_b: float
    pct_change: float


def _group_by_test(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = f"pp{r['n_prompt_target']}+tg{r['n_gen_target']}"
        if r.get("n_depth"):
            key += f" @ d{r['n_depth']}"
        groups.setdefault(key, []).append(r)
    return groups


def compare_runs(records_a: list[dict], records_b: list[dict], *, threshold_pct: float = 5.0) -> list[RegressionFinding]:
    findings = []
    groups_a = _group_by_test(records_a)
    groups_b = _group_by_test(records_b)
    for test_name in sorted(set(groups_a) & set(groups_b)):
        ra, rb = groups_a[test_name], groups_b[test_name]
        row_a = metrics.aggregate_client(ra, test_name)
        row_b = metrics.aggregate_client(rb, test_name)
        if row_a.tps_mean and row_b.tps_mean:
            pct = (row_b.tps_mean - row_a.tps_mean) / row_a.tps_mean * 100
            if abs(pct) >= threshold_pct:
                findings.append(RegressionFinding(
                    test_name=test_name, src="client", metric="tps_mean",
                    value_a=row_a.tps_mean, value_b=row_b.tps_mean, pct_change=pct,
                ))
        sa = metrics.aggregate_server(ra, test_name)
        sb = metrics.aggregate_server(rb, test_name)
        if sa and sb and sa.tps_mean and sb.tps_mean:
            pct = (sb.tps_mean - sa.tps_mean) / sa.tps_mean * 100
            if abs(pct) >= threshold_pct:
                findings.append(RegressionFinding(
                    test_name=test_name, src="server", metric="tps_mean",
                    value_a=sa.tps_mean, value_b=sb.tps_mean, pct_change=pct,
                ))
    return findings
