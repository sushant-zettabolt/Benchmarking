#!/usr/bin/env python3
"""Re-render a stored llmbench run in any output format, without re-benchmarking.

`llmbench` picks its stdout format at run time via `-o`, but always writes the raw
per-request records to `--out-dir` (JSONL + SQLite). Since `metrics.py` computes every
statistic from those raw records rather than at collection time (README, "Quick start"),
a finished run can be re-rendered in any format after the fact. This script does that.

It reuses llmbench's own aggregation (`metrics.aggregate_client` / `aggregate_server`)
and printers, so the numbers are identical to what `-o <fmt>` would have printed live.

Usage:
    scripts/llmbench-report.py out/run2-vllm                  # JSON (default)
    scripts/llmbench-report.py out/run2-vllm -f md            # markdown table
    scripts/llmbench-report.py out/run2-llamacpp/*.jsonl
    scripts/llmbench-report.py out/run2-vllm/llmbench.db --list
    scripts/llmbench-report.py out/run2-vllm --src client -f csv

The input may be a directory (a *.jsonl inside it is preferred, else llmbench.db), a
.jsonl file, or a .db file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llmbench import metrics  # noqa: E402
from llmbench.cli import _to_print_row  # noqa: E402
from llmbench.config import CmdParams  # noqa: E402
from llmbench.printers.common import PrintRow  # noqa: E402
from llmbench.printers.csv import render_csv  # noqa: E402
from llmbench.printers.json import render_json  # noqa: E402
from llmbench.printers.jsonl import render_jsonl  # noqa: E402
from llmbench.printers.markdown import render_llamabench_table, render_table  # noqa: E402
from llmbench.printers.sql import render_sql  # noqa: E402
from llmbench.store import RunStore  # noqa: E402


def resolve_source(path: Path) -> Path:
    """A directory holds both a JSONL and an llmbench.db; prefer the JSONL."""
    if path.is_dir():
        jsonl = sorted(path.glob("*.jsonl"))
        if jsonl:
            return jsonl[0]
        db = path / "llmbench.db"
        if db.exists():
            return db
        raise SystemExit(f"no *.jsonl or llmbench.db under {path}")
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    return path


def load_records(src: Path, run_id: str | None) -> list[dict]:
    if src.suffix == ".jsonl":
        records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        if run_id:
            records = [r for r in records if r.get("run_id") == run_id]
        return records

    store = RunStore(src)
    try:
        runs = store.list_runs()
        if not runs:
            raise SystemExit(f"{src} contains no runs")
        if run_id is None and len(runs) > 1:
            raise SystemExit(
                f"{src} holds {len(runs)} runs; pick one with --run-id (see --list): {', '.join(runs)}"
            )
        return store.load_run(run_id or runs[0])
    finally:
        store.close()


def test_name_of(rec: dict) -> str:
    """Replicates Instance.test_name() (config.py:223) from stored record fields."""
    n_prompt, n_gen = rec["n_prompt_target"], rec["n_gen_target"]
    if n_gen == 0:
        base = f"pp{n_prompt}"
    elif n_prompt == 0:
        base = f"tg{n_gen}"
    else:
        base = f"pp{n_prompt}+tg{n_gen}"
    if rec.get("n_depth"):
        base += f" @ d{rec['n_depth']}"
    return base


def group_by_test(records: list[dict]) -> dict[str, list[dict]]:
    """Group in first-seen order so the output row order matches the original run."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(test_name_of(rec), []).append(rec)
    return groups


def build_rows(records: list[dict], src_filter: str) -> list[PrintRow]:
    """Rebuild the shims _to_print_row reads, so aggregation stays in sync with the CLI.

    Group-2 knobs (ngl/threads/batch/...) are not stored per record -- they live in
    server_flags_json, which is "{}" in attach mode. Absent values surface as None
    rather than being invented.
    """
    rows: list[PrintRow] = []
    for test_name, recs in group_by_test(records).items():
        head = recs[0]
        flags = json.loads(head.get("server_flags_json") or "{}")
        inst = SimpleNamespace(
            model=head.get("model", ""),
            concurrency=head.get("concurrency", 1),
            shared_prefix=head.get("shared_prefix_n", 0),
            ngl=flags.get("ngl"), threads=flags.get("threads"),
            batch=flags.get("batch"), ubatch=flags.get("ubatch"),
            ctk=flags.get("ctk"), ctv=flags.get("ctv"),
            flash_attn=flags.get("flash_attn"), n_parallel=flags.get("n_parallel"),
            n_ctx=flags.get("n_ctx"),
        )
        info = SimpleNamespace(
            backend=head.get("backend", ""),
            model_size_bytes=head.get("model_size_bytes"),
            model_n_params=head.get("model_n_params"),
        )
        params = CmdParams(
            url=head.get("url", "") or "",
            tag=head.get("tag"),
            request_rate=head.get("request_rate"),
        )

        if src_filter in ("client", "both"):
            rows.append(_to_print_row(metrics.aggregate_client(recs, test_name), inst, info, params))
        if src_filter in ("server", "both"):
            srow = metrics.aggregate_server(recs, test_name)
            if srow is not None:
                rows.append(_to_print_row(srow, inst, info, params))
    return rows


# The markdown printers take `defaults` to apply llama-bench's "show a column if it differs
# from the compiled default" rule. Raw records don't carry the run's CmdParams, so an
# unrecorded knob reads as None, which differs from every default and would print a column
# of "None" as though it had been measured. Passing defaults=None drops that half of the
# rule, leaving the other half -- show a column only where it actually varies across rows.
RENDERERS = {
    "json": render_json,
    "jsonl": render_jsonl,
    "csv": render_csv,
    "sql": render_sql,
    "md": lambda rows: render_table(rows, defaults=None),
    "md-llamabench": lambda rows: render_llamabench_table(rows, defaults=None),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-render a stored llmbench run from its raw records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    ap.add_argument("path", type=Path, help="out-dir, *.jsonl, or llmbench.db")
    ap.add_argument("-f", "--format", default="json", choices=sorted(RENDERERS),
                    help="output format (default: json)")
    ap.add_argument("--src", default="both", choices=["client", "server", "both"],
                    help="measurement path to emit (default: both)")
    ap.add_argument("--run-id", default=None, help="select one run when the source holds several")
    ap.add_argument("--list", action="store_true", help="list run ids in the source and exit")
    args = ap.parse_args()

    src = resolve_source(args.path)

    if args.list:
        if src.suffix == ".jsonl":
            seen = dict.fromkeys(
                json.loads(l)["run_id"] for l in src.read_text().splitlines() if l.strip()
            )
            runs = list(seen)
        else:
            store = RunStore(src)
            try:
                runs = store.list_runs()
            finally:
                store.close()
        print(f"# {src}")
        for r in runs:
            print(r)
        return 0

    records = load_records(src, args.run_id)
    if not records:
        raise SystemExit(f"no records in {src}" + (f" for run-id {args.run_id}" if args.run_id else ""))

    sys.stdout.write(RENDERERS[args.format](build_rows(records, args.src)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
