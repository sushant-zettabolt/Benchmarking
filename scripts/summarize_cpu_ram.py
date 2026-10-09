#!/usr/bin/env python3
"""Per-variant, per-test CPU and RAM summary of a `llmbench sweep` output directory.

Joins report_reps.csv (client throughput per rep) with cores/<deployment>.csv (the 250 ms sampler
series, whose rows are labelled with the trial and the rep they fall in) and prints a Markdown
table plus writes cpu_ram_summary.csv next to the results.

  scripts/summarize_cpu_ram.py out/<sweep-name>

CPU: busy_cores = whole-core equivalents busy on the allocated cores; our_cores = the servers' share;
     mean over the rows inside measured requests (phase rep<N>).
RAM: server_rss/anon/file = the servers' resident memory (GiB), mean and peak inside measured requests;
     (the pod's own cgroup is not visible from inside it); `load_peak_rss` = the deployment's peak
     server RSS over its whole life (model load + warm-up included).
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return st.fmean(v) if v else None


def peak(v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def main(out: Path) -> None:
    deps = {d["id"]: d["axes"]["backend"] for d in json.loads((out / "deployments.json").read_text())}
    reps = defaultdict(list)                                  # (backend, test) -> [t/s per rep]
    prefill = defaultdict(list)
    with open(out / "report_reps.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            reps[(r["backend"], r["test"])].append(_f(r["t/s"]))
            prefill[(r["backend"], r["test"])].append(_f(r["prefill t/s"]))

    rows = []
    for dep_id, backend in deps.items():
        path = out / "cores" / f"{dep_id}.csv"
        if not path.exists():
            continue
        series = list(csv.DictReader(open(path, newline="")))
        load_peak = peak([_f(r.get("server_rss_gib")) for r in series])
        by_test = defaultdict(list)
        for r in series:
            if r["phase"].startswith("rep"):
                by_test[r["test"]].append(r)
        for test, rs in by_test.items():
            g = lambda k: [_f(r.get(k)) for r in rs]
            tps = reps.get((backend, test), [])
            rows.append({
                "backend": backend, "test": test,
                "tps_mean": mean(tps), "tps_sd": st.pstdev([x for x in tps if x is not None]) if len(tps) > 1 else None,
                "prefill_tps_mean": mean(prefill.get((backend, test), [])),
                "reps": len(tps), "samples": len(rs),
                "busy_cores_mean": mean(g("busy_cores")), "our_cores_mean": mean(g("our_cores")),
                "foreign_cores_mean": mean(g("foreign_cores")), "mhz_mean": mean(g("mhz")),
                "rss_gib_mean": mean(g("server_rss_gib")), "rss_gib_peak": peak(g("server_rss_gib")),
                "anon_gib_mean": mean(g("server_anon_gib")), "file_gib_mean": mean(g("server_file_gib")),
                "load_peak_rss_gib": load_peak,
            })
    rows.sort(key=lambda r: (r["test"], r["backend"]))
    if not rows:
        sys.exit("no rows: missing cores/*.csv memory columns or no measured reps")
    with open(out / "cpu_ram_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("| test | variant | t/s (mean ± sd) | prefill t/s | busy cores | server cores | foreign cores | MHz "
          "| RSS GiB mean (peak) | anon GiB | file GiB | peak RSS during load GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['test']} | {r['backend']} | {fmt(r['tps_mean'])} ± {fmt(r['tps_sd'])} | {fmt(r['prefill_tps_mean'])} "
              f"| {fmt(r['busy_cores_mean'])} | {fmt(r['our_cores_mean'])} | {fmt(r['foreign_cores_mean'], 2)} | {fmt(r['mhz_mean'], 0)} "
              f"| {fmt(r['rss_gib_mean'])} ({fmt(r['rss_gib_peak'])}) | {fmt(r['anon_gib_mean'])} | {fmt(r['file_gib_mean'])} "
              f"| {fmt(r['load_peak_rss_gib'])} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
