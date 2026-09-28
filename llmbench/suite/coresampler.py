"""Per-core CPU utilisation of a deployment's cores, as a time series.

ContentionMonitor (contention.py) answers "was anyone else on our cores during this trial?"
with one number per trial. This answers the finer question of *how* the cores were used: one
row every `interval_s` while the deployment's servers are up, with every core's busy % in its
own column. It is what shows a prefill using all 32 cores against a decode that keeps one
core busy while 31 wait, or a request that leaves the cores idle altogether.

`/proc/stat` is not namespaced, so inside a container these are the physical cores' figures
-- including load from other pods sharing them, which is the point.

Every row is labelled with the trial it fell in and its phase within that trial:
  idle      no trial running (between trials, before the first, after the last)
  warmup    inside a trial, before its first measured request (warm-up, capacity probe)
  rep<N>    during measured request N (matches `rep` in report_reps.csv)
  between   inside a trial, outside any measured request (after the last rep, or between
            reps under concurrency)
A row covers an interval; its phase is the one containing the interval's midpoint.

Load columns, in whole-core equivalents over the row's interval: busy_cores (everything on
these cores) = our_cores (the deployment's server processes and their children) +
harness_cores (this sweep process: the HTTP client and the sampler) + foreign_cores (the rest:
other users, other pods on the same physical cores).

Resolution: /proc/stat counts in clock ticks (10 ms on x86 Linux), so at the default 250 ms
a single core's busy % moves in steps of about 4%. Aggregates over many cores are finer.
"""
from __future__ import annotations

import csv
import datetime
import os
import threading
import time
from pathlib import Path
from typing import Iterable

from .contention import CLK_TCK, _descendant_pids, _process_cpu_ticks

# /proc/stat per-cpu fields, in order.
_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def _read_cpu_fields(cpus: set[int]) -> dict[int, tuple[int, ...]]:
    """Per-cpu tick counters for the requested cpus: the eight _FIELDS, in order."""
    out: dict[int, tuple[int, ...]] = {}
    try:
        text = Path("/proc/stat").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        try:
            cpu = int(parts[0][3:])
        except ValueError:
            continue
        if cpu not in cpus:
            continue
        try:
            values = tuple(int(v) for v in parts[1:9])
        except ValueError:
            continue
        if len(values) == len(_FIELDS):
            out[cpu] = values
    return out


class CoreSampler:
    """Samples a cpu set in a background thread and writes labelled rows to a CSV file.

    The sweep's event loop calls begin_trial()/end_trial() around each trial; the thread only
    measures and buffers. Rows reach disk when a trial ends and when the sampler stops, so the
    labels can use the trial's actual request windows, which are only known afterwards.
    """

    def __init__(self, cpus: Iterable[int], pids: Iterable[int], path: str | Path, *,
                 interval_s: float = 0.25):
        self.cpus = sorted(set(cpus))
        self.pids = list(pids)
        self.path = Path(path)
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buf: list[dict] = []
        self._label: tuple[str, str] = ("", "")          # (trial_id, test)
        self._t0_ns = 0
        self.n_rows = 0

    # -- lifecycle --

    def start(self) -> None:
        self._t0_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._loop, name="core-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.interval_s + 5)
        self._write(self._take(), rep_windows=[])

    # -- trial labelling --

    def begin_trial(self, trial_id: str, test: str) -> None:
        pending = self._take()          # everything so far was outside any trial
        self._write(pending, rep_windows=[])
        with self._lock:
            self._label = (trial_id, test)

    def end_trial(self, rep_windows: list[tuple[int, int, int]]) -> None:
        """rep_windows: (rep_idx, t_send_ns, t_end_ns) of each measured request, on the
        time.perf_counter_ns clock the runner stamps its records with."""
        with self._lock:
            rows, self._buf = self._buf, []
            self._label = ("", "")
        self._write(rows, rep_windows=rep_windows)

    # -- sampling thread --

    def _our_ticks(self) -> int:
        return sum(_process_cpu_ticks(p) for pid in self.pids for p in _descendant_pids(pid))

    def _loop(self) -> None:
        wanted = set(self.cpus)
        me = os.getpid()
        prev_t = time.perf_counter_ns()
        prev = _read_cpu_fields(wanted)
        prev_ours, prev_mine = self._our_ticks(), _process_cpu_ticks(me)
        while not self._stop.wait(self.interval_s):
            now_t = time.perf_counter_ns()
            cur = _read_cpu_fields(wanted)
            ours, mine = self._our_ticks(), _process_cpu_ticks(me)
            row = self._row(prev, cur, prev_t, now_t, ours - prev_ours, mine - prev_mine)
            if row is not None:
                with self._lock:
                    row["trial"], row["test"] = self._label
                    self._buf.append(row)
            prev, prev_t, prev_ours, prev_mine = cur, now_t, ours, mine

    def _row(self, prev, cur, t0_ns: int, t1_ns: int, our_ticks: int,
             harness_ticks: int) -> dict | None:
        elapsed_s = (t1_ns - t0_ns) / 1e9
        if elapsed_s <= 0 or not prev or not cur:
            return None
        per_cpu: dict[int, float | None] = {}
        field_sums = [0] * len(_FIELDS)
        total_sum = busy_ticks = 0
        for cpu in self.cpus:
            a, b = prev.get(cpu), cur.get(cpu)
            if a is None or b is None:
                per_cpu[cpu] = None
                continue
            d = [max(0, y - x) for x, y in zip(a, b)]
            total = sum(d)
            idle = d[3] + d[4]                                  # idle + iowait
            per_cpu[cpu] = 100.0 * (total - idle) / total if total else None
            for i, v in enumerate(d):
                field_sums[i] += v
            total_sum += total
            busy_ticks += total - idle

        def pct(*names: str) -> float | None:
            if not total_sum:
                return None
            return 100.0 * sum(field_sums[_FIELDS.index(n)] for n in names) / total_sum

        busy_cores = busy_ticks / CLK_TCK / elapsed_s
        our_cores = min(busy_cores, max(0, our_ticks) / CLK_TCK / elapsed_s)
        # The sweep process itself (HTTP client, this sampler). In a pod it shares the servers'
        # cpuset, so it is on these cores; it is neither the servers nor a foreign load.
        harness_cores = min(busy_cores - our_cores, max(0, harness_ticks) / CLK_TCK / elapsed_s)
        return {
            "t0_ns": t0_ns, "t1_ns": t1_ns,
            "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "interval_s": elapsed_s,
            "busy_pct": pct("user", "nice", "system", "irq", "softirq", "steal"),
            "user_pct": pct("user", "nice"),
            "system_pct": pct("system"),
            "iowait_pct": pct("iowait"),
            "irq_pct": pct("irq", "softirq"),
            "steal_pct": pct("steal"),
            "busy_cores": busy_cores,
            "our_cores": our_cores,
            "harness_cores": harness_cores,
            "foreign_cores": max(0.0, busy_cores - our_cores - harness_cores),
            "per_cpu": per_cpu,
        }

    # -- output --

    def _take(self) -> list[dict]:
        with self._lock:
            rows, self._buf = self._buf, []
        return rows

    @staticmethod
    def _phase(mid_ns: int, trial_id: str, rep_windows) -> str:
        if not trial_id:
            return "idle"
        for rep, start, end in rep_windows:
            if start <= mid_ns <= end:
                return f"rep{rep}"
        first = min((start for _, start, _ in rep_windows), default=None)
        if first is None or mid_ns < first:
            return "warmup"
        return "between"

    def _write(self, rows: list[dict], *, rep_windows) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "t_s", "interval_s", "trial", "test", "phase",
                            "busy_pct", "user_pct", "system_pct", "iowait_pct", "irq_pct",
                            "steal_pct", "busy_cores", "our_cores", "harness_cores", "foreign_cores"]
                           + [f"cpu{c}" for c in self.cpus])
            for r in rows:
                mid = (r["t0_ns"] + r["t1_ns"]) // 2
                w.writerow(
                    [r["ts_utc"], _r((r["t1_ns"] - self._t0_ns) / 1e9, 3), _r(r["interval_s"], 3),
                     r["trial"], r["test"],
                     self._phase(mid, r["trial"], rep_windows)]
                    + [_r(r[k], 1) for k in ("busy_pct", "user_pct", "system_pct",
                                             "iowait_pct", "irq_pct", "steal_pct")]
                    + [_r(r[k], 2) for k in ("busy_cores", "our_cores", "harness_cores",
                                             "foreign_cores")]
                    + [_r(r["per_cpu"].get(c), 1) for c in self.cpus]
                )
        self.n_rows += len(rows)


def _r(value, digits: int):
    return "" if value is None else round(value, digits)


__all__ = ["CoreSampler"]
