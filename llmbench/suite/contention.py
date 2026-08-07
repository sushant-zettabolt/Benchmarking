"""Detect CPU contention from processes this run does not own.

Pinning a server to cores 96-191 guarantees it runs *only* there. It guarantees nothing about
who else is running there too. On a shared box an unpinned job from another user spreads
across every core and silently halves your throughput, and nothing in the measurement itself
reveals it -- the numbers just come out low, reproducibly enough to look real.

That is not hypothetical here. During development of this module a `llama-mtmd-cli` from
another session ran unpinned across all 384 logical CPUs at ~85% per core, and made an
nginx-vs-client comparison look like a 10x regression that was entirely contention. It was
diagnosed only because the numbers were implausible enough to go looking.

So the sweep samples it. `/proc/stat` gives per-CPU busy time; the delta over a short window
gives per-CPU utilisation. Subtracting the CPU our own managed processes consumed over the
same window leaves the foreign load. That number goes into every trial's provenance and, past
a threshold, into the report as a warning.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _clk_tck() -> float:
    """Ticks per second for /proc/stat and /proc/<pid>/stat.

    100 on every Linux/x86 configuration in practice, but ask rather than assume: a kernel
    built with a different CONFIG_HZ would put a constant multiplicative error into every
    contention figure, which is exactly the kind of quiet wrongness this module exists to
    catch elsewhere.
    """
    try:
        value = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return 100.0
    return float(value) if value and value > 0 else 100.0


CLK_TCK = _clk_tck()


def _read_proc_stat() -> dict[int, tuple[int, int]]:
    """Per-cpu (busy_ticks, total_ticks) from /proc/stat.

    Idle is user-visible idle plus iowait; everything else counts as busy.
    """
    out: dict[int, tuple[int, int]] = {}
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
            values = [int(v) for v in parts[1:11]]
        except (ValueError, IndexError):
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
        total = sum(values)
        out[cpu] = (total - idle, total)
    return out


def _process_cpu_ticks(pid: int) -> int:
    """utime+stime for a process and its reaped children, in clock ticks."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (OSError, IndexError):
        return 0
    try:
        # After the comm field: state is [0], so utime is index 11, stime 12.
        return int(fields[11]) + int(fields[12])
    except (ValueError, IndexError):
        return 0


def _descendant_pids(pid: int) -> list[int]:
    """A server's worker threads live in its own PID, but vLLM forks an engine child."""
    pids = [pid]
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return pids
    for c in children:
        try:
            pids.extend(_descendant_pids(int(c)))
        except ValueError:
            continue
    return pids


@dataclass
class ContentionSample:
    """Utilisation of a set of cpus over a sampling window, split ours vs foreign."""

    cpus: list[int] = field(default_factory=list)
    window_s: float = 0.0
    busy_cores: float = 0.0        # total busy, in whole-core equivalents
    our_cores: float = 0.0         # attributable to pids we own
    foreign_cores: float = 0.0     # everything else
    n_cpus: int = 0

    @property
    def foreign_pct(self) -> float:
        """Foreign load as a percentage of the sampled cpus' total capacity."""
        return (self.foreign_cores / self.n_cpus * 100.0) if self.n_cpus else 0.0

    @property
    def busy_pct(self) -> float:
        return (self.busy_cores / self.n_cpus * 100.0) if self.n_cpus else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cpus": self.n_cpus,
            "window_s": round(self.window_s, 2),
            "busy_cores": round(self.busy_cores, 2),
            "our_cores": round(self.our_cores, 2),
            "foreign_cores": round(self.foreign_cores, 2),
            "foreign_pct": round(self.foreign_pct, 1),
            "busy_pct": round(self.busy_pct, 1),
        }

    def warning(self, threshold_pct: float) -> str | None:
        if self.foreign_pct < threshold_pct:
            return None
        return (
            f"{self.foreign_cores:.1f} core(s) of foreign CPU load "
            f"({self.foreign_pct:.0f}% of the {self.n_cpus} allocated cpus) were active during "
            f"this measurement -- processes this run does not own are competing for the same "
            f"cores. Pinning prevents our servers from leaving their cores; it cannot stop "
            f"anyone else from using them. Treat these numbers as contaminated."
        )


class ContentionMonitor:
    """Sample foreign CPU load on a set of cpus across a measurement."""

    def __init__(self, cpus: Iterable[int], pids: Iterable[int] = ()):
        self.cpus = sorted(set(cpus))
        self.pids = list(pids)
        self._t0 = 0.0
        self._stat0: dict[int, tuple[int, int]] = {}
        self._proc0 = 0

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._stat0 = _read_proc_stat()
        self._proc0 = self._our_ticks()

    def _our_ticks(self) -> int:
        total = 0
        for pid in self.pids:
            for p in _descendant_pids(pid):
                total += _process_cpu_ticks(p)
        return total

    def sample(self) -> ContentionSample:
        elapsed = time.monotonic() - self._t0
        stat1 = _read_proc_stat()
        if elapsed <= 0 or not self._stat0 or not stat1:
            return ContentionSample(cpus=self.cpus, n_cpus=len(self.cpus))

        busy_ticks = 0
        for cpu in self.cpus:
            a, b = self._stat0.get(cpu), stat1.get(cpu)
            if a and b:
                busy_ticks += max(0, b[0] - a[0])

        our_ticks = max(0, self._our_ticks() - self._proc0)
        busy_cores = busy_ticks / CLK_TCK / elapsed
        # Our processes may also run on cpus outside this set (they should not, but a failed
        # pinning is exactly what we want to notice); clamping keeps foreign >= 0 either way.
        our_cores = min(busy_cores, our_ticks / CLK_TCK / elapsed)
        return ContentionSample(
            cpus=self.cpus, window_s=elapsed, n_cpus=len(self.cpus),
            busy_cores=busy_cores, our_cores=our_cores,
            foreign_cores=max(0.0, busy_cores - our_cores),
        )


def snapshot(cpus: Iterable[int], *, window_s: float = 1.0) -> ContentionSample:
    """One-shot measurement of how busy a cpu set is right now, with no ownership split.

    Used as a pre-flight before a sweep starts: if the machine is already loaded, say so
    before spending hours producing numbers that describe someone else's job as much as yours.
    """
    mon = ContentionMonitor(cpus, pids=[])
    mon.start()
    time.sleep(window_s)
    return mon.sample()


__all__ = ["ContentionMonitor", "ContentionSample", "snapshot"]
