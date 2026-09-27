"""A durable ledger of every process a sweep started, so none can outlive it unseen.

Everything the suite launches runs in its own session (`start_new_session=True`), because
teardown needs a process group to signal. The side effect is that those processes do not
receive the terminal's SIGHUP: when an overnight sweep's SSH session drops, or the sweep is
`kill`ed, or a scheduler times it out, Python dies without running a single `finally`, and
every server it launched keeps running -- holding its port, its cores and its KV memory. The
next run's port pre-flight then refuses to start, and nothing on disk says whose they are.

Three layers close that:

  1. `start()` asks the kernel to SIGTERM the child when the sweep dies
     (`PR_SET_PDEATHSIG`). That covers the direct child -- the server, or the offline tool --
     on Linux. It does not reach grandchildren (vLLM's engine core), which is why:
  2. every start is recorded in `out_dir/pids.json`, with each process's kernel start time
     so a reused PID can never be mistaken for ours, and
  3. `cleanup()` (`llmbench sweep cleanup <out_dir>`) signals exactly the processes in that
     ledger that are verifiably still ours: same PID, same start time, or a member of the
     session we created that started after its leader did.

The rule from deploy.py still holds: nothing here ever matches a process by name.
"""
from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PIDS_FILE = "pids.json"
PR_SET_PDEATHSIG = 1


class RunStopping(RuntimeError):
    """Raised by `start()` once the run has begun stopping; nothing new may launch."""


def atomic_write_text(path: Path, text: str) -> None:
    """Write-then-rename, so a crash mid-write leaves the previous version, not half a file.

    Shared by every manifest this suite rewrites in place (run.json, plan.json,
    deployments.json, pids.json). A truncated run.json used to be enough to break the report
    step of the run that wrote it.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --- /proc helpers (Linux; every one degrades to "unknown" elsewhere) ---


def _stat_fields(pid: int) -> list[str] | None:
    """/proc/<pid>/stat after the comm field: index 0 is field 3 (state)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (OSError, IndexError):
        return None


def start_ticks(pid: int, *, live_only: bool = False) -> int | None:
    """Kernel start time of a process (stat field 22), in clock ticks since boot.

    PID plus start time identifies a process uniquely for the life of the machine; PID alone
    does not, and a ledger that trusted PID alone would one day kill a stranger.

    `live_only` reports a zombie as absent: it has exited and holds no cores, ports or
    memory, it is merely waiting for its parent to reap it.
    """
    fields = _stat_fields(pid)
    if not fields or (live_only and fields[0] == "Z"):
        return None
    try:
        return int(fields[19])
    except (ValueError, IndexError):
        return None


def _session_of(fields: list[str]) -> int | None:
    try:
        return int(fields[3])        # stat field 6
    except (ValueError, IndexError):
        return None


def _session_members(sid: int, not_before: int) -> list[int]:
    """Live PIDs in session `sid` that started no earlier than its leader did."""
    out = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for name in entries:
        if not name.isdigit():
            continue
        fields = _stat_fields(int(name))
        if not fields or fields[0] == "Z" or _session_of(fields) != sid:
            continue
        try:
            if int(fields[19]) >= not_before:
                out.append(int(name))
        except (ValueError, IndexError):
            continue
    return out


# --- die-with-parent ---


def _load_prctl():
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        # Resolved once, in the parent. The child between fork and exec must not import or
        # allocate: another thread may have held the import lock at the moment of the fork.
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


_PRCTL = _load_prctl()


def _die_with_parent(parent_pid: int):
    if _PRCTL is None:
        return None

    def preexec() -> None:
        _PRCTL(PR_SET_PDEATHSIG, signal.SIGTERM)
        # The parent may already have died between fork() and prctl(); then no signal will
        # ever come, and this child would be exactly the orphan we are trying to prevent.
        if os.getppid() != parent_pid:
            os._exit(1)

    return preexec


# --- the ledger ---


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ProcessRegistry:
    """Processes this run started and has not yet seen exit, mirrored to pids.json."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.stopping = False
        self._lock = threading.Lock()
        self._procs: dict[int, subprocess.Popen] = {}
        self._entries: dict[int, dict[str, Any]] = {}
        self._owner = {"pid": os.getpid(), "start_ticks": start_ticks(os.getpid())}
        self._active = True
        self._flush()

    def add(self, proc: subprocess.Popen, *, name: str, argv: list[str],
            port: int | None = None) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except (OSError, AttributeError):
            pgid = proc.pid
        with self._lock:
            self._procs[proc.pid] = proc
            self._entries[proc.pid] = {
                "pid": proc.pid, "pgid": pgid, "name": name, "port": port,
                "start_ticks": start_ticks(proc.pid), "started_at_utc": _now(),
                "argv": list(argv),
            }
            self._flush()

    def discard(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if self._procs.pop(proc.pid, None) is not None:
                self._entries.pop(proc.pid, None)
                self._flush()

    def live_names(self) -> list[str]:
        with self._lock:
            return [self._entries[pid]["name"] for pid, p in self._procs.items()
                    if p.poll() is None]

    def stop(self) -> list[str]:
        """Refuse any further launch and take down whatever is still running.

        The last line of defence: at the end of a run, normal teardown has already stopped
        everything, so anything this finds is a leak and is named in the return value.
        """
        from .deploy import terminate_process_group

        self.stopping = True
        with self._lock:
            leftover = [(self._entries[pid]["name"], p) for pid, p in self._procs.items()]
        leaked = []
        for name, proc in leftover:
            if proc.poll() is None:
                leaked.append(name)
            terminate_process_group(proc)
            self.discard(proc)
        return leaked

    def close(self) -> None:
        with self._lock:
            self._active = False
            self._flush()

    def _flush(self) -> None:
        # Called with the lock held (or from __init__).
        try:
            atomic_write_text(self.path, json.dumps({
                "run_id": self.run_id,
                "owner": {**self._owner, "active": self._active},
                "updated_at_utc": _now(),
                "processes": list(self._entries.values()),
            }, indent=2))
        except OSError:
            pass  # the ledger is a safety net; failing to write it must not fail a launch


_active: ProcessRegistry | None = None


def activate(path: Path, run_id: str) -> ProcessRegistry:
    global _active
    _active = ProcessRegistry(path, run_id)
    return _active


def deactivate(registry: ProcessRegistry) -> None:
    global _active
    registry.close()
    if _active is registry:
        _active = None


def start(argv: list[str], *, name: str, port: int | None = None, **popen_kwargs) -> subprocess.Popen:
    """Popen in a new session, dying with this process, recorded in the active ledger."""
    if _active is not None and _active.stopping:
        raise RunStopping(f"not starting {name}: the sweep is stopping")
    proc = subprocess.Popen(
        argv, start_new_session=True, preexec_fn=_die_with_parent(os.getpid()), **popen_kwargs,
    )
    if _active is not None:
        _active.add(proc, name=name, argv=argv, port=port)
    return proc


def untrack(proc: subprocess.Popen) -> None:
    if _active is not None and proc.poll() is not None:
        _active.discard(proc)


# --- finding and removing leftovers from a dead run ---


@dataclass
class Leftover:
    name: str
    pid: int
    port: int | None
    members: list[int] = field(default_factory=list)   # every live pid of its session

    def describe(self) -> str:
        where = f" on :{self.port}" if self.port else ""
        return f"{self.name}{where} (pids {', '.join(map(str, self.members))})"


@dataclass
class LedgerState:
    run_id: str = ""
    owner_alive: bool = False
    owner_pid: int | None = None
    leftovers: list[Leftover] = field(default_factory=list)


def _verified_members(entry: dict[str, Any]) -> list[int]:
    started = entry.get("start_ticks")
    if started is None:
        return []           # recorded without /proc; cannot prove anything is ours
    pid = int(entry["pid"])
    now = start_ticks(pid)
    if now is not None and now != started:
        return []           # the PID has been reused by an unrelated process
    return _session_members(int(entry.get("pgid", pid)), int(started))


def inspect(path: Path) -> LedgerState:
    """What a previous run's ledger says is still alive."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return LedgerState()
    owner = data.get("owner") or {}
    owner_pid = owner.get("pid")
    owner_alive = bool(
        owner.get("active") and owner_pid and owner_pid != os.getpid()
        and owner.get("start_ticks") is not None
        and start_ticks(int(owner_pid)) == owner.get("start_ticks")
    )
    state = LedgerState(run_id=data.get("run_id", ""), owner_alive=owner_alive,
                        owner_pid=owner_pid)
    for entry in data.get("processes", []):
        members = _verified_members(entry)
        if members:
            state.leftovers.append(Leftover(
                name=entry.get("name", "?"), pid=int(entry["pid"]), port=entry.get("port"),
                members=members,
            ))
    return state


def cleanup(path: Path, *, dry_run: bool = False, timeout_s: float = 30.0) -> list[str]:
    """SIGTERM, then SIGKILL, every verified leftover in a ledger. Returns what it did."""
    state = inspect(path)
    if state.owner_alive:
        return [f"run {state.run_id} is still running (pid {state.owner_pid}); stop that "
                f"process instead -- it tears its own servers down on SIGTERM"]
    if not state.leftovers:
        return ["nothing from this run is still alive"]

    actions = []
    targets: dict[int, int | None] = {}
    for lo in state.leftovers:
        actions.append(f"{'would stop' if dry_run else 'stopping'} {lo.describe()}")
        for pid in lo.members:
            targets[pid] = start_ticks(pid)
    if dry_run:
        return actions

    def still_ours(pid: int) -> bool:
        return targets[pid] is not None and start_ticks(pid, live_only=True) == targets[pid]

    for pid in targets:
        if still_ours(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and any(still_ours(p) for p in targets):
        time.sleep(0.25)
    for pid in targets:
        if still_ours(pid):       # re-verified: a PID freed during the wait is not touched
            try:
                os.kill(pid, signal.SIGKILL)
                actions.append(f"pid {pid} ignored SIGTERM for {timeout_s:.0f}s; sent SIGKILL")
            except (ProcessLookupError, PermissionError):
                pass

    try:
        data = json.loads(Path(path).read_text())
        data["processes"] = [e for e in data.get("processes", []) if _verified_members(e)]
        data["owner"] = {**(data.get("owner") or {}), "active": False}
        data["cleaned_at_utc"] = _now()
        atomic_write_text(Path(path), json.dumps(data, indent=2))
    except (OSError, ValueError):
        pass
    return actions


__all__ = [
    "PIDS_FILE", "ProcessRegistry", "RunStopping", "LedgerState", "Leftover",
    "activate", "deactivate", "start", "untrack", "inspect", "cleanup", "atomic_write_text",
    "start_ticks",
]
