"""Launch, supervise and tear down a fleet of backend servers with explicit CPU/NUMA binding.

Safety rule, non-negotiable: this module only ever signals PIDs it started itself. It never
pattern-matches process names. A `pkill -f llama-server` on a shared box will happily kill a
colleague's week-old server, and that has already nearly happened on this host -- there is a
long-running llama-server on port 18080 belonging to another session. Every teardown path
here goes through `ManagedProcess`, which holds a real `Popen`.

The second half of that rule is the port pre-flight: before launching anything we check every
port we intend to use is actually free, and refuse to start if it is not. Binding would fail
anyway, but the useful part is refusing *before* tearing down the previous deployment.

Binding is applied two ways, because the backends respect different mechanisms:

  llama.cpp  numactl --physcpubind/--membind for placement, plus `-t N` and `--numa numactl`
             so ggml re-pins its worker threads to the numactl cpuset before each graph
             compute (ggml-cpu.c set_numa_thread_affinity). Note that ggml *clears* the
             affinity mask after each compute, so `taskset -cp <pid>` on the main PID reads
             back as the full machine even when pinning is working correctly -- verify
             per-thread via /proc/<pid>/task/*/status instead. `verify_affinity()` does this.

  vLLM       numactl for memory placement, plus VLLM_CPU_OMP_THREADS_BIND, which is what the
             CPU backend actually reads to place its OpenMP workers.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import procs
from .plan import DeploymentPlan, InstancePlan
from .spec import CpuSpec, SuiteSpec
from .topology import CoreSet, format_cpu_list, parse_cpu_list


class DeploymentError(RuntimeError):
    pass


class PortInUseError(DeploymentError):
    """A port we need is already served by a process we do not own."""


class CapacityFailure(DeploymentError):
    """The server refused to start in a way consistent with not fitting in memory.

    A legitimate data point ("this config does not fit"), not a harness bug -- the sweep
    records it and moves on rather than aborting.
    """


_OOM_MARKERS = (
    "out of memory", "cannot allocate memory", "failed to allocate",
    "oom", "killed", "std::bad_alloc", "cudamalloc failed",
    "no available memory for the cache blocks",
)

# Substrings that indicate a fatal startup failure. Deliberately narrow: an earlier version
# matched bare "error" and tripped on vLLM's benign
# "Disabling Triton to prevent runtime errors." banner.
_FATAL_MARKERS = (
    "Traceback (most recent call last)",
    "Engine core initialization failed",
    "RuntimeError:",
    "ValueError:",
    "AssertionError:",
    "error while loading shared libraries",
    "failed to load model",
)


def terminate_process_group(proc: subprocess.Popen, *, timeout: float = 30.0) -> None:
    """SIGTERM a process group we started, then SIGKILL what survives.

    Process *group*, because the things this suite launches spawn children (vLLM's engine
    core runs in a separate process, `vllm bench` likewise); signalling only the leader
    orphans them, and an orphaned vLLM engine keeps its port bound and its cores busy.

    Module-level rather than a method because the offline tool runner needs exactly the same
    escalation on timeout, and `subprocess.run(timeout=...)` does NOT do it -- it kills only
    the direct child, which with `start_new_session=True` leaves every grandchild alive.
    """
    try:
        _signal_group(proc, timeout=timeout)
    finally:
        procs.untrack(proc)


def _signal_group(proc: subprocess.Popen, *, timeout: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


@dataclass
class ManagedProcess:
    """A process we started, and therefore a process we may signal."""

    name: str
    proc: subprocess.Popen
    log_path: Path
    argv: list[str]
    env_overrides: dict[str, str] = field(default_factory=dict)
    port: int | None = None
    cores: CoreSet | None = None
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def log_tail(self, n_lines: int = 40) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n_lines:])

    def terminate(self, *, timeout: float = 30.0) -> None:
        terminate_process_group(self.proc, timeout=timeout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "pid": self.pid, "port": self.port,
            "argv": self.argv, "env": self.env_overrides,
            "log": str(self.log_path),
            "cpus": self.cores.physcpubind if self.cores else None,
        }


# --- port hygiene ---


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) != 0


def assert_ports_free(ports: list[int]) -> None:
    """Refuse to proceed if any target port is already served.

    This is the guard that keeps a sweep from colliding with an unrelated server on this
    shared host. We do not try to identify or stop whatever is listening -- that is somebody
    else's process by definition, since ours are not running yet.
    """
    busy = [p for p in ports if not port_is_free(p)]
    if busy:
        raise PortInUseError(
            f"port(s) {busy} are already in use by a process this run does not own. "
            f"Refusing to launch. Choose different ports via `backends.<name>.base_port` "
            f"/ `lb.port`, or stop whatever is listening -- llmbench will not kill processes "
            f"it did not start."
        )


async def wait_for_port_release(port: int, *, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if port_is_free(port):
            return True
        await asyncio.sleep(0.25)
    return False


# --- command construction ---


def numactl_prefix(cores: CoreSet, cpu: CpuSpec) -> list[str]:
    """Placement wrapper.

    `--physcpubind` rather than `--cpunodebind`: cpunodebind restricts to a node but still
    permits every logical cpu on it, including SMT siblings we deliberately excluded.
    physcpubind pins to exactly the cpus we allocated.
    """
    argv = ["numactl", f"--physcpubind={cores.physcpubind}"]
    if cpu.numa_policy == "membind" and cores.membind:
        argv.append(f"--membind={cores.membind_arg}")
    elif cpu.numa_policy == "interleave" and cores.membind:
        argv.append(f"--interleave={cores.membind_arg}")
    return argv


def llamacpp_server_argv(
    dep: DeploymentPlan, inst: InstancePlan, cpu: CpuSpec,
) -> tuple[list[str], dict[str, str]]:
    b = dep.backend_spec
    threads = dep.threads_per_instance or inst.cores.n_threads
    argv = numactl_prefix(inst.cores, cpu) + [
        "--",
        str(Path(b.server_bin).resolve()) if Path(b.server_bin).exists() else b.server_bin,
        "-m", b.model,
        "--host", "127.0.0.1",
        "--port", str(inst.port),
        "-t", str(threads),
        "-tb", str(threads),
        "-c", str(dep.n_ctx),
        "-np", str(dep.n_parallel),
        "-b", str(dep.batch),
        "-ub", str(dep.ubatch),
        "--metrics",           # exposes /metrics so the src=server path has data to read
    ]
    if b.served_model_name:
        argv += ["--alias", b.served_model_name]
    argv += list(b.extra_args)
    return argv, dict(b.env)


def vllm_server_argv(
    dep: DeploymentPlan, inst: InstancePlan, cpu: CpuSpec,
) -> tuple[list[str], dict[str, str]]:
    b = dep.backend_spec
    argv = numactl_prefix(inst.cores, cpu) + [
        "--",
        str(Path(b.server_bin).resolve()) if Path(b.server_bin).exists() else b.server_bin,
        "serve", b.model,
        "--host", "127.0.0.1",
        "--port", str(inst.port),
        "--max-model-len", str(dep.n_ctx),
        "--max-num-seqs", str(dep.n_parallel),
        "--max-num-batched-tokens", str(dep.batch),
    ]
    if b.served_model_name:
        argv += ["--served-model-name", b.served_model_name]
    argv += list(b.extra_args)

    env = dict(b.env)
    # The CPU backend places its OpenMP workers from this, independently of numactl.
    env.setdefault("VLLM_CPU_OMP_THREADS_BIND", inst.cores.physcpubind)
    env.setdefault("VLLM_CPU_KVCACHE_SPACE", "32")
    env.setdefault("OMP_NUM_THREADS", str(dep.threads_per_instance or inst.cores.n_threads))
    return argv, env


def build_server_command(
    dep: DeploymentPlan, inst: InstancePlan, cpu: CpuSpec,
) -> tuple[list[str], dict[str, str]]:
    # By engine type, not by name: `dep.backend` may be a named variant (llamacpp-zendnn-q8).
    kind = dep.backend_spec.type
    if kind == "llamacpp":
        return llamacpp_server_argv(dep, inst, cpu)
    if kind == "vllm":
        return vllm_server_argv(dep, inst, cpu)
    raise DeploymentError(f"no server launcher for backend type {kind!r} ({dep.backend})")


def health_url(dep: DeploymentPlan, inst: InstancePlan) -> str:
    return f"{inst.url}/health"


# --- launching ---


def _keep_previous_log(log_path: Path) -> None:
    """Move an existing log aside rather than truncating it.

    Deployment ids restart at d000 every run, so a resumed or repeated sweep relaunches into
    the same log path -- and the log it would overwrite is usually the one explaining why the
    previous attempt failed.
    """
    if not log_path.exists() or not log_path.stat().st_size:
        return
    n = 1
    while (archived := log_path.with_name(f"{log_path.stem}.{n}{log_path.suffix}")).exists():
        n += 1
    log_path.rename(archived)


def spawn(
    argv: list[str], *, name: str, log_path: Path, env_overrides: dict[str, str],
    port: int | None = None, cores: CoreSet | None = None,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(env_overrides)

    _keep_previous_log(log_path)
    header = (
        f"# llmbench managed process: {name}\n"
        f"# argv: {shlex.join(argv)}\n"
        f"# env : {' '.join(f'{k}={v}' for k, v in sorted(env_overrides.items()))}\n"
        f"# cpus: {cores.physcpubind if cores else 'unpinned'}\n\n"
    )
    with open(log_path, "w") as log_f:
        log_f.write(header)
        log_f.flush()
        # Own session, so teardown can killpg; recorded in pids.json and set to die with
        # this process, so a sweep killed outright does not strand it (see procs.py).
        proc = procs.start(
            argv, name=name, port=port, stdout=log_f, stderr=subprocess.STDOUT, env=env,
        )
    return ManagedProcess(
        name=name, proc=proc, log_path=log_path, argv=argv,
        env_overrides=env_overrides, port=port, cores=cores,
    )


# How long a server gets to become healthy *after* a fatal marker appears in its log before
# we stop waiting. A marker is not proof of death -- a backend can log a handled traceback and
# recover -- so it shortens the wait rather than ending it outright.
FATAL_GRACE_S = 20.0


def _startup_failure(mp: ManagedProcess, tail: str, what: str) -> DeploymentError:
    """Classify a failed startup: 'did not fit' is a data point, anything else is an error."""
    if any(m in tail.lower() for m in _OOM_MARKERS):
        return CapacityFailure(
            f"{mp.name} {what} with an allocation-failure marker -- treating as "
            f"'did not fit'. See {mp.log_path}"
        )
    return DeploymentError(
        f"{mp.name} {what}. See {mp.log_path}\n--- log tail ---\n{tail}"
    )


async def wait_until_ready(
    mp: ManagedProcess, url: str, *, timeout_s: float, poll_s: float = 1.0,
) -> None:
    """Poll /health until 200, watching for early exit and fatal log markers.

    Polling the endpoint is authoritative: readiness is never inferred from log text, which
    is what produced the premature-exit bugs in the earlier iteration of this harness. Log
    scraping does two narrower jobs. It decides *which* error to raise once the process is
    already dead, and -- for a process that is still alive but has clearly failed -- it caps
    the remaining wait at `FATAL_GRACE_S` instead of burning the full startup timeout. A
    deployment that dies on its config with `startup_timeout_s: 900` otherwise costs fifteen
    idle minutes per sweep point, which on a grid is hours of nothing.
    """
    deadline = time.monotonic() + timeout_s
    polls = 0
    fatal_marker: str | None = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if not mp.alive:
                raise _startup_failure(
                    mp, mp.log_tail(60), f"exited during startup (rc={mp.proc.returncode})",
                )
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass

            polls += 1
            if fatal_marker is None and polls % 5 == 0:
                # Every fifth poll only: these logs reach megabytes and re-reading one every
                # second for fifteen minutes is real I/O for no extra information.
                tail = mp.log_tail(200)
                fatal_marker = next((m for m in _FATAL_MARKERS if m in tail), None)
                if fatal_marker:
                    deadline = min(deadline, time.monotonic() + FATAL_GRACE_S)

            await asyncio.sleep(poll_s)

    tail = mp.log_tail(60)
    mp.terminate()
    if fatal_marker:
        raise _startup_failure(
            mp, tail,
            f"logged {fatal_marker!r} and was still not serving /health "
            f"{FATAL_GRACE_S:.0f}s later",
        )
    raise _startup_failure(mp, tail, f"did not become ready within {timeout_s:.0f}s")


def _online_cpus() -> set[int]:
    try:
        return set(parse_cpu_list(Path("/sys/devices/system/cpu/online").read_text().strip()))
    except (OSError, ValueError):
        return set(range(os.cpu_count() or 1))


def verify_affinity(mp: ManagedProcess) -> dict[str, Any]:
    """Read the real per-thread CPU masks from /proc and classify them.

    `taskset -cp <pid>` is not usable here. ggml calls clear_numa_thread_affinity()
    (ggml-cpu.c) after each graph compute, which resets the calling thread's mask to every CPU
    on the machine; the main thread therefore reads back as the full host even when every
    worker is correctly pinned. Observed on this box: 401 of 402 llama-server threads inside
    the allocation, one reset to 0-383.

    So a thread outside the allocation is only a real failure if it is pinned somewhere
    *else*. A thread whose mask is exactly the set of online CPUs has simply had its affinity
    cleared, which is expected and harmless -- it is re-pinned before the next compute.
    """
    if not mp.alive or mp.cores is None:
        return {"verified": False, "reason": "process not running or unpinned"}

    allowed = set(mp.cores.cpus)
    host = _online_cpus()
    task_dir = Path(f"/proc/{mp.pid}/task")
    inside = cleared = elsewhere = unknown = 0
    observed: set[int] = set()
    stray: set[int] = set()
    try:
        for t in task_dir.iterdir():
            status = t / "status"
            try:
                for line in status.read_text().splitlines():
                    if not line.startswith("Cpus_allowed_list:"):
                        continue
                    cpus = set(parse_cpu_list(line.split(":", 1)[1].strip()))
                    observed |= cpus
                    if cpus <= allowed:
                        inside += 1
                    elif cpus >= host:
                        cleared += 1
                    else:
                        elsewhere += 1
                        stray |= cpus - allowed
                    break
                else:
                    unknown += 1
            except (OSError, ValueError):
                unknown += 1
    except OSError as e:
        return {"verified": False, "reason": f"cannot read {task_dir}: {e}"}

    out = {
        "verified": elsewhere == 0 and inside > 0,
        "threads_total": inside + cleared + elsewhere + unknown,
        "threads_within_allocation": inside,
        "threads_affinity_cleared": cleared,
        "threads_on_other_cpus": elsewhere,
        "threads_unreadable": unknown,
        "requested_cpus": mp.cores.physcpubind,
        "observed_cpu_union": format_cpu_list(observed),
    }
    if cleared:
        out["note"] = (
            f"{cleared} thread(s) have an unrestricted mask. This is ggml's "
            f"clear_numa_thread_affinity() resetting affinity after a graph compute, not a "
            f"placement failure; workers are re-pinned before the next compute."
        )
    if elsewhere:
        out["stray_cpus"] = format_cpu_list(stray)
        out["error"] = (
            f"{elsewhere} thread(s) are pinned to cpus outside the allocation "
            f"({format_cpu_list(stray)}) -- this deployment is not core-isolated and its "
            f"numbers should not be compared against correctly-placed ones"
        )
    return out


# A deployment whose resident memory sits less than this fraction on its bound NUMA nodes is
# reading weights across the interconnect. Measured on Turin (2026-09-29): an mmap'd Qwen3.6 GGUF
# whose page cache another pod had left on the other socket ran llama.cpp prefill at 0.55x.
MEMORY_LOCAL_MIN_FRACTION = 0.9


def _numa_resident_kb(pid: int) -> dict[int, int]:
    """Resident kB per NUMA node over every mapping of a process."""
    try:
        return parse_numa_maps(Path(f"/proc/{pid}/numa_maps").read_text())
    except OSError:
        return {}


def parse_numa_maps(text: str) -> dict[int, int]:
    """Resident kB per NUMA node from /proc/<pid>/numa_maps text: every mapping's N<node>=<pages>
    counts, times that mapping's kernelpagesize_kB (4 unless stated, e.g. 2048 for huge pages)."""
    per_node: dict[int, int] = {}
    for line in text.splitlines():
        page_kb = 4
        counts = []
        for tok in line.split()[2:]:
            if tok.startswith("kernelpagesize_kB="):
                page_kb = int(tok.split("=", 1)[1])
            elif tok[0] == "N" and "=" in tok:
                node, n = tok[1:].split("=", 1)
                if node.isdigit() and n.isdigit():
                    counts.append((int(node), int(n)))
        for node, n in counts:
            per_node[node] = per_node.get(node, 0) + n * page_kb
    return per_node


def verify_memory_placement(mp: ManagedProcess) -> dict[str, Any]:
    """Where a server's memory actually is, per NUMA node, against the nodes it was bound to.

    `numactl --membind` only governs pages the server allocates itself. An mmap'd model runs from
    the host-wide page cache, which stays on whatever node first read the file -- possibly
    another pod's, on the other socket -- and --membind does not move it. So a server can be
    pinned perfectly (verify_affinity) and still read every weight remotely. Summed over the
    server and its children (vLLM keeps the weights in its worker process).
    """
    if not mp.alive or mp.cores is None:
        return {"verified": False, "reason": "process not running or unpinned"}
    from .contention import _descendant_pids

    per_node: dict[int, int] = {}
    for pid in _descendant_pids(mp.pid):
        for node, kb in _numa_resident_kb(pid).items():
            per_node[node] = per_node.get(node, 0) + kb
    total = sum(per_node.values())
    bound = list(mp.cores.membind)
    out: dict[str, Any] = {
        "membind": bound,
        "resident_gb": round(total / 1048576, 1),
        "per_node_gb": {str(n): round(kb / 1048576, 1) for n, kb in sorted(per_node.items())},
    }
    if not total:
        out.update(verified=False, reason=f"no numa_maps readable for pid {mp.pid}")
        return out
    if not bound:
        out.update(verified=True, reason="no membind requested")
        return out
    local = sum(kb for n, kb in per_node.items() if n in bound) / total
    out["local_fraction"] = round(local, 3)
    out["verified"] = local >= MEMORY_LOCAL_MIN_FRACTION
    if not out["verified"]:
        remote = {n: round(kb / 1048576, 1) for n, kb in per_node.items() if n not in bound and kb}
        out["error"] = (
            f"only {100 * local:.0f}% of the server's {total / 1048576:.0f} GB resident memory is on "
            f"its bound node(s) {bound}; {remote} GB on other nodes. An mmap'd model's page cache "
            f"stays where the first reader left it; load with --load-mode none/dio, or drop the "
            f"file from the page cache, to measure with local weights"
        )
    return out


@dataclass
class LiveDeployment:
    """A running fleet. Always use as an async context manager so teardown is guaranteed."""

    plan: DeploymentPlan
    processes: list[ManagedProcess] = field(default_factory=list)
    lb_process: ManagedProcess | None = None
    affinity: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    started_at_utc: str = ""
    launch_seconds: float = 0.0

    @property
    def urls(self) -> list[str]:
        return self.plan.urls

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.plan.id,
            "started_at_utc": self.started_at_utc,
            "launch_seconds": round(self.launch_seconds, 2),
            "processes": [p.to_dict() for p in self.processes],
            "lb_process": self.lb_process.to_dict() if self.lb_process else None,
            "affinity": self.affinity,
            "memory": self.memory,
        }

    async def teardown(self, *, settle_s: float = 3.0, port_timeout_s: float = 30.0) -> list[str]:
        """Stop everything this deployment started and wait for its ports to come back.

        Returns any port that never freed. That matters: the next deployment's
        `assert_ports_free` pre-flight would fail on it, and knowing it was *our* leftover
        rather than somebody else's server is the difference between a fixable bug and an
        unexplained refusal to launch.
        """
        for mp in [self.lb_process, *reversed(self.processes)]:
            if mp is not None:
                mp.terminate()
        ports = [p.port for p in self.processes if p.port]
        if self.lb_process and self.lb_process.port:
            ports.append(self.lb_process.port)
        # Concurrently: these waits are independent, and serialising them turns one wedged
        # server into `30s x n_instances` of dead time before the next deployment.
        released = await asyncio.gather(*(
            wait_for_port_release(port, timeout_s=port_timeout_s) for port in ports
        ))
        stuck = [
            f"port {port} was still bound {port_timeout_s:.0f}s after teardown of "
            f"{self.plan.id}; the next "
            f"deployment's port pre-flight will refuse to start on it"
            for port, ok in zip(ports, released) if not ok
        ]
        if settle_s > 0:
            # Let page cache / memory pressure settle before the next launch, so deployment
            # N+1 does not inherit deployment N's cleanup cost as a slow first request.
            await asyncio.sleep(settle_s)
        return stuck


async def launch_deployment(
    dep: DeploymentPlan, spec: SuiteSpec, *, out_dir: Path,
) -> LiveDeployment:
    """Bring up every instance in a deployment, then the load balancer if there is one.

    Instances are launched concurrently but waited on together: on a CPU box each server
    spends most of its startup reading weights, and serialising that would multiply the
    sweep's wall-clock by the instance count.
    """
    import datetime

    from .lb import start_load_balancer

    ports = [i.port for i in dep.instances]
    if dep.lb_kind == "nginx":
        ports.append(dep.lb_port)
    assert_ports_free(ports)

    log_dir = out_dir / "logs" / dep.id
    live = LiveDeployment(
        plan=dep,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    t0 = time.monotonic()

    try:
        for inst in dep.instances:
            argv, env = build_server_command(dep, inst, spec.cpu)
            mp = spawn(
                argv, name=f"{dep.id}-{dep.backend}-i{inst.index}",
                log_path=log_dir / f"server-i{inst.index}.log",
                env_overrides=env, port=inst.port, cores=inst.cores,
            )
            live.processes.append(mp)

        await asyncio.gather(*(
            wait_until_ready(mp, health_url(dep, inst), timeout_s=spec.startup_timeout_s)
            for mp, inst in zip(live.processes, dep.instances)
        ))

        live.affinity = [verify_affinity(mp) for mp in live.processes]
        live.memory = [verify_memory_placement(mp) for mp in live.processes]

        if dep.lb_kind == "nginx":
            live.lb_process = await start_load_balancer(dep, spec, out_dir=out_dir)
    except BaseException:
        await live.teardown(settle_s=0.0)
        raise

    live.launch_seconds = time.monotonic() - t0
    return live


__all__ = [
    "LiveDeployment", "ManagedProcess", "launch_deployment", "build_server_command",
    "numactl_prefix", "verify_affinity", "verify_memory_placement", "assert_ports_free", "port_is_free",
    "terminate_process_group", "DeploymentError", "CapacityFailure", "PortInUseError",
]
