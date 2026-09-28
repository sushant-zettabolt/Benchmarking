"""What the sweep ran on: the host (or Kubernetes pod) and the server software's versions.

Captured once per run attempt and stored in run.json as `host` and `software`, so a result
file says which machine, which cpuset and which builds produced it -- not just a hostname.
Everything here is best-effort: an unreadable file or a binary that will not print its
version yields an empty field, never an exception that costs the run.
"""
from __future__ import annotations

import os
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any

_K8S_NAMESPACE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
_VERSION_TIMEOUT_S = 120      # `vllm --version` imports torch; allow for a cold NFS page cache
_PY_PACKAGES = ("vllm", "torch", "zentorch", "transformers", "compressed-tensors")


def _read(path: str | Path) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _field(text: str, key: str, sep: str = ":") -> str:
    for line in text.splitlines():
        k, _, v = line.partition(sep)
        if k.strip() == key:
            return v.strip().strip('"')
    return ""


def _kib_to_gib(value: str) -> float | None:
    try:
        return round(int(value.split()[0]) / 1024 / 1024, 1)
    except (ValueError, IndexError):
        return None


def _numa_nodes() -> list[dict[str, Any]]:
    nodes = []
    for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"),
                    key=lambda p: int(p.name[4:])):
        meminfo = _read(d / "meminfo")
        total = free = ""
        for line in meminfo.splitlines():
            # "Node 5 MemTotal:       297145552 kB"
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "MemTotal:":
                total = parts[3]
            elif len(parts) >= 4 and parts[2] == "MemFree:":
                free = parts[3]
        nodes.append({
            "node": int(d.name[4:]), "cpus": _read(d / "cpulist"),
            "mem_total_gib": _kib_to_gib(total), "mem_free_gib": _kib_to_gib(free),
        })
    return nodes


def _memlock() -> str:
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return ""
    return "unlimited" if soft == resource.RLIM_INFINITY else f"{soft // 1024} KiB"


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def capture_host() -> dict[str, Any]:
    status = _read("/proc/self/status")
    meminfo = _read("/proc/meminfo")
    namespace = _read(_K8S_NAMESPACE)
    return {
        "hostname": platform.node(),
        "kubernetes": {
            # Inside a pod the hostname is the pod name; the node name is only visible if the
            # pod spec exposes it through the downward API.
            "in_pod": bool(os.environ.get("KUBERNETES_SERVICE_HOST") or namespace),
            "pod": platform.node() if namespace else "",
            "namespace": namespace,
            "node": os.environ.get("NODE_NAME") or os.environ.get("K8S_NODE_NAME") or "",
        },
        "os": _field(_read("/etc/os-release"), "PRETTY_NAME", "="),
        "kernel": platform.release(),
        "cpu_model": _field(_read("/proc/cpuinfo"), "model name"),
        "cpus_online": _read("/sys/devices/system/cpu/online"),
        # What this process may actually use -- in a pod, the cpuset, which can be far
        # narrower than what `numactl -H` or `cpus_online` shows.
        "cpus_allowed": _field(status, "Cpus_allowed_list"),
        "mems_allowed": _field(status, "Mems_allowed_list"),
        "mem_total_gib": _kib_to_gib(_field(meminfo, "MemTotal")),
        "numa_nodes": _numa_nodes(),
        "cgroup": {
            "cpu_max": _read("/sys/fs/cgroup/cpu.max"),
            "memory_max": _read("/sys/fs/cgroup/memory.max"),
        },
        "memlock_limit": _memlock(),
        "python": platform.python_version(),
        "llmbench_commit": _git_commit(),
    }


def _run_version(argv: list[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable ({type(e).__name__})"
    lines = [l.strip() for l in (out.stdout + "\n" + out.stderr).splitlines() if l.strip()]
    # llama-server prints "version: N (hash)" and "built with ..."; keep just those.
    keep = [l for l in lines if l.startswith(("version", "built with"))] or lines[-1:]
    return " | ".join(keep)


def _python_packages(server_bin: str) -> dict[str, str]:
    """Package versions from the venv a Python server (vLLM) runs in, via importlib.metadata
    -- which reads dist-info and does not import the (slow) packages themselves."""
    python = Path(server_bin).parent / "python"
    if not python.exists():
        return {}
    code = (
        "import importlib.metadata as m\n"
        f"for p in {list(_PY_PACKAGES)!r}:\n"
        "    try: print(p, m.version(p))\n"
        "    except m.PackageNotFoundError: pass\n"
    )
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)


def capture_software(backends: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Version of every distinct server binary the sweep will launch, keyed by path."""
    out: dict[str, dict[str, Any]] = {}
    for b in backends.values():
        path = getattr(b, "server_bin", "") or ""
        if not path or path in out:
            continue
        entry: dict[str, Any] = {"type": b.type, "resolved": str(Path(path).resolve())}
        entry["version"] = _run_version([path, "--version"])
        if b.type == "vllm":
            entry["packages"] = _python_packages(path)
        out[path] = entry
    return out


__all__ = ["capture_host", "capture_software"]
