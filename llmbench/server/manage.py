"""--server-mode manage (spec §7, milestone 5). Launches/tears down a backend server per
Group-2 combination, from a `--server-cmd` template. Loop ordering (Group-2 outer, Group-1
inner against one live server) is handled by config.get_cmd_params_instances()'s nesting
(docs/reference-notes.md §1) -- this module only owns one server's lifecycle at a time.
"""
from __future__ import annotations

import asyncio
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


class ServerStartupError(RuntimeError):
    pass


class ServerCapacityFailure(ServerStartupError):
    """Server process exited/refused to start in a way consistent with a capacity limit
    (e.g. OOM from n_ctx * n_seq_max) -- a legitimate data point ('did not fit'), spec §7,
    not a benchmark bug."""


_OOM_MARKERS = ("out of memory", "cudaMalloc failed", "cannot allocate memory", "OOM", "killed")


@dataclass
class ManagedServer:
    proc: subprocess.Popen
    log_path: Path
    port: int

    def terminate(self, *, timeout: float = 10.0) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5.0)


def render_server_cmd(template: str, flags: dict) -> list[str]:
    rendered = template.format(**flags)
    return shlex.split(rendered)


def launch_server(cmd_template: str, flags: dict, *, out_dir: str | Path, tag: str) -> ManagedServer:
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"
    argv = render_server_cmd(cmd_template, flags)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT)
    port = flags.get("port", 8080)
    return ManagedServer(proc=proc, log_path=log_path, port=int(port))


async def wait_for_ready(
    server: ManagedServer, *, health_url: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0
) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if server.proc.poll() is not None:
                log_text = server.log_path.read_text(errors="replace")
                if any(marker.lower() in log_text.lower() for marker in _OOM_MARKERS):
                    raise ServerCapacityFailure(
                        f"server process exited (code {server.proc.returncode}) with an "
                        f"OOM/allocation-failure marker in its log -- treating as 'did not fit', "
                        f"see {server.log_path}"
                    )
                raise ServerStartupError(
                    f"server process exited early (code {server.proc.returncode}); see {server.log_path}"
                )
            try:
                r = await client.get(health_url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll_interval_s)
    server.terminate()
    raise ServerStartupError(f"server did not become ready within {timeout_s}s; see {server.log_path}")


async def wait_for_port_release(port: int, *, timeout_s: float = 15.0) -> None:
    import socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        await asyncio.sleep(0.2)
