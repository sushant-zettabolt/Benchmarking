"""Process supervision: startup detection, failure classification, and teardown.

These drive real subprocesses. That is the point -- the properties under test are about
signals, process groups and file descriptors, none of which survive being mocked. What they
deliberately do NOT do is launch a real backend: `sh` stands in for llama-server, because
what is being tested is the supervision around it, not inference.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from llmbench.suite.deploy import (
    CapacityFailure, DeploymentError, LiveDeployment, ManagedProcess, spawn,
    terminate_process_group, wait_until_ready,
)
from tests.fake_server.server import FakeSSEServer


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def fake_server_process(tmp_path, script: str, *, name: str = "srv") -> ManagedProcess:
    return spawn(["sh", "-c", script], name=name, log_path=tmp_path / f"{name}.log",
                 env_overrides={}, port=None, cores=None)


# --- process groups ---


def test_terminate_takes_the_whole_process_group(tmp_path):
    """Both backends spawn children -- vLLM's engine core is a separate process. Signalling
    only the leader orphans it, and an orphaned engine keeps its port bound and its cores
    busy for the rest of the sweep."""
    pid_file = tmp_path / "child.pid"
    mp = fake_server_process(tmp_path, f"sleep 120 & echo $! > {pid_file}; wait")
    for _ in range(50):
        if pid_file.exists():
            break
        time.sleep(0.05)
    child = int(pid_file.read_text().strip())

    mp.terminate(timeout=5.0)

    assert _wait_gone(mp.pid), "the leader survived terminate()"
    assert _wait_gone(child), f"child {child} was orphaned rather than signalled"


def test_terminate_is_safe_on_an_already_dead_process(tmp_path):
    mp = fake_server_process(tmp_path, "exit 0")
    mp.proc.wait(timeout=5)
    mp.terminate(timeout=1.0)                       # must not raise
    terminate_process_group(mp.proc, timeout=1.0)   # nor must the helper it delegates to


def test_terminate_escalates_to_sigkill(tmp_path):
    """A server that ignores SIGTERM must not hold the sweep for the full timeout and then be
    left running anyway."""
    mp = fake_server_process(tmp_path, "trap '' TERM; sleep 120")
    t0 = time.monotonic()
    mp.terminate(timeout=1.0)
    assert _wait_gone(mp.pid), "SIGTERM was ignored and SIGKILL never followed"
    assert time.monotonic() - t0 < 15.0


# --- startup failure classification ---


def test_a_server_that_dies_on_allocation_is_capacity_not_error(tmp_path):
    """'This configuration did not fit' is a legitimate result -- the sweep records it and
    moves on. Reporting it as a harness error would abort a grid over context lengths at the
    first size that does not fit, which is exactly the size you were looking for."""
    mp = fake_server_process(
        tmp_path, "echo 'ggml_backend_alloc: failed to allocate buffer'; exit 1")

    with pytest.raises(CapacityFailure):
        asyncio.run(wait_until_ready(mp, "http://127.0.0.1:1/health",
                                     timeout_s=10.0, poll_s=0.05))


def test_a_server_that_dies_for_another_reason_is_an_error(tmp_path):
    mp = fake_server_process(tmp_path, "echo 'error while loading shared libraries'; exit 127")

    with pytest.raises(DeploymentError) as e:
        asyncio.run(wait_until_ready(mp, "http://127.0.0.1:1/health",
                                     timeout_s=10.0, poll_s=0.05))
    assert not isinstance(e.value, CapacityFailure)
    assert "exited during startup" in str(e.value)


def test_a_fatal_marker_caps_the_wait_instead_of_burning_the_startup_timeout(
        tmp_path, monkeypatch):
    """A backend can log a traceback and hang rather than exit. With `startup_timeout_s: 900`
    that used to cost fifteen idle minutes per sweep point -- hours across a grid -- because
    the only exit condition was the process actually dying."""
    import llmbench.suite.deploy as deploy_mod
    monkeypatch.setattr(deploy_mod, "FATAL_GRACE_S", 0.3)

    mp = fake_server_process(
        tmp_path, "echo 'Traceback (most recent call last):'; echo '  boom'; sleep 120")

    t0 = time.monotonic()
    with pytest.raises(DeploymentError) as e:
        asyncio.run(wait_until_ready(mp, "http://127.0.0.1:1/health",
                                     timeout_s=60.0, poll_s=0.05))
    elapsed = time.monotonic() - t0

    assert elapsed < 10.0, f"waited {elapsed:.1f}s despite a fatal marker in the log"
    assert "Traceback (most recent call last)" in str(e.value)
    mp.terminate(timeout=5.0)
    assert _wait_gone(mp.pid), "the hung server must be stopped, not left running"


def test_a_benign_log_does_not_trip_the_fatal_path(tmp_path):
    """The marker list is deliberately narrow: an earlier version matched bare 'error' and
    tripped on vLLM's 'Disabling Triton to prevent runtime errors.' banner. A slow-but-healthy
    startup must keep its full timeout."""
    mp = fake_server_process(
        tmp_path, "echo 'Disabling Triton to prevent runtime errors.'; sleep 120")

    with pytest.raises(DeploymentError) as e:
        asyncio.run(wait_until_ready(mp, "http://127.0.0.1:1/health",
                                     timeout_s=1.0, poll_s=0.05))
    assert "did not become ready within" in str(e.value)
    assert "logged" not in str(e.value)
    mp.terminate(timeout=5.0)


def test_a_server_that_becomes_healthy_is_accepted(tmp_path):
    """The happy path, and the guarantee that readiness comes from the endpoint rather than
    from log text -- deciding it from logs is what produced the premature-exit bugs in the
    earlier iteration of this harness."""
    async def drive():
        async with FakeSSEServer() as server:
            mp = fake_server_process(tmp_path, "sleep 30")
            try:
                await wait_until_ready(mp, f"{server.base_url}/health",
                                       timeout_s=10.0, poll_s=0.05)
            finally:
                mp.terminate(timeout=5.0)

    asyncio.run(drive())


# --- teardown ---


def test_teardown_reports_a_port_that_never_came_back(tmp_path):
    """A leftover of ours looks identical to somebody else's server to the next deployment's
    port pre-flight, which then refuses to launch with no clue whose it was."""
    async def drive():
        async with FakeSSEServer() as server:
            # A ManagedProcess that is already dead, claiming a port something else still
            # holds: the shape of a server whose child kept the socket open.
            mp = fake_server_process(tmp_path, "exit 0")
            mp.proc.wait(timeout=5)
            mp.port = server.port
            live = LiveDeployment(plan=_FakePlan(), processes=[mp])
            return await live.teardown(settle_s=0.0, port_timeout_s=1.0)

    stuck = asyncio.run(drive())
    assert stuck and "still bound" in stuck[0]


def test_teardown_of_a_clean_deployment_reports_nothing(tmp_path):
    mp = fake_server_process(tmp_path, "sleep 120")
    mp.port = _free_port()
    live = LiveDeployment(plan=_FakePlan(), processes=[mp])
    assert asyncio.run(live.teardown(settle_s=0.0)) == []
    assert _wait_gone(mp.pid)


class _FakePlan:
    id = "d000"


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- log handling ---


def test_spawn_does_not_leak_a_file_descriptor_per_launch(tmp_path):
    """A grid can launch hundreds of servers in one sweep."""
    before = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    procs = [fake_server_process(tmp_path, "sleep 30", name=f"s{i}") for i in range(20)]
    after = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    for mp in procs:
        mp.terminate(timeout=5.0)
    assert after - before < 10, f"{after - before} fds retained across 20 launches"
