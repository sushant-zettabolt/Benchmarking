"""Load balancing across a multi-instance deployment.

Two modes, and the choice changes what the numbers mean:

  `client` -- llmbench fans requests out to the instances itself (suite.lb.fanout). No proxy
              process exists, so nothing is added to the measured wire-to-wire window and no
              cores are spent forwarding. This is the default and the more accurate choice
              for measurement.

  `nginx`  -- a real reverse proxy in front of the fleet. The proxy hop IS inside the measured
              window, which is the point: it is the production topology. Rows produced this
              way are labelled so they are never silently compared with `client` rows.

`none` is the degenerate single-instance case: the client talks straight to one server.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ...backends.base import Backend
from ...backends.llamacpp import LlamaCppBackend
from ...backends.vllm import VllmBackend
from ..plan import DeploymentPlan
from ..spec import SuiteSpec
from .fanout import FanoutBackend

_BACKEND_CLASSES = {"llamacpp": LlamaCppBackend, "vllm": VllmBackend}


def make_backend(dep: DeploymentPlan, spec: SuiteSpec) -> Backend:
    """Build the Backend the runner will drive for this deployment.

    With `nginx` there is a single address to talk to, so this is one plain backend pointed at
    the proxy. With `client` it is a FanoutBackend over every instance. Either way the runner
    sees one object and needs no knowledge of the fleet.
    """
    cls = _BACKEND_CLASSES.get(dep.backend_spec.type)
    if cls is None:
        raise ValueError(f"no backend client for type {dep.backend_spec.type!r} ({dep.backend})")

    if dep.lb_kind == "nginx":
        return cls(dep.client_url, None, spec.request_timeout_s)

    members = [cls(inst.url, None, spec.request_timeout_s) for inst in dep.instances]
    if len(members) == 1:
        return members[0]
    return FanoutBackend(members, strategy=spec.lb.strategy)


async def start_load_balancer(dep: DeploymentPlan, spec: SuiteSpec, *, out_dir: Path):
    """Start nginx and wait for it to accept connections. Returns a ManagedProcess."""
    from ..deploy import DeploymentError, spawn
    from . import nginx

    exe, conf_path, prefix = nginx.prepare(dep, spec, out_dir=out_dir)
    argv = nginx.build_argv(exe, conf_path, prefix, dep, spec)

    mp = spawn(
        argv, name=f"{dep.id}-nginx", log_path=out_dir / "logs" / dep.id / "nginx.log",
        env_overrides={}, port=dep.lb_port, cores=None,
    )

    # nginx binds its listen socket before it is fully up; poll the socket rather than /health,
    # because a 502 here means the upstreams are not ready, not that nginx failed.
    from ..deploy import port_is_free

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not mp.alive:
            raise DeploymentError(
                f"nginx exited during startup (rc={mp.proc.returncode}).\n"
                f"config: {conf_path}\n--- log tail ---\n{mp.log_tail(40)}"
            )
        if not port_is_free(dep.lb_port):
            return mp
        await asyncio.sleep(0.2)

    tail = mp.log_tail(40)
    mp.terminate()
    raise DeploymentError(
        f"nginx did not start listening on :{dep.lb_port} within 30s.\n"
        f"config: {conf_path}\n--- log tail ---\n{tail}"
    )


__all__ = ["make_backend", "start_load_balancer", "FanoutBackend"]
