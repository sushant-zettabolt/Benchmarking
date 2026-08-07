"""Shared plumbing for the native offline benchmark drivers.

"Offline" here means each backend's *own* benchmark tool, driven as a subprocess, with its
output parsed into a common row shape. It explicitly does NOT mean a common measurement
boundary -- these tools measure different things:

  `llama-bench` times `llama_decode()` in-process. No HTTP, no scheduler, no admission
  control, no sampling of a real request. It is llama.cpp's best case.

  `llama-batched-bench` also runs in-process but does drive `-npl` sequences through one
  batch, so it has a real notion of static batch size.

  `vllm bench latency` runs the full vLLM engine offline -- scheduler, paged KV, continuous
  batching -- just without a network hop.

So a llama.cpp offline number and a vLLM offline number are not the same measurement, and
every row this package produces carries `comparable_across_backends=False`. What they ARE
good for is comparing each backend to *itself* online: offline minus online is that backend's
HTTP-plus-scheduler tax, which is a real and useful quantity.
"""
from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..plan import OfflinePlan
from ..spec import CpuSpec


@dataclass
class OfflineResult:
    """One measured point from a native tool, normalised into common fields.

    Any metric a given tool does not report stays None rather than being derived from a
    guess. `llama-bench` for instance reports a single t/s for a pp-only or tg-only test and
    has no separate prefill/decode split within one row.
    """

    plan_id: str
    backend: str
    tool: str
    test: str
    batch_size: int
    n_prompt: int
    n_gen: int
    rep: int = 0

    pp_tps: float | None = None        # prefill tokens/s
    tg_tps: float | None = None        # decode tokens/s
    total_tps: float | None = None     # all tokens / total wall time
    latency_s: float | None = None     # wall time for one batch iteration

    n_threads: int | None = None
    cpus: str = ""
    comparable_across_backends: bool = False

    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    stdout_path: str = ""
    returncode: int | None = None
    error: str | None = None
    duration_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ToolRun:
    argv: list[str]
    env: dict[str, str]
    stdout: str
    stderr: str
    returncode: int
    duration_s: float
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def numactl_wrap(argv: list[str], plan: OfflinePlan, cpu: CpuSpec) -> list[str]:
    from ..deploy import numactl_prefix

    return numactl_prefix(plan.cores, cpu) + ["--"] + argv


def run_tool(
    argv: list[str], *, env_overrides: dict[str, str], log_path: Path, timeout_s: float,
) -> ToolRun:
    """Run a native tool to completion, capturing stdout separately from the log.

    stdout is captured rather than redirected because it carries the machine-readable results
    for two of the three tools; a copy still lands in the log alongside stderr so a failed run
    is diagnosable.

    Popen rather than `subprocess.run(timeout=...)`, deliberately. `run()` reacts to a timeout
    by killing the direct child only; combined with `start_new_session=True` that leaves every
    grandchild alive, and `vllm bench` runs its engine in a child process. A timed-out offline
    trial would then leave an orphaned engine holding the cores and the KV allocation for the
    rest of the sweep -- silently poisoning every measurement after it. Every exit path here
    goes through `terminate_process_group`, which signals the group we created and nothing
    else (the same rule deploy.py follows: never pattern-match process names).
    """
    from ..deploy import terminate_process_group

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(env_overrides)

    t0 = time.monotonic()
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            start_new_session=True,
        )
    except OSError as e:
        rc, out, err = -1, "", f"[llmbench] failed to execute: {e}"
    else:
        try:
            out, err = proc.communicate(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            terminate_process_group(proc)
            out, err = _drain(proc)
            rc = -1
            err += f"\n[llmbench] timed out after {timeout_s:.0f}s; process group killed"
        except BaseException:
            # Ctrl-C included: the tool must not outlive the sweep that started it.
            terminate_process_group(proc)
            raise
    duration = time.monotonic() - t0

    log_path.write_text(
        f"# argv: {shlex.join(argv)}\n"
        f"# env : {' '.join(f'{k}={v}' for k, v in sorted(env_overrides.items()))}\n"
        f"# rc  : {rc}   duration: {duration:.1f}s\n\n"
        f"--- stdout ---\n{out}\n--- stderr ---\n{err}\n"
    )
    return ToolRun(argv=argv, env=env_overrides, stdout=out, stderr=err,
                   returncode=rc, duration_s=duration, log_path=log_path)


def _drain(proc: subprocess.Popen) -> tuple[str, str]:
    """Collect whatever the tool managed to write before it was killed.

    Bounded: if a grandchild we could not signal still holds the pipe open, `communicate()`
    would block forever and the sweep would hang instead of recording a timeout.
    """
    try:
        out, err = proc.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:
        return "", "[llmbench] output pipes still held open after the process group was killed"
    return out or "", err or ""


def failed_result(plan: OfflinePlan, run: ToolRun, reason: str) -> OfflineResult:
    tail = "\n".join((run.stderr or run.stdout).splitlines()[-15:])
    return OfflineResult(
        plan_id=plan.id, backend=plan.backend, tool=plan.tool, test=plan.test_name(),
        batch_size=plan.batch_size, n_prompt=plan.n_prompt, n_gen=plan.n_gen,
        cpus=plan.cores.physcpubind, argv=run.argv, env=run.env,
        stdout_path=str(run.log_path), returncode=run.returncode,
        error=f"{reason}: {tail}" if tail else reason, duration_s=run.duration_s,
    )


__all__ = ["OfflineResult", "ToolRun", "run_tool", "numactl_wrap", "failed_result"]
