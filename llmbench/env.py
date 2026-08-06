"""Host/env capture for provenance (spec non-negotiable #3: a number without provenance is
a bug) and observed-vs-requested tracking (course_correct.txt §5/§2.10).
"""
from __future__ import annotations

import dataclasses
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvSnapshot:
    hostname: str = ""
    os: str = ""
    python_version: str = ""
    cpu: str = ""
    cpu_count: int = 0
    gpu: list[str] = field(default_factory=list)
    gpu_driver: str | None = None
    ts_utc: str = ""

    # observed-vs-requested deltas, filled by backends where probeable
    requested: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _nvidia_smi_gpus() -> tuple[list[str], str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return [], None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return [], None
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        names = [l.split(",")[0].strip() for l in lines]
        driver = lines[0].split(",")[1].strip() if lines and "," in lines[0] else None
        return names, driver
    except (subprocess.SubprocessError, OSError):
        return [], None


def capture_env() -> EnvSnapshot:
    import datetime

    gpus, driver = _nvidia_smi_gpus()
    return EnvSnapshot(
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        cpu_count=_cpu_count(),
        gpu=gpus,
        gpu_driver=driver,
        ts_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def record_observed_vs_requested(snapshot: EnvSnapshot, axis: str, requested: Any, observed: Any) -> None:
    snapshot.requested[axis] = requested
    snapshot.observed[axis] = observed
