"""Sweep specification: the YAML schema, its validation, and its expansion into trials.

The spec is deliberately split into two axis groups with different costs:

  `deployment` axes change how a *server* is launched (backend, instance count, core split,
  n_ctx, n_parallel, batch/ubatch). Changing one means tearing down and relaunching every
  instance -- minutes on a CPU box with an 8B model.

  `workload` axes change only what the *client* sends (prompt/gen lengths, concurrency,
  request rate). They run against a deployment that is already live and warm.

Expansion therefore nests deployment outermost so a run relaunches servers as rarely as
possible. This is the same rationale as `config.NESTING_ORDER` in the single-endpoint path
(docs/reference-notes.md §1, after llama-bench.cpp:1294-1435), applied one level up.

Every axis accepts either a scalar or a list; scalars are promoted to single-element lists so
`n_ctx: 4096` and `n_ctx: [4096]` mean the same thing.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from ..config import parse_int_range
from .topology import SmtPolicy

BackendName = Literal["llamacpp", "vllm"]
BACKENDS: tuple[str, ...] = ("llamacpp", "vllm")
_BACKEND_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _expand(value: Any) -> str:
    return os.path.expanduser(str(value)) if value else ""


class SpecError(ValueError):
    """A malformed spec. Always names the offending key."""


# --- helpers ---


def _as_list(value: Any, key: str) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _int_axis(value: Any, key: str, *, default: list[int] | None = None) -> list[int]:
    """Accept 4096, [1,2,4], or llama-bench range syntax "1-16*2"."""
    if value is None:
        return list(default) if default is not None else []
    out: list[int] = []
    for item in _as_list(value, key):
        if isinstance(item, bool):
            raise SpecError(f"{key}: expected an integer, got boolean {item!r}")
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str):
            try:
                out.extend(parse_int_range(item, allow_negative=True))
            except ValueError as e:
                raise SpecError(f"{key}: {e}") from e
        else:
            raise SpecError(f"{key}: expected int or range string, got {type(item).__name__}")
    if not out:
        raise SpecError(f"{key}: expanded to an empty list")
    return out


def _str_axis(value: Any, key: str, *, default: list[str] | None = None,
              choices: Iterable[str] | None = None) -> list[str]:
    if value is None:
        return list(default) if default is not None else []
    out = [str(v) for v in _as_list(value, key)]
    if choices is not None:
        allowed = set(choices)
        bad = [v for v in out if v not in allowed]
        if bad:
            raise SpecError(f"{key}: unknown value(s) {bad}; allowed: {sorted(allowed)}")
    return out


def _one_of(value: Any, key: str, choices: Iterable[str], default: str) -> str:
    if value is None:
        return default
    value = str(value)
    if value not in set(choices):
        raise SpecError(f"{key}: {value!r} is not one of {sorted(choices)}")
    return value


def _unknown_keys(data: dict, known: Iterable[str], where: str) -> None:
    """Fail loudly on typos. A silently-ignored `instaces: 4` would run the wrong sweep."""
    extra = sorted(set(data) - set(known))
    if extra:
        raise SpecError(f"{where}: unknown key(s) {extra}; known keys: {sorted(known)}")


# --- sections ---


@dataclass
class CpuSpec:
    # Node 0 by default. Both sockets are 96 physical cores, so either satisfies a "96 cores"
    # constraint; node 0 is the quieter one on this host. Always confirm with the contention
    # figure in the run manifest rather than assuming -- this is a shared machine.
    budget: str = "0-95"
    smt: SmtPolicy = "exclude"
    ccd_align: bool = True
    reserve: int = 0
    membind: str = "auto"
    numa_policy: str = "membind"   # membind | interleave | none

    @classmethod
    def from_dict(cls, data: dict | None) -> "CpuSpec":
        data = data or {}
        _unknown_keys(data, ("budget", "smt", "ccd_align", "reserve", "membind", "numa_policy"), "cpu")
        return cls(
            budget=str(data.get("budget", "0-95")),
            smt=_one_of(data.get("smt"), "cpu.smt", ("exclude", "include", "only-siblings"), "exclude"),
            ccd_align=bool(data.get("ccd_align", True)),
            reserve=int(data.get("reserve", 0)),
            membind=str(data.get("membind", "auto")),
            numa_policy=_one_of(data.get("numa_policy"), "cpu.numa_policy",
                                ("membind", "interleave", "none"), "membind"),
        )


@dataclass
class LbSpec:
    kind: str = "client"          # client | nginx | none
    cpus: str | None = None       # None => auto (see topology.auxiliary_cpus)
    n_cpus: int = 4
    port: int = 18081
    strategy: str = "least-outstanding"   # least-outstanding | round-robin
    uniform: bool = True
    nginx_bin: str = "nginx"
    worker_processes: int | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "LbSpec":
        data = data or {}
        _unknown_keys(data, ("kind", "cpus", "n_cpus", "port", "strategy", "uniform",
                             "nginx_bin", "worker_processes"), "lb")
        return cls(
            kind=_one_of(data.get("kind"), "lb.kind", ("client", "nginx", "none"), "client"),
            cpus=str(data["cpus"]) if data.get("cpus") else None,
            n_cpus=int(data.get("n_cpus", 4)),
            port=int(data.get("port", 18081)),
            strategy=_one_of(data.get("strategy"), "lb.strategy",
                             ("least-outstanding", "round-robin"), "least-outstanding"),
            uniform=bool(data.get("uniform", True)),
            nginx_bin=str(data.get("nginx_bin", "nginx")),
            worker_processes=(int(data["worker_processes"]) if data.get("worker_processes") else None),
        )


@dataclass
class BackendSpec:
    """Where a backend's binaries and weights live, plus any fixed extra args/env.

    Kept separate from the sweep axes because these are properties of the *installation*,
    not of the experiment.

    `name` is the key under `backends:` and is what reports, config labels and
    `deployment.backend` refer to. `type` is which engine it is (llamacpp | vllm) and is what
    decides the launcher, the HTTP client and the offline tool. They coincide unless the spec
    names a variant -- two llama.cpp builds, or one engine with two models:

        backends:
          llamacpp-zendnn-q8: {type: llamacpp, server_bin: ..., model: ...Q8_0.gguf}
          vllm-w8a8:          {type: vllm, model: ...quantized.w8a8}

    A variant is a separate installation, not a sweep axis over one, because its binary and
    its weights differ: nothing about it can be varied by a flag on the same server.
    """

    name: str
    type: str = ""                 # llamacpp | vllm; defaults to `name`
    model: str = ""
    served_model_name: str = ""
    server_bin: str = ""
    offline_bin: str = ""          # llama-bench / `vllm bench latency`
    batched_bin: str = ""          # llama-batched-bench (llama.cpp only)
    extra_args: list[str] = field(default_factory=list)
    offline_extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    base_port: int = 0
    # Server axes this installation sweeps instead of the global `deployment:` values. For
    # settings that are not comparable across engines anyway -- llama.cpp's `-c 32000` next
    # to vLLM's `--max-model-len 8192` -- without crossing each value with every backend.
    deployment: dict[str, list[int]] = field(default_factory=dict)

    _DEFAULT_PORT = {"llamacpp": 8100, "vllm": 8200}
    OVERRIDABLE = ("n_ctx", "n_parallel", "batch", "ubatch", "threads_per_instance")

    def __post_init__(self) -> None:
        if not self.type:
            self.type = self.name

    @classmethod
    def from_dict(cls, name: str, data: dict | None) -> "BackendSpec":
        data = data or {}
        _unknown_keys(data, ("type", "model", "served_model_name", "server_bin", "offline_bin",
                             "batched_bin", "extra_args", "offline_extra_args", "env",
                             "base_port", "deployment"), f"backends.{name}")
        overrides_in = data.get("deployment") or {}
        if not isinstance(overrides_in, dict):
            raise SpecError(f"backends.{name}.deployment must be a mapping")
        _unknown_keys(overrides_in, cls.OVERRIDABLE, f"backends.{name}.deployment")
        overrides: dict[str, list[int]] = {}
        for key, value in overrides_in.items():
            values = _int_axis(value, f"backends.{name}.deployment.{key}")
            if any(v < 1 for v in values):
                raise SpecError(f"backends.{name}.deployment.{key}: every value must be >= 1")
            overrides[key] = values
        if not _BACKEND_NAME.fullmatch(name):
            raise SpecError(f"backends: {name!r} is not a usable name; use letters, digits, "
                            f"'-', '_' and '.' only")
        kind = str(data.get("type", name))
        if kind not in BACKENDS:
            hint = ("" if "type" in data else
                    f"; to name a variant, keep the key and add `type: llamacpp` or `type: vllm`")
            raise SpecError(f"backends.{name}: unknown backend type {kind!r} "
                            f"(known: {list(BACKENDS)}){hint}")
        env = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
        return cls(
            name=name,
            type=kind,
            # Expanded here because nothing downstream does: argv goes to the server without
            # a shell, so `~/models/x.gguf` would reach llama-server as a literal `~`.
            model=_expand(data.get("model", "")),
            served_model_name=str(data.get("served_model_name", "")),
            server_bin=_expand(data.get("server_bin", "")),
            offline_bin=_expand(data.get("offline_bin", "")),
            batched_bin=_expand(data.get("batched_bin", "")),
            extra_args=[str(a) for a in _as_list(data.get("extra_args"), "extra_args")],
            offline_extra_args=[str(a) for a in _as_list(data.get("offline_extra_args"), "offline_extra_args")],
            env=env,
            base_port=int(data.get("base_port", cls._DEFAULT_PORT.get(kind, 8100))),
            deployment=overrides,
        )

    def axis(self, name: str, default: list):
        """This installation's values for a server axis: its override, else the global."""
        return list(self.deployment.get(name) or default)

    def model_label(self) -> str:
        """Short name for reports: the filename stem, not the whole path."""
        return self.served_model_name or (Path(self.model).name if self.model else self.name)


@dataclass
class DeploymentAxes:
    """Server-side axes. Any change here forces a relaunch."""

    backend: list[str] = field(default_factory=lambda: ["llamacpp"])
    instances: list[int] = field(default_factory=lambda: [1])
    n_parallel: list[int] = field(default_factory=lambda: [1])
    n_ctx: list[int] = field(default_factory=lambda: [4096])
    batch: list[int] = field(default_factory=lambda: [2048])
    ubatch: list[int] = field(default_factory=lambda: [512])
    cores_per_instance: list[int] = field(default_factory=list)   # empty => split the budget
    threads_per_instance: list[int] = field(default_factory=list)  # empty => one per granted cpu

    @classmethod
    def from_dict(cls, data: dict | None) -> "DeploymentAxes":
        data = data or {}
        known = ("backend", "instances", "n_parallel", "n_ctx", "batch", "ubatch",
                 "cores_per_instance", "threads_per_instance")
        _unknown_keys(data, known, "deployment")
        axes = cls(
            # Names, not types: any key under `backends:`. SuiteSpec.validate checks each one
            # has a matching section.
            backend=_str_axis(data.get("backend"), "deployment.backend", default=["llamacpp"]),
            instances=_int_axis(data.get("instances"), "deployment.instances", default=[1]),
            n_parallel=_int_axis(data.get("n_parallel"), "deployment.n_parallel", default=[1]),
            n_ctx=_int_axis(data.get("n_ctx"), "deployment.n_ctx", default=[4096]),
            batch=_int_axis(data.get("batch"), "deployment.batch", default=[2048]),
            ubatch=_int_axis(data.get("ubatch"), "deployment.ubatch", default=[512]),
            cores_per_instance=_int_axis(data.get("cores_per_instance"),
                                         "deployment.cores_per_instance", default=[]),
            threads_per_instance=_int_axis(data.get("threads_per_instance"),
                                           "deployment.threads_per_instance", default=[]),
        )
        # Each of these is passed straight to a server flag (`-np`, `-c`, `-b`, `-ub`, `-t`).
        # A zero or negative value produces a command line the backend rejects at launch, an
        # hour into a sweep, with an error that points at the backend rather than the spec.
        # `n_parallel: 0` was additionally silent: it disabled the capacity pre-flight's
        # slot-count guard, so every trial ran unchecked.
        for field_name in ("instances", "n_parallel", "n_ctx", "batch", "ubatch",
                           "cores_per_instance", "threads_per_instance"):
            bad = [v for v in getattr(axes, field_name) if v < 1]
            if bad:
                raise SpecError(f"deployment.{field_name}: every value must be >= 1, got {bad}")
        return axes


@dataclass
class WorkloadAxes:
    """Client-side axes, run against a live deployment."""

    n_prompt: list[int] = field(default_factory=lambda: [512])
    n_gen: list[int] = field(default_factory=lambda: [128])
    pg: list[tuple[int, int]] = field(default_factory=list)
    n_depth: list[int] = field(default_factory=lambda: [0])
    concurrency: list[int] = field(default_factory=lambda: [1])
    shared_prefix: list[int] = field(default_factory=lambda: [0])
    request_rate: list[float] = field(default_factory=list)
    reps: int = 5
    warmup_fixed: int | None = None
    no_warmup: bool = False
    # Bypass the runner's capacity preflight. Set this only to measure queueing
    # behaviour on purpose -- concurrency above the fleet's slot count otherwise times
    # our own client-side queue rather than the engine.
    force: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorkloadAxes":
        data = data or {}
        known = ("n_prompt", "n_gen", "pg", "n_depth", "concurrency", "shared_prefix",
                 "request_rate", "reps", "warmup_fixed", "no_warmup", "force")
        _unknown_keys(data, known, "workload")
        pg: list[tuple[int, int]] = []
        for item in _as_list(data.get("pg"), "workload.pg"):
            if isinstance(item, str):
                pp_s, tg_s = item.split(",")
                pg.append((int(pp_s), int(tg_s)))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pg.append((int(item[0]), int(item[1])))
            else:
                raise SpecError(f"workload.pg: expected 'pp,tg' or [pp, tg], got {item!r}")
        return cls(
            n_prompt=_int_axis(data.get("n_prompt"), "workload.n_prompt", default=[512]),
            n_gen=_int_axis(data.get("n_gen"), "workload.n_gen", default=[128]),
            pg=pg,
            n_depth=_int_axis(data.get("n_depth"), "workload.n_depth", default=[0]),
            concurrency=_int_axis(data.get("concurrency"), "workload.concurrency", default=[1]),
            shared_prefix=_int_axis(data.get("shared_prefix"), "workload.shared_prefix", default=[0]),
            request_rate=[float(v) for v in _as_list(data.get("request_rate"), "workload.request_rate")],
            reps=int(data.get("reps", 5)),
            warmup_fixed=(int(data["warmup_fixed"]) if data.get("warmup_fixed") is not None else None),
            no_warmup=bool(data.get("no_warmup", False)),
            force=bool(data.get("force", False)),
        )


@dataclass
class OfflineAxes:
    """Axes for the native offline tools.

    `batch_size` is the static batch: `llama-batched-bench -npl B` and
    `vllm bench latency --batch-size B`. It is the one axis that maps cleanly onto both.
    """

    batch_size: list[int] = field(default_factory=lambda: [1])
    n_prompt: list[int] = field(default_factory=lambda: [512])
    n_gen: list[int] = field(default_factory=lambda: [128])
    reps: int = 3
    shared_prompt: bool = False       # llama-batched-bench -pps
    throughput_prompts: list[int] = field(default_factory=list)  # vllm bench throughput

    @classmethod
    def from_dict(cls, data: dict | None) -> "OfflineAxes":
        data = data or {}
        known = ("batch_size", "n_prompt", "n_gen", "reps", "shared_prompt", "throughput_prompts")
        _unknown_keys(data, known, "offline")
        return cls(
            batch_size=_int_axis(data.get("batch_size"), "offline.batch_size", default=[1]),
            n_prompt=_int_axis(data.get("n_prompt"), "offline.n_prompt", default=[512]),
            n_gen=_int_axis(data.get("n_gen"), "offline.n_gen", default=[128]),
            reps=int(data.get("reps", 3)),
            shared_prompt=bool(data.get("shared_prompt", False)),
            throughput_prompts=_int_axis(data.get("throughput_prompts"),
                                         "offline.throughput_prompts", default=[]),
        )


@dataclass
class Constraint:
    metric: str
    max: float | None = None
    min: float | None = None

    def satisfied_by(self, value: float | None) -> bool:
        """A missing metric fails the constraint -- an unmeasured SLO is not a met SLO."""
        if value is None:
            return False
        if self.max is not None and value > self.max:
            return False
        if self.min is not None and value < self.min:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.min is not None:
            parts.append(f"{self.metric} >= {self.min:g}")
        if self.max is not None:
            parts.append(f"{self.metric} <= {self.max:g}")
        return " and ".join(parts)

    @classmethod
    def from_dict(cls, data: dict) -> "Constraint":
        _unknown_keys(data, ("metric", "max", "min"), "constraints[]")
        if "metric" not in data:
            raise SpecError("constraints[]: missing required key 'metric'")
        if data.get("max") is None and data.get("min") is None:
            raise SpecError(f"constraints[{data['metric']}]: needs at least one of 'max'/'min'")
        return cls(
            metric=str(data["metric"]),
            max=(float(data["max"]) if data.get("max") is not None else None),
            min=(float(data["min"]) if data.get("min") is not None else None),
        )


@dataclass
class ObjectiveSpec:
    metric: str = "total_token_throughput"
    goal: str = "max"                 # max | min
    src: str = "client"               # which measurement path the objective reads
    constraints: list[Constraint] = field(default_factory=list)
    # A row that lost more than this share of its requests is not ranked. 0 means any lost
    # request disqualifies: a config that drops requests has not delivered the throughput
    # computed from the ones it kept.
    max_error_pct: float = 0.0
    # Rank rows flagged as contaminated (foreign CPU load, threads outside the allocation,
    # lost requests) instead of excluding them. For diagnosing a noisy box, not for answers.
    rank_flagged: bool = False

    @classmethod
    def from_dict(cls, data: dict | None, constraints: Any) -> "ObjectiveSpec":
        data = data or {}
        _unknown_keys(data, ("metric", "goal", "src", "max_error_pct", "rank_flagged"),
                      "objective")
        parsed = [Constraint.from_dict(c) for c in _as_list(constraints, "constraints")]
        max_error_pct = float(data.get("max_error_pct", 0.0))
        if not 0.0 <= max_error_pct <= 100.0:
            raise SpecError("objective.max_error_pct must be between 0 and 100")
        return cls(
            metric=str(data.get("metric", "total_token_throughput")),
            goal=_one_of(data.get("goal"), "objective.goal", ("max", "min"), "max"),
            src=_one_of(data.get("src"), "objective.src", ("client", "server"), "client"),
            constraints=parsed,
            max_error_pct=max_error_pct,
            rank_flagged=bool(data.get("rank_flagged", False)),
        )

    def describe(self) -> str:
        base = f"{self.goal}imise {self.metric} (src={self.src})"
        if self.constraints:
            base += " subject to " + ", ".join(c.describe() for c in self.constraints)
        return base


# --- top level ---


@dataclass
class SuiteSpec:
    name: str = "sweep"
    out_dir: str = "out/sweep"
    mode: str = "online"              # online | offline | both
    cpu: CpuSpec = field(default_factory=CpuSpec)
    lb: LbSpec = field(default_factory=LbSpec)
    backends: dict[str, BackendSpec] = field(default_factory=dict)
    deployment: DeploymentAxes = field(default_factory=DeploymentAxes)
    workload: WorkloadAxes = field(default_factory=WorkloadAxes)
    offline: OfflineAxes = field(default_factory=OfflineAxes)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)

    startup_timeout_s: float = 900.0
    request_timeout_s: float = 600.0
    settle_s: float = 3.0             # pause after teardown before the next launch
    continue_on_error: bool = True
    endpoint: str = "completions"
    dry_run: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteSpec":
        known = ("name", "out_dir", "mode", "cpu", "lb", "backends", "deployment", "workload",
                 "offline", "objective", "constraints", "startup_timeout_s",
                 "request_timeout_s", "settle_s", "continue_on_error", "endpoint", "dry_run")
        # Top-level `x-*` keys are ignored, as in docker-compose: they exist to hold YAML
        # anchors (`x-llamacpp-env: &env {...}`) shared by several backends. Only the prefix
        # is exempt, so a misspelt real key still fails.
        data = {k: v for k, v in data.items() if not str(k).startswith("x-")}
        _unknown_keys(data, known, "spec")

        backends = {
            name: BackendSpec.from_dict(name, cfg)
            for name, cfg in (data.get("backends") or {}).items()
        }
        spec = cls(
            name=str(data.get("name", "sweep")),
            out_dir=str(data.get("out_dir", "out/sweep")),
            mode=_one_of(data.get("mode"), "mode", ("online", "offline", "both"), "online"),
            cpu=CpuSpec.from_dict(data.get("cpu")),
            lb=LbSpec.from_dict(data.get("lb")),
            backends=backends,
            deployment=DeploymentAxes.from_dict(data.get("deployment")),
            workload=WorkloadAxes.from_dict(data.get("workload")),
            offline=OfflineAxes.from_dict(data.get("offline")),
            objective=ObjectiveSpec.from_dict(data.get("objective"), data.get("constraints")),
            startup_timeout_s=float(data.get("startup_timeout_s", 900.0)),
            request_timeout_s=float(data.get("request_timeout_s", 600.0)),
            settle_s=float(data.get("settle_s", 3.0)),
            continue_on_error=bool(data.get("continue_on_error", True)),
            endpoint=_one_of(data.get("endpoint"), "endpoint", ("completions", "chat"), "completions"),
            dry_run=bool(data.get("dry_run", False)),
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuiteSpec":
        import yaml

        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise SpecError(f"{path}: top level must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def validate(self) -> None:
        missing = [b for b in self.deployment.backend if b not in self.backends]
        if missing:
            raise SpecError(
                f"deployment.backend names {missing} but there is no matching `backends:` "
                f"section for them (found: {sorted(self.backends)})"
            )
        for name in self.deployment.backend:
            spec = self.backends[name]
            if not spec.model:
                raise SpecError(f"backends.{name}.model is required")
            if self.mode in ("online", "both") and not spec.server_bin:
                raise SpecError(f"backends.{name}.server_bin is required for mode={self.mode}")
        if self.workload.reps < 1:
            raise SpecError("workload.reps must be >= 1")
        if self.offline.reps < 1:
            raise SpecError("offline.reps must be >= 1")
        if any(c < 1 for c in self.workload.concurrency):
            raise SpecError("workload.concurrency: every value must be >= 1")
        if any(b < 1 for b in self.offline.batch_size):
            raise SpecError("offline.batch_size: every value must be >= 1")
        if self.lb.kind == "nginx":
            if not (0 < self.lb.port < 65536):
                raise SpecError("lb.port must be a valid port number for lb.kind=nginx")
            if self.lb.n_cpus < 1 and not self.lb.cpus:
                raise SpecError("lb.n_cpus must be >= 1 (nginx needs at least one cpu to run on)")
        # Instance ports are assigned as base_port + instance index, so a base port close to
        # the top of the range silently wraps out of it for a multi-instance fleet.
        max_instances = max(self.deployment.instances, default=1)
        for name in self.deployment.backend:
            top = self.backends[name].base_port + max_instances - 1
            if not (0 < self.backends[name].base_port and top < 65536):
                raise SpecError(
                    f"backends.{name}.base_port={self.backends[name].base_port} cannot host "
                    f"{max_instances} instance(s): ports are assigned base_port..{top}"
                )
            if self.lb.kind == "nginx" and self.backends[name].base_port <= self.lb.port <= top:
                raise SpecError(
                    f"lb.port={self.lb.port} collides with the instance port range for "
                    f"backends.{name} ({self.backends[name].base_port}..{top}); nginx and an "
                    f"instance cannot both bind it"
                )
        if self.mode in ("offline", "both"):
            for name in self.deployment.backend:
                spec = self.backends[name]
                if spec.type == "llamacpp" and not (spec.offline_bin or spec.batched_bin):
                    raise SpecError(
                        f"backends.{name} needs offline_bin (llama-bench) and/or batched_bin "
                        f"(llama-batched-bench) for offline mode"
                    )
                if spec.type == "vllm" and not spec.offline_bin:
                    raise SpecError(f"backends.{name}.offline_bin (the `vllm` binary) is required for offline mode")
        # A multi-instance deployment with no load balancer means the client has no defined
        # way to address the fleet; refuse rather than silently benchmarking instance 0 only.
        if self.lb.kind == "none" and max(self.deployment.instances, default=1) > 1:
            raise SpecError(
                "lb.kind=none cannot serve a multi-instance deployment "
                f"(deployment.instances includes {max(self.deployment.instances)}); "
                "use lb.kind=client or lb.kind=nginx"
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT_SPEC_PATH = "sweep.yaml"

__all__ = [
    "SuiteSpec", "CpuSpec", "LbSpec", "BackendSpec", "DeploymentAxes", "WorkloadAxes",
    "OfflineAxes", "ObjectiveSpec", "Constraint", "SpecError", "BACKENDS",
]
