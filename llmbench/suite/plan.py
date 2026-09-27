"""Expand a SuiteSpec into the concrete, ordered list of trials a run will execute.

The output of this module is fully resolved -- every port, every core list, every membind
node is decided here, before anything is launched. That means `llmbench sweep --dry-run` can
print exactly what would happen, and the run manifest can record exactly what did.

Ordering rule: deployment outermost, workload innermost. Relaunching an 8B model on CPU costs
far more than changing a prompt length, so the plan groups every workload that can share a
live deployment.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from .spec import BackendSpec, SuiteSpec
from .topology import (
    AllocationError, AllocationPlan, AuxPlacement, CoreSet, Topology, allocate, auxiliary_cpus,
)

# Prefix caches match on whole blocks. vLLM's default block_size is 16 tokens; llama.cpp
# reuses whole processed chunks. Used only to warn when a shared prefix is too short to
# produce any hit at all.
PREFIX_CACHE_BLOCK = 16


@dataclass
class InstancePlan:
    """One server process: where it listens and which cores it owns."""

    index: int
    port: int
    cores: CoreSet

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "port": self.port, "url": self.url, **self.cores.to_dict()}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class DeploymentPlan:
    """A fully-resolved fleet: N server processes plus the endpoint the client will hit."""

    id: str
    backend: str
    backend_spec: BackendSpec
    n_instances: int
    n_ctx: int
    n_parallel: int
    batch: int
    ubatch: int
    instances: list[InstancePlan]
    allocation: AllocationPlan
    lb_kind: str
    lb_placement: AuxPlacement | None = None
    lb_port: int = 0
    threads_per_instance: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [i.url for i in self.instances]

    @property
    def client_url(self) -> str:
        """Where the benchmark client sends requests.

        With lb.kind=nginx this is the proxy, which is inside the measured wire-to-wire
        window; with lb.kind=client the client addresses instances directly and there is no
        extra hop. The report records which, because it changes what the latency means.
        """
        if self.lb_kind == "nginx":
            return f"http://127.0.0.1:{self.lb_port}"
        return self.instances[0].url

    def axes(self) -> dict[str, Any]:
        """The knobs that define this deployment -- what the report groups and sorts by."""
        return {
            "backend": self.backend,
            "instances": self.n_instances,
            "cores_per_instance": self.instances[0].cores.n_physical_cores if self.instances else 0,
            "threads_per_instance": (
                self.threads_per_instance
                if self.threads_per_instance is not None
                else (self.instances[0].cores.n_threads if self.instances else 0)
            ),
            "n_ctx": self.n_ctx,
            "n_parallel": self.n_parallel,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "lb": self.lb_kind,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "axes": self.axes(),
            "backend_type": self.backend_spec.type,
            "model": self.backend_spec.model,
            "model_label": self.backend_spec.model_label(),
            "client_url": self.client_url,
            "instances": [i.to_dict() for i in self.instances],
            "allocation": self.allocation.to_dict(),
            "lb": {
                "kind": self.lb_kind,
                "port": self.lb_port,
                "placement": self.lb_placement.to_dict() if self.lb_placement else None,
            },
            "warnings": list(self.warnings),
        }


@dataclass
class WorkloadPlan:
    """One client-side measurement against a live deployment."""

    id: str
    n_prompt: int
    n_gen: int
    n_depth: int
    concurrency: int
    shared_prefix: int
    reps: int
    request_rate: float | None = None
    is_pg: bool = False

    def test_name(self) -> str:
        """Same naming as the single-endpoint path (config.Instance.test_name)."""
        if self.is_pg:
            base = f"pp{self.n_prompt}+tg{self.n_gen}"
        elif self.n_gen == 0:
            base = f"pp{self.n_prompt}"
        elif self.n_prompt == 0:
            base = f"tg{self.n_gen}"
        else:
            base = f"pp{self.n_prompt}+tg{self.n_gen}"
        if self.n_depth > 0:
            base += f" @ d{self.n_depth}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "test": self.test_name()}


@dataclass
class OfflinePlan:
    """One invocation of a backend's own offline benchmark tool."""

    id: str
    backend: str
    backend_spec: BackendSpec
    tool: str                 # llama-bench | llama-batched-bench | vllm-latency | vllm-throughput
    batch_size: int
    n_prompt: int
    n_gen: int
    reps: int
    cores: CoreSet
    shared_prompt: bool = False
    num_prompts: int | None = None

    def test_name(self) -> str:
        base = f"pp{self.n_prompt}+tg{self.n_gen}" if self.n_gen and self.n_prompt else (
            f"pp{self.n_prompt}" if self.n_prompt else f"tg{self.n_gen}"
        )
        return f"{base} @ b{self.batch_size}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "backend": self.backend, "tool": self.tool,
            "batch_size": self.batch_size, "n_prompt": self.n_prompt, "n_gen": self.n_gen,
            "reps": self.reps, "shared_prompt": self.shared_prompt,
            "num_prompts": self.num_prompts, "test": self.test_name(),
            "model": self.backend_spec.model, "cores": self.cores.to_dict(),
        }


@dataclass
class SweepPlan:
    """Everything a run will do, resolved. `groups` preserves execution order."""

    spec: SuiteSpec
    topology: Topology
    groups: list[tuple[DeploymentPlan, list[WorkloadPlan]]] = field(default_factory=list)
    offline: list[OfflinePlan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_deployments(self) -> int:
        return len(self.groups)

    @property
    def n_online_trials(self) -> int:
        return sum(len(w) for _, w in self.groups)

    @property
    def n_offline_trials(self) -> int:
        return len(self.offline)

    @property
    def n_trials(self) -> int:
        return self.n_online_trials + self.n_offline_trials

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "mode": self.spec.mode,
            "objective": self.spec.objective.describe(),
            "n_deployments": self.n_deployments,
            "n_online_trials": self.n_online_trials,
            "n_offline_trials": self.n_offline_trials,
            "topology": self.topology.to_dict(),
            "deployments": [
                {"deployment": d.to_dict(), "workloads": [w.to_dict() for w in ws]}
                for d, ws in self.groups
            ],
            "offline": [o.to_dict() for o in self.offline],
            "warnings": list(self.warnings),
        }

    def fingerprint(self) -> str:
        """Identity of what this plan *measures*, so `sweep run --resume` can refuse to splice
        rows from two different experiments into one table.

        Covers every trial, port and core list, the host topology, and every spec field that
        changes what a server is launched with or what a client sends. Deliberately leaves
        out what can legitimately change between attempts without changing a measurement:
        timeouts, `settle_s`, `continue_on_error`, the objective (it only affects ranking),
        the sweep name and `out_dir`. Raising `startup_timeout_s` after a slow model failed
        to come up is the commonest reason to resume at all.
        """
        plan = self.to_dict()
        for volatile in ("name", "objective", "warnings"):
            plan.pop(volatile, None)
        s = self.spec
        body = {
            "plan": plan,
            "backends": {k: dataclasses.asdict(v) for k, v in sorted(s.backends.items())},
            "workload": dataclasses.asdict(s.workload),
            "offline": dataclasses.asdict(s.offline),
            "cpu": dataclasses.asdict(s.cpu),
            "lb": dataclasses.asdict(s.lb),
            "endpoint": s.endpoint,
            "mode": s.mode,
        }
        blob = json.dumps(body, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _cartesian(axes: list[list]) -> Iterator[tuple]:
    if not axes:
        yield ()
        return
    head, *rest = axes
    for h in head:
        for t in _cartesian(rest):
            yield (h,) + t


def _expand_workloads(spec: SuiteSpec) -> list[WorkloadPlan]:
    """Mirrors get_cmd_params_instances(): pp / tg / pg are parallel inner loops, not nested."""
    w = spec.workload
    rates: list[float | None] = list(w.request_rate) if w.request_rate else [None]
    out: list[WorkloadPlan] = []
    idx = 0

    for depth, conc, prefix, rate in _cartesian([w.n_depth, w.concurrency, w.shared_prefix, rates]):
        common = dict(n_depth=depth, concurrency=conc, shared_prefix=prefix,
                      reps=w.reps, request_rate=rate)
        for n_prompt in w.n_prompt:
            if n_prompt == 0:
                continue
            out.append(WorkloadPlan(id=f"w{idx:04d}", n_prompt=n_prompt, n_gen=0, **common))
            idx += 1
        for n_gen in w.n_gen:
            if n_gen == 0:
                continue
            out.append(WorkloadPlan(id=f"w{idx:04d}", n_prompt=0, n_gen=n_gen, **common))
            idx += 1
        for pp, tg in w.pg:
            if pp == 0 and tg == 0:
                continue
            out.append(WorkloadPlan(id=f"w{idx:04d}", n_prompt=pp, n_gen=tg, is_pg=True, **common))
            idx += 1
    return out


def _deployment_points(spec: SuiteSpec, cores_axis: list[int | None]) -> Iterator[tuple]:
    """Every (backend, instances, cores, n_ctx, n_parallel, batch, ubatch, threads) point.

    Backend outermost, as before; within each backend, its own `backends.<name>.deployment`
    overrides replace the global axis values rather than being crossed with them.
    """
    d = spec.deployment
    for backend in d.backend:
        b = spec.backends[backend]
        threads = b.axis("threads_per_instance", d.threads_per_instance) or [None]
        for rest in _cartesian([
            d.instances, cores_axis, b.axis("n_ctx", d.n_ctx), b.axis("n_parallel", d.n_parallel),
            b.axis("batch", d.batch), b.axis("ubatch", d.ubatch), threads,
        ]):
            yield (backend, *rest)


def _port_base(spec: SuiteSpec, backend: str) -> int:
    return spec.backends[backend].base_port


def build_plan(spec: SuiteSpec, topo: Topology | None = None) -> SweepPlan:
    """Resolve a spec into an ordered, fully-decided plan. Raises SpecError/AllocationError
    for anything unsatisfiable -- better to fail here than three hours into a sweep."""
    topo = topo or Topology.detect()
    plan = SweepPlan(spec=spec, topology=topo)
    d = spec.deployment

    cores_axis: list[int | None] = list(d.cores_per_instance) or [None]

    workloads = _expand_workloads(spec) if spec.mode in ("online", "both") else []

    if spec.mode in ("online", "both"):
        dep_idx = 0
        for backend, n_inst, cpi, n_ctx, n_par, batch, ubatch, threads in _deployment_points(
            spec, cores_axis,
        ):
            try:
                alloc = allocate(
                    topo, spec.cpu.budget, n_instances=n_inst, smt=spec.cpu.smt,
                    ccd_align=spec.cpu.ccd_align, reserve=spec.cpu.reserve,
                    membind=spec.cpu.membind, cores_per_instance=cpi,
                )
            except AllocationError as e:
                # An unsatisfiable core split is a property of this one point, not of the
                # whole sweep -- drop the point and keep going, but say so loudly.
                plan.warnings.append(
                    f"skipped deployment (backend={backend}, instances={n_inst}, "
                    f"cores_per_instance={cpi}): {e}"
                )
                continue

            base_port = _port_base(spec, backend)
            instances = [
                InstancePlan(index=cs.index, port=base_port + cs.index, cores=cs)
                for cs in alloc.instances
            ]

            lb_kind = spec.lb.kind
            # `uniform: false` means the proxy only appears when it is actually needed. That
            # is cheaper, but it makes 1-instance and N-instance latencies incomparable,
            # because only one of them pays the proxy hop.
            if lb_kind == "nginx" and n_inst == 1 and not spec.lb.uniform:
                lb_kind = "client"
            if lb_kind == "client" and n_inst == 1:
                lb_kind = "none"

            lb_placement = None
            if lb_kind == "nginx":
                lb_placement = auxiliary_cpus(
                    topo, alloc, want=spec.lb.n_cpus, explicit=spec.lb.cpus,
                )

            dep = DeploymentPlan(
                id=f"d{dep_idx:03d}", backend=backend, backend_spec=spec.backends[backend],
                n_instances=n_inst, n_ctx=n_ctx, n_parallel=n_par, batch=batch, ubatch=ubatch,
                instances=instances, allocation=alloc, lb_kind=lb_kind,
                lb_placement=lb_placement, lb_port=spec.lb.port,
                threads_per_instance=threads, warnings=list(alloc.warnings),
            )
            if lb_placement and lb_placement.note:
                dep.warnings.append(f"lb placement: {lb_placement.note}")
            # `threads_per_instance` is passed straight through to `-t`. Asking for more
            # threads than the instance owns cpus does not buy any: the extra ones are pinned
            # into the same cpuset by numactl, so they only add context switches to a
            # barrier-synchronised thread pool. Asking for fewer deliberately idles cores.
            granted = instances[0].cores.n_threads if instances else 0
            if threads and granted:
                if threads > granted:
                    dep.warnings.append(
                        f"threads_per_instance={threads} exceeds the {granted} cpu(s) each "
                        f"instance is pinned to; the surplus threads oversubscribe the same "
                        f"cpuset and ggml's per-compute barrier makes that strictly slower"
                    )
                elif threads < granted:
                    dep.warnings.append(
                        f"threads_per_instance={threads} is below the {granted} cpu(s) "
                        f"allocated to each instance; {granted - threads} cpu(s) per instance "
                        f"stay idle and the row understates what the allocation can do"
                    )
            if lb_kind == "nginx":
                dep.warnings.append(
                    "nginx is inside the measured wire-to-wire window; reported latency "
                    "includes the proxy hop"
                )
            # A fleet cannot be saturated by fewer in-flight requests than it has instances:
            # with concurrency 1 against 4 instances, 3 sit idle for the whole measurement and
            # the row describes one instance on a quarter of the cores. Load balancing cannot
            # fix this -- there is simply no second request to place.
            starved = sorted({w.concurrency for w in workloads if w.concurrency < n_inst})
            if starved and n_inst > 1:
                dep.warnings.append(
                    f"concurrency {starved} is below the instance count ({n_inst}); at least "
                    f"{n_inst - max(starved)} instance(s) stay idle for those workloads, so "
                    f"they measure a partial fleet rather than the full deployment"
                )

            # The runner's capacity preflight refuses concurrency above the fleet's total slot
            # count, because queueing behind a full server measures our own queue rather than
            # the engine. Catching it here turns "every trial fails after the servers finally
            # came up" into an error before anything is launched.
            # A closed-loop measurement needs enough requests that steady state dominates the
            # ramp-up and drain at either end. With reps only slightly above concurrency, the
            # first `concurrency` requests all start against an empty server and the last few
            # finish against a draining one, and that transient IS the measurement. Observed
            # here: reps=6 at concurrency=4 gave 26-46% relative stddev on the same workload
            # that reps=40 measured at 3-8%.
            thin = sorted({w.concurrency for w in workloads
                           if w.concurrency > 1 and w.reps < w.concurrency * 4})
            if thin:
                dep.warnings.append(
                    f"reps={spec.workload.reps} is less than 4x concurrency {thin}; the "
                    f"measurement will be dominated by ramp-up and drain rather than steady "
                    f"state, and run-to-run spread will be large. Use reps >= 10x concurrency "
                    f"for a stable closed-loop number."
                )

            # Prefix caches match at block granularity, not per token: vLLM's default block
            # is 16 tokens and llama.cpp matches whole processed chunks. A shared prefix
            # shorter than one block can never produce a hit, so the experiment silently
            # measures nothing and looks like "prefix caching does not help".
            short = sorted({w.shared_prefix for w in workloads
                            if 0 < w.shared_prefix < PREFIX_CACHE_BLOCK})
            if short:
                dep.warnings.append(
                    f"shared_prefix {short} is below the {PREFIX_CACHE_BLOCK}-token block a "
                    f"prefix cache matches on, so no block can be reused and these rows will "
                    f"show zero cached tokens whatever the backend does. Use a shared_prefix "
                    f"of at least {PREFIX_CACHE_BLOCK}, and check `cached_tokens_mean` to "
                    f"confirm the hit rather than assuming it"
                )
            # A prompt no longer than its own shared prefix leaves nothing unique per request,
            # so every request is a full cache hit and the row measures retrieval, not prefill.
            degenerate = sorted({(w.shared_prefix, w.n_prompt) for w in workloads
                                 if w.shared_prefix and w.n_prompt
                                 and w.shared_prefix >= w.n_prompt})
            if degenerate:
                dep.warnings.append(
                    f"shared_prefix >= n_prompt for {degenerate}; the entire prompt is shared, "
                    f"so after the first request there is no prefill left to measure"
                )

            fleet_slots = n_par * n_inst if n_par > 0 else None
            over = sorted({w.concurrency for w in workloads
                           if fleet_slots and w.concurrency > fleet_slots})
            if over and not spec.workload.force:
                plan.warnings.append(
                    f"{dep.id}: concurrency {over} exceeds this deployment's total slot count "
                    f"({n_par} slots x {n_inst} instance(s) = {fleet_slots}). Those trials "
                    f"will be recorded as 'capacity' without measuring anything. Raise "
                    f"`deployment.n_parallel`, lower `workload.concurrency`, or set "
                    f"`workload.force: true` to measure the queueing behaviour deliberately."
                )

            plan.groups.append((dep, list(workloads)))
            dep_idx += 1

    if spec.mode in ("offline", "both"):
        plan.offline = _build_offline(spec, topo, plan)

    return plan


def _build_offline(spec: SuiteSpec, topo: Topology, plan: SweepPlan) -> list[OfflinePlan]:
    """Offline runs are single-process by nature, so they take the whole budget as one core
    set. Instance-count sweeps are meaningless here -- the native tools have no fleet."""
    o = spec.offline
    try:
        alloc = allocate(
            topo, spec.cpu.budget, n_instances=1, smt=spec.cpu.smt,
            ccd_align=spec.cpu.ccd_align, reserve=spec.cpu.reserve, membind=spec.cpu.membind,
        )
    except AllocationError as e:
        plan.warnings.append(f"offline trials skipped: {e}")
        return []
    cores = alloc.instances[0]

    out: list[OfflinePlan] = []
    idx = 0
    for backend in spec.deployment.backend:
        bspec = spec.backends[backend]
        for batch_size, n_prompt, n_gen in _cartesian([o.batch_size, o.n_prompt, o.n_gen]):
            tool = _offline_tool(bspec.type, bspec, batch_size)
            if tool is None:
                plan.warnings.append(
                    f"offline: no tool available for backend={backend} at batch_size="
                    f"{batch_size}; configure backends.{backend}.batched_bin/offline_bin"
                )
                continue
            out.append(OfflinePlan(
                id=f"o{idx:04d}", backend=backend, backend_spec=bspec, tool=tool,
                batch_size=batch_size, n_prompt=n_prompt, n_gen=n_gen, reps=o.reps,
                cores=cores, shared_prompt=o.shared_prompt,
            ))
            idx += 1
        for num_prompts in o.throughput_prompts:
            if bspec.type != "vllm":
                plan.warnings.append(
                    f"offline.throughput_prompts only applies to vllm "
                    f"(`vllm bench throughput`); ignored for backend={backend}"
                )
                break
            for n_prompt, n_gen in _cartesian([o.n_prompt, o.n_gen]):
                out.append(OfflinePlan(
                    id=f"o{idx:04d}", backend=backend, backend_spec=bspec,
                    tool="vllm-throughput", batch_size=0, n_prompt=n_prompt, n_gen=n_gen,
                    reps=o.reps, cores=cores, num_prompts=num_prompts,
                ))
                idx += 1
    return out


def _offline_tool(backend_type: str, bspec: BackendSpec, batch_size: int) -> str | None:
    """Pick the native tool that can express this batch size.

    llama-bench has no request-batch concept at all -- it drives a single sequence -- so any
    batch_size > 1 must go to llama-batched-bench (`-npl`). At batch_size == 1 either works;
    llama-bench is preferred because it is the canonical, most-cited llama.cpp number.
    """
    if backend_type == "llamacpp":
        if batch_size > 1:
            return "llama-batched-bench" if bspec.batched_bin else None
        if bspec.offline_bin:
            return "llama-bench"
        return "llama-batched-bench" if bspec.batched_bin else None
    if backend_type == "vllm":
        return "vllm-latency" if bspec.offline_bin else None
    return None


__all__ = [
    "SweepPlan", "DeploymentPlan", "WorkloadPlan", "OfflinePlan", "InstancePlan", "build_plan",
]
