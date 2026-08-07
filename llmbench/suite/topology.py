"""CPU / NUMA / CCD discovery and core-budget allocation.

Everything the suite does to place a server -- thread counts, `numactl --physcpubind`,
`--membind`, vLLM's `VLLM_CPU_OMP_THREADS_BIND`, llama.cpp's `-C/--cpu-mask` -- is derived
from one `Topology` snapshot taken from sysfs, so a run's placement is reproducible and
auditable after the fact (it is serialised into the run manifest).

Why CCD alignment matters here: on this class of AMD EPYC part, L3 is private per CCD (8
physical cores sharing 32 MiB on a 9R14). An instance whose cores straddle a CCD boundary
gets two partial L3 slices instead of one whole one, and the weight-streaming working set of
a decode step is exactly the kind of thing that notices. Aligning instance boundaries to CCD
boundaries is therefore the default, and a misaligned split is reported as a warning rather
than silently accepted.

Nothing here shells out to `numactl` to *discover* anything -- sysfs is the source of truth
and is always present. `numactl` is only used later, to *apply* a placement.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

SYS_CPU = Path("/sys/devices/system/cpu")
SYS_NODE = Path("/sys/devices/system/node")

SmtPolicy = Literal["exclude", "include", "only-siblings"]


class TopologyError(RuntimeError):
    pass


class AllocationError(ValueError):
    pass


# --- cpu-list parsing/formatting (the "0-7,192-199" format used by sysfs and numactl) ---


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux cpulist ("0-7,192-199", "96-191", "3") into sorted unique ids."""
    out: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"descending cpu range: {token!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return sorted(out)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Inverse of parse_cpu_list: collapse to the compact range form numactl expects."""
    ids = sorted(set(cpus))
    if not ids:
        return ""
    parts: list[str] = []
    start = prev = ids[0]
    for cpu in ids[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def cpu_mask_hex(cpus: Iterable[int]) -> str:
    """llama.cpp's `-C/--cpu-mask` takes an arbitrarily long hex bitmask (common/arg.cpp).

    Emitted without a `0x` prefix and without separators, which is what its parser accepts.
    """
    mask = 0
    for cpu in cpus:
        mask |= 1 << cpu
    return format(mask, "x")


# --- discovery ---


@dataclass(frozen=True)
class Cpu:
    id: int
    core_id: int          # physical core, unique only within a socket
    socket: int           # physical_package_id
    node: int             # NUMA node
    ccd: int              # index of the L3 domain this cpu belongs to
    siblings: tuple[int, ...]   # every logical cpu sharing this physical core, incl. self

    @property
    def is_primary(self) -> bool:
        """SMT siblings share a physical core; the lowest id is the 'primary' thread.

        Excluding SMT means keeping only primaries -- one logical cpu per physical core.
        """
        return self.id == min(self.siblings)


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _detect_nodes() -> dict[int, list[int]]:
    nodes: dict[int, list[int]] = {}
    for node_dir in sorted(SYS_NODE.glob("node[0-9]*")):
        try:
            node_id = int(node_dir.name.removeprefix("node"))
            nodes[node_id] = parse_cpu_list((node_dir / "cpulist").read_text().strip())
        except (OSError, ValueError):
            continue
    return nodes


def _node_distances() -> dict[int, dict[int, int]]:
    """node<->node relative distances (10 == local). Used to flag cross-socket placements."""
    dist: dict[int, dict[int, int]] = {}
    for node_dir in sorted(SYS_NODE.glob("node[0-9]*")):
        try:
            node_id = int(node_dir.name.removeprefix("node"))
            values = [int(v) for v in (node_dir / "distance").read_text().split()]
        except (OSError, ValueError):
            continue
        dist[node_id] = {other: value for other, value in enumerate(values)}
    return dist


@dataclass
class Topology:
    cpus: dict[int, Cpu]
    nodes: dict[int, list[int]]
    distances: dict[int, dict[int, int]]
    ccd_members: dict[int, list[int]]        # ccd index -> all logical cpus in it
    model_name: str = ""

    # -- construction --

    @classmethod
    def detect(cls) -> "Topology":
        nodes = _detect_nodes()
        if not nodes:
            raise TopologyError(f"no NUMA nodes found under {SYS_NODE}")
        cpu_to_node = {cpu: node for node, cpu_list in nodes.items() for cpu in cpu_list}

        # An L3 domain is a CCD on EPYC. Group by the shared_cpu_list string, then index the
        # groups in ascending order of their lowest cpu so ccd ids are stable across boots.
        l3_groups: dict[str, list[int]] = {}
        cpu_to_l3key: dict[int, str] = {}
        for cpu_dir in SYS_CPU.glob("cpu[0-9]*"):
            try:
                cpu_id = int(cpu_dir.name.removeprefix("cpu"))
            except ValueError:
                continue
            shared = cpu_dir / "cache" / "index3" / "shared_cpu_list"
            # Fall back to the NUMA node when there is no L3 level exposed (VMs, non-x86).
            key = shared.read_text().strip() if shared.exists() else f"node{cpu_to_node.get(cpu_id, 0)}"
            l3_groups.setdefault(key, parse_cpu_list(key) if shared.exists() else [])
            cpu_to_l3key[cpu_id] = key
        for key, members in l3_groups.items():
            if not members:
                l3_groups[key] = [c for c, k in cpu_to_l3key.items() if k == key]
        ordered = sorted(l3_groups.items(), key=lambda kv: min(kv[1]))
        l3key_to_ccd = {key: idx for idx, (key, _) in enumerate(ordered)}
        ccd_members = {idx: sorted(members) for idx, (_, members) in enumerate(ordered)}

        cpus: dict[int, Cpu] = {}
        for cpu_id, l3key in cpu_to_l3key.items():
            topo_dir = SYS_CPU / f"cpu{cpu_id}" / "topology"
            try:
                socket = _read_int(topo_dir / "physical_package_id")
                core_id = _read_int(topo_dir / "core_id")
                siblings = tuple(parse_cpu_list((topo_dir / "thread_siblings_list").read_text().strip()))
            except (OSError, ValueError):
                socket, core_id, siblings = 0, cpu_id, (cpu_id,)
            cpus[cpu_id] = Cpu(
                id=cpu_id, core_id=core_id, socket=socket,
                node=cpu_to_node.get(cpu_id, 0), ccd=l3key_to_ccd[l3key], siblings=siblings,
            )

        return cls(cpus=cpus, nodes=nodes, distances=_node_distances(),
                   ccd_members=ccd_members, model_name=_cpu_model_name())

    # -- queries --

    @property
    def n_logical(self) -> int:
        return len(self.cpus)

    @property
    def n_physical(self) -> int:
        return sum(1 for c in self.cpus.values() if c.is_primary)

    @property
    def smt_enabled(self) -> bool:
        return self.n_logical > self.n_physical

    @property
    def cores_per_ccd(self) -> int:
        """Physical cores per CCD, the natural quantum for instance sizing."""
        counts = {
            sum(1 for cpu in members if self.cpus[cpu].is_primary)
            for members in self.ccd_members.values()
        }
        return min(counts) if counts else 0

    def ccds_of(self, cpus: Iterable[int]) -> list[int]:
        return sorted({self.cpus[c].ccd for c in cpus if c in self.cpus})

    def nodes_of(self, cpus: Iterable[int]) -> list[int]:
        return sorted({self.cpus[c].node for c in cpus if c in self.cpus})

    def sockets_of(self, cpus: Iterable[int]) -> list[int]:
        return sorted({self.cpus[c].socket for c in cpus if c in self.cpus})

    def primaries(self, cpus: Iterable[int]) -> list[int]:
        return sorted(c for c in cpus if c in self.cpus and self.cpus[c].is_primary)

    def siblings_of(self, cpus: Iterable[int]) -> list[int]:
        """Every logical cpu sharing a physical core with any of `cpus` (includes `cpus`)."""
        out: set[int] = set()
        for c in cpus:
            if c in self.cpus:
                out.update(self.cpus[c].siblings)
        return sorted(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialised into the run manifest -- placement must be auditable after the fact."""
        return {
            "model_name": self.model_name,
            "n_logical": self.n_logical,
            "n_physical": self.n_physical,
            "smt_enabled": self.smt_enabled,
            "sockets": sorted({c.socket for c in self.cpus.values()}),
            "cores_per_ccd": self.cores_per_ccd,
            "n_ccds": len(self.ccd_members),
            "nodes": {str(n): format_cpu_list(cpus) for n, cpus in sorted(self.nodes.items())},
            "node_distances": {str(a): b for a, b in sorted(self.distances.items())},
            "ccds": {str(i): format_cpu_list(m) for i, m in sorted(self.ccd_members.items())},
        }


def _cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


# --- allocation ---


@dataclass
class CoreSet:
    """One instance's CPU placement, and every form the launchers need it in."""

    index: int
    cpus: list[int]                 # every logical cpu handed to this instance
    physical_cpus: list[int]        # the primary (non-SMT-sibling) subset
    nodes: list[int]
    sockets: list[int]
    ccds: list[int]
    membind: list[int]              # NUMA nodes to bind memory to

    @property
    def n_threads(self) -> int:
        """Thread count to request from the backend: one per logical cpu we granted."""
        return len(self.cpus)

    @property
    def n_physical_cores(self) -> int:
        return len(self.physical_cpus)

    @property
    def physcpubind(self) -> str:
        return format_cpu_list(self.cpus)

    @property
    def membind_arg(self) -> str:
        return ",".join(str(n) for n in self.membind)

    @property
    def cross_socket(self) -> bool:
        return len(self.sockets) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cpus": self.physcpubind,
            "n_logical": len(self.cpus),
            "n_physical_cores": self.n_physical_cores,
            "n_threads": self.n_threads,
            "nodes": self.nodes,
            "sockets": self.sockets,
            "ccds": self.ccds,
            "membind": self.membind,
            "cross_socket": self.cross_socket,
        }


@dataclass
class AllocationPlan:
    instances: list[CoreSet]
    reserved: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    budget: list[int] = field(default_factory=list)
    smt: str = "exclude"
    ccd_aligned: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": format_cpu_list(self.budget),
            "n_budget_logical": len(self.budget),
            "smt": self.smt,
            "ccd_aligned": self.ccd_aligned,
            "reserved": format_cpu_list(self.reserved),
            "n_reserved": len(self.reserved),
            "instances": [cs.to_dict() for cs in self.instances],
            "warnings": list(self.warnings),
        }


def allocate(
    topo: Topology,
    budget_spec: str,
    *,
    n_instances: int = 1,
    smt: SmtPolicy = "exclude",
    ccd_align: bool = True,
    reserve: int = 0,
    membind: str = "auto",
    cores_per_instance: int | None = None,
) -> AllocationPlan:
    """Partition `budget_spec` into `n_instances` disjoint, contiguous CoreSets.

    `reserve` holds back that many *physical* cores from the top of the budget before
    splitting -- these are for anything that shares the machine with the servers but must not
    steal their cores (nginx workers, the benchmark client itself). Reserved cores are
    reported, never silently absorbed.

    `smt`:
      exclude        one logical cpu per physical core (default; the usual choice for
                     throughput benchmarking, and what `--physcpubind` of primaries gives)
      include        both SMT siblings of every granted physical core
      only-siblings  the non-primary threads only -- for deliberately measuring SMT contention

    Splitting is by *physical core*, then siblings are attached, so an instance never gets one
    half of a physical core while its sibling goes to a different instance -- that would make
    two "isolated" instances silently share execution resources.
    """
    if n_instances < 1:
        raise AllocationError(f"n_instances must be >= 1, got {n_instances}")

    budget = [c for c in parse_cpu_list(budget_spec) if c in topo.cpus]
    unknown = set(parse_cpu_list(budget_spec)) - set(topo.cpus)
    warnings: list[str] = []
    if unknown:
        warnings.append(
            f"cpu budget referenced {len(unknown)} cpu(s) not present on this host "
            f"({format_cpu_list(unknown)}); ignored"
        )
    if not budget:
        raise AllocationError(f"cpu budget {budget_spec!r} resolved to no usable cpus")

    physical = topo.primaries(budget)
    if not physical:
        raise AllocationError(
            f"cpu budget {budget_spec!r} contains no primary threads; it appears to be "
            f"SMT siblings only"
        )

    if reserve < 0:
        raise AllocationError(f"reserve must be >= 0, got {reserve}")
    if reserve >= len(physical):
        raise AllocationError(
            f"reserve={reserve} would consume the entire budget of {len(physical)} physical cores"
        )
    if reserve and ccd_align:
        # A partial-CCD reserve poisons alignment for every subsequent split: 96 cores is 12
        # whole CCDs and divides evenly 1/2/3/4/6/12 ways, but 92 divides evenly none of
        # those. Round the reserve up to whole CCDs so what is left is still CCD-quantised.
        quantum = topo.cores_per_ccd or 1
        rounded = -(-reserve // quantum) * quantum
        if rounded != reserve:
            warnings.append(
                f"reserve={reserve} rounded up to {rounded} (one whole CCD is {quantum} cores) "
                f"to keep the remaining budget CCD-aligned; pass ccd_align=false to reserve "
                f"an exact core count instead"
            )
        if rounded >= len(physical):
            raise AllocationError(
                f"reserve={reserve} rounds up to {rounded} whole-CCD cores, which would consume "
                f"the entire budget of {len(physical)} physical cores"
            )
        reserve = rounded
    # Reserve from the top so instance 0 keeps the low, CCD-aligned end of the budget.
    reserved_phys = physical[len(physical) - reserve:] if reserve else []
    usable_phys = physical[: len(physical) - reserve] if reserve else list(physical)

    if cores_per_instance is not None:
        if cores_per_instance < 1:
            raise AllocationError(f"cores_per_instance must be >= 1, got {cores_per_instance}")
        needed = cores_per_instance * n_instances
        if needed > len(usable_phys):
            raise AllocationError(
                f"{n_instances} instance(s) x {cores_per_instance} core(s) = {needed} physical "
                f"cores, but only {len(usable_phys)} are available in budget {budget_spec!r}"
                + (f" after reserving {reserve}" if reserve else "")
            )
        if needed < len(usable_phys):
            warnings.append(
                f"cores_per_instance={cores_per_instance} x {n_instances} leaves "
                f"{len(usable_phys) - needed} physical core(s) of the budget unused"
            )
            usable_phys = usable_phys[:needed]
    elif len(usable_phys) < n_instances:
        raise AllocationError(
            f"cannot split {len(usable_phys)} physical core(s) across {n_instances} instances"
        )

    chunks, align_ok, align_warnings = _split_physical(
        topo, usable_phys, n_instances, ccd_align=ccd_align,
    )
    warnings.extend(align_warnings)

    instances: list[CoreSet] = []
    for idx, chunk in enumerate(chunks):
        cpus = _apply_smt(topo, chunk, smt)
        if not cpus:
            raise AllocationError(
                f"smt policy {smt!r} left instance {idx} with no cpus "
                f"(physical cores: {format_cpu_list(chunk)})"
            )
        nodes = topo.nodes_of(cpus)
        core_set = CoreSet(
            index=idx, cpus=cpus, physical_cpus=sorted(chunk), nodes=nodes,
            sockets=topo.sockets_of(cpus), ccds=topo.ccds_of(cpus),
            membind=_resolve_membind(membind, nodes),
        )
        if core_set.cross_socket:
            warnings.append(
                f"instance {idx} spans sockets {core_set.sockets}; cross-socket memory access "
                f"on this host costs ~{_worst_distance(topo, nodes)} vs 10 local"
            )
        instances.append(core_set)

    return AllocationPlan(
        instances=instances, reserved=_apply_smt(topo, reserved_phys, smt) if reserved_phys else [],
        warnings=warnings, budget=budget, smt=smt, ccd_aligned=align_ok,
    )


def _worst_distance(topo: Topology, nodes: list[int]) -> int:
    worst = 10
    for a in nodes:
        for b in nodes:
            worst = max(worst, topo.distances.get(a, {}).get(b, 10))
    return worst


def _apply_smt(topo: Topology, physical_cpus: Iterable[int], smt: SmtPolicy) -> list[int]:
    physical_cpus = list(physical_cpus)
    if smt == "exclude":
        return sorted(physical_cpus)
    if smt == "include":
        return topo.siblings_of(physical_cpus)
    if smt == "only-siblings":
        primaries = set(physical_cpus)
        return sorted(c for c in topo.siblings_of(physical_cpus) if c not in primaries)
    raise AllocationError(f"unknown smt policy: {smt!r}")


def _resolve_membind(membind: str, nodes: list[int]) -> list[int]:
    """`auto` binds memory to exactly the node(s) the instance's cores live on.

    That is the whole point of the exercise on a 2-socket box: cores on node 1 reading
    weights out of node 0 pay the remote-distance penalty on every layer.
    """
    if membind in ("auto", "", None):
        return list(nodes)
    if membind in ("none", "off"):
        return []
    if membind == "interleave":
        return list(nodes)  # caller switches numactl to --interleave; see deploy.py
    return [int(n) for n in str(membind).replace(",", " ").split()]


def _split_physical(
    topo: Topology, physical: list[int], n: int, *, ccd_align: bool,
) -> tuple[list[list[int]], bool, list[str]]:
    """Split physical cores into n contiguous chunks, preferring whole-CCD boundaries."""
    warnings: list[str] = []

    if ccd_align:
        by_ccd: dict[int, list[int]] = {}
        for cpu in physical:
            by_ccd.setdefault(topo.cpus[cpu].ccd, []).append(cpu)
        ccd_ids = sorted(by_ccd)
        sizes = {len(v) for v in by_ccd.values()}

        if len(ccd_ids) >= n and len(ccd_ids) % n == 0 and len(sizes) == 1:
            per = len(ccd_ids) // n
            chunks = [
                sorted(c for ccd in ccd_ids[i * per : (i + 1) * per] for c in by_ccd[ccd])
                for i in range(n)
            ]
            return chunks, True, warnings

        if len(ccd_ids) < n:
            # More instances than CCDs: instances must share L3. Legitimate (it is how you
            # measure L3 contention) but it is not isolation, so say so.
            warnings.append(
                f"ccd_align requested but {n} instances exceed the {len(ccd_ids)} CCD(s) in "
                f"budget; instances will share L3 slices and are not cache-isolated"
            )
        elif len(sizes) != 1:
            warnings.append(
                f"ccd_align requested but the budget covers partial CCDs (sizes {sorted(sizes)}); "
                f"falling back to an even split"
            )
        else:
            warnings.append(
                f"ccd_align requested but {len(ccd_ids)} CCD(s) do not divide evenly across "
                f"{n} instances; falling back to an even split, so some instances straddle a "
                f"CCD boundary and get partial L3"
            )

    # Even contiguous split. Remainder goes to the lowest-indexed instances.
    total = len(physical)
    base, extra = divmod(total, n)
    chunks, pos = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(physical[pos : pos + size])
        pos += size
    if extra:
        warnings.append(
            f"{total} physical cores do not divide evenly across {n} instances; "
            f"{extra} instance(s) get {base + 1} cores and {n - extra} get {base} -- "
            f"per-instance throughput is therefore not directly comparable"
        )
    return chunks, False, warnings


@dataclass
class AuxPlacement:
    """Where a non-inference helper process (nginx, the client) is pinned."""

    cpus: list[int]
    membind: list[int]
    source: str          # how these cpus were found -- see auxiliary_cpus()
    note: str = ""

    @property
    def physcpubind(self) -> str:
        return format_cpu_list(self.cpus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpus": self.physcpubind, "n_cpus": len(self.cpus),
            "membind": self.membind, "source": self.source, "note": self.note,
        }


def auxiliary_cpus(
    topo: Topology, plan: AllocationPlan, *, want: int = 4, explicit: str | None = None,
) -> AuxPlacement:
    """Pick cpus for a helper process that must not steal cycles from the instances.

    `explicit` short-circuits everything and is used verbatim -- all of it, not the first
    `want` of it. Otherwise cpus are drawn from these tiers in order until `want` are found;
    a tier that cannot fill the request on its own contributes what it has and the next tier
    tops it up, so a partial `cpu.reserve` is used rather than discarded:

    1. `plan.reserved` -- cores the allocator held back for exactly this purpose.
    2. Any cpu on the instances' own NUMA node whose physical core is running no inference.
       That includes budget cores no instance was given (a `cores_per_instance` remainder) and
       cores outside the budget entirely. Primaries first, then SMT siblings of *idle*
       physical cores -- those share execution units with nothing that matters.
    3. The same, on another node. Cross-socket for the helper, but it still does not touch an
       inference core.
    4. SMT siblings of the *instance* cores -- last resort, and warned about.
    5. The instance cores themselves -- only when the instances hold every logical cpu.

    Note the ordering of 2-3 against 4. An earlier version of this function preferred SMT
    siblings of the instance cores because they cost zero physical cores from the budget,
    which sounds free and is not. A hyperthread is not an idle resource: it shares execution
    units with its sibling, and ggml synchronises all worker threads on a barrier at every
    graph compute, so one thread descheduled by a busy sibling stalls the entire decode step.
    Measured on this host with a 32-core budget and 2 instances, moving nginx off the instance
    cores' siblings onto free physical cores took tg32 TTFT from 144ms to 62ms (direct: 65ms)
    and throughput from 15.3 to 18.7 t/s (direct: 19.2). Sharing SMT with an inference thread
    costs far more than giving the helper its own core.
    """
    inst_cpus = {c for cs in plan.instances for c in cs.cpus}
    inst_physical = {min(topo.cpus[c].siblings) for c in inst_cpus if c in topo.cpus}
    inst_nodes = sorted({n for cs in plan.instances for n in cs.nodes}) or [0]

    def finish(cpus: list[int], source: str, notes: list[str]) -> AuxPlacement:
        overlap = sorted(set(cpus) & inst_cpus)
        if overlap:
            notes = notes + [
                f"overlaps instance cpus {format_cpu_list(overlap)} -- this helper will "
                f"compete with an inference server for those cores"
            ]
        nodes = topo.nodes_of(cpus) or inst_nodes
        return AuxPlacement(cpus=cpus, membind=nodes, source=source,
                            note=" ".join(n.strip() for n in notes if n).strip())

    if explicit:
        # Verbatim: if the config named eight cpus, the helper gets eight. Silently keeping
        # only the first `want` would quietly halve a deliberate placement.
        return finish(parse_cpu_list(explicit), "explicit", [])

    def free_of_inference(c: int) -> bool:
        """True when neither this cpu nor its physical core is running inference."""
        return c not in inst_cpus and min(topo.cpus[c].siblings) not in inst_physical

    def preferred(cpus: Iterable[int]) -> list[int]:
        """Primaries before SMT siblings, then ascending -- deterministic across runs."""
        return sorted(cpus, key=lambda c: (not topo.cpus[c].is_primary, c))

    pool: list[int] = []
    sources: list[str] = []
    notes: list[str] = []

    def take(candidates: Iterable[int], source: str, note: str = "") -> bool:
        """Top the pool up from one tier. Returns True once `want` cpus are held."""
        before = len(pool)
        for c in candidates:
            if len(pool) >= want:
                break
            if c not in pool:
                pool.append(c)
        if len(pool) > before:
            sources.append(source)
            if note:
                notes.append(note)
        return len(pool) >= want

    free = [c for c in topo.cpus if free_of_inference(c)]
    tiers: list[tuple[Iterable[int], str, str]] = [
        (plan.reserved, "reserved", ""),
        (preferred(c for c in free if topo.cpus[c].node in inst_nodes),
         "same-node-idle-cores", ""),
        (preferred(c for c in free if topo.cpus[c].node not in inst_nodes),
         "other-node-idle-cores",
         f"fewer than {want} free cpu(s) on node(s) {inst_nodes}, so the helper (or part of "
         f"it) sits on another socket; it does not contend with inference, but its own "
         f"traffic crosses a socket boundary"),
        (sorted((c for c in topo.siblings_of(inst_cpus) if c not in inst_cpus), reverse=True),
         "smt-siblings-of-instances",
         f"no free cpu remained that does not share a physical core with an inference thread. "
         f"This helper will contend for execution units and, because ggml barriers every "
         f"worker at each graph compute, can stall whole decode steps. Set `cpu.reserve: "
         f"{topo.cores_per_ccd or want}` to give it a CCD of its own instead."),
        (sorted(inst_cpus, reverse=True), "shared-with-instances",
         f"the instances hold every logical cpu on this host, so the helper has nowhere of "
         f"its own to run. Set `cpu.reserve: {topo.cores_per_ccd or want}` or narrow "
         f"`cpu.budget` before trusting latency from this deployment."),
    ]
    for candidates, source, note in tiers:
        if take(candidates, source, note):
            break

    # "+" joins the tiers actually drawn from, so a mixed placement is visible in the manifest
    # rather than being labelled by whichever tier happened to be first.
    return finish(sorted(pool), "+".join(dict.fromkeys(sources)) or "none", notes)


__all__ = [
    "Topology", "Cpu", "CoreSet", "AllocationPlan", "AuxPlacement", "allocate",
    "auxiliary_cpus", "parse_cpu_list", "format_cpu_list", "cpu_mask_hex",
    "TopologyError", "AllocationError",
]
