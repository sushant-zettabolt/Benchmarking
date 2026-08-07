"""Topology parsing and core allocation.

These tests build a synthetic 2-socket / 24-CCD / SMT machine rather than reading sysfs, so
they assert the allocator's logic rather than the host they happen to run on.
"""
from __future__ import annotations

import pytest

from llmbench.suite.topology import (
    AllocationError, Cpu, Topology, allocate, auxiliary_cpus, cpu_mask_hex, format_cpu_list,
    parse_cpu_list,
)


def make_topology(*, sockets=2, ccds_per_socket=12, cores_per_ccd=8, smt=True) -> Topology:
    """Mirrors this project's EPYC 9R14 host: node N holds cores [96N..96N+95] with SMT
    siblings offset by 192, and L3 is private per 8-core CCD."""
    cpus: dict[int, Cpu] = {}
    nodes: dict[int, list[int]] = {}
    ccd_members: dict[int, list[int]] = {}
    per_socket = ccds_per_socket * cores_per_ccd
    total_physical = sockets * per_socket

    for socket in range(sockets):
        node_cpus = []
        for i in range(per_socket):
            primary = socket * per_socket + i
            siblings = (primary, primary + total_physical) if smt else (primary,)
            ccd = socket * ccds_per_socket + i // cores_per_ccd
            for cpu_id in siblings:
                cpus[cpu_id] = Cpu(id=cpu_id, core_id=i, socket=socket, node=socket,
                                   ccd=ccd, siblings=siblings)
                node_cpus.append(cpu_id)
                ccd_members.setdefault(ccd, []).append(cpu_id)
        nodes[socket] = sorted(node_cpus)

    distances = {a: {b: (10 if a == b else 32) for b in range(sockets)} for a in range(sockets)}
    return Topology(cpus=cpus, nodes=nodes, distances=distances,
                    ccd_members={k: sorted(v) for k, v in ccd_members.items()},
                    model_name="synthetic EPYC")


@pytest.fixture
def topo() -> Topology:
    return make_topology()


# --- cpu list round-tripping ---


@pytest.mark.parametrize("spec,expected", [
    ("96-99", [96, 97, 98, 99]),
    ("0-7,192-199", [0, 1, 2, 3, 4, 5, 6, 7, 192, 193, 194, 195, 196, 197, 198, 199]),
    ("5", [5]),
    ("3,1,2", [1, 2, 3]),
    ("", []),
])
def test_parse_cpu_list(spec, expected):
    assert parse_cpu_list(spec) == expected


def test_parse_cpu_list_rejects_descending():
    with pytest.raises(ValueError, match="descending"):
        parse_cpu_list("10-2")


@pytest.mark.parametrize("cpus,expected", [
    ([96, 97, 98, 99], "96-99"),
    ([1, 2, 3, 7, 8], "1-3,7-8"),
    ([5], "5"),
    ([], ""),
])
def test_format_cpu_list(cpus, expected):
    assert format_cpu_list(cpus) == expected
    assert parse_cpu_list(expected) == sorted(cpus)


def test_cpu_mask_hex_sets_the_right_bits():
    # llama.cpp's -C takes a hex bitmask; bit N == cpu N.
    assert cpu_mask_hex([0, 1, 2, 3]) == "f"
    assert cpu_mask_hex([4]) == "10"
    assert int(cpu_mask_hex([96, 97]), 16) == (1 << 96) | (1 << 97)


# --- topology derivation ---


def test_topology_shape(topo):
    assert topo.n_logical == 384
    assert topo.n_physical == 192
    assert topo.smt_enabled
    assert topo.cores_per_ccd == 8
    assert len(topo.ccd_members) == 24
    assert topo.nodes_of(parse_cpu_list("96-191")) == [1]


def test_primaries_excludes_smt_siblings(topo):
    assert topo.primaries(parse_cpu_list("96-99,288-291")) == [96, 97, 98, 99]


def test_siblings_of_includes_self(topo):
    assert topo.siblings_of([96]) == [96, 288]


# --- allocation ---


@pytest.mark.parametrize("n,cores_each", [(1, 96), (2, 48), (3, 32), (4, 24), (6, 16), (12, 8)])
def test_ccd_aligned_splits(topo, n, cores_each):
    """96 cores is 12 whole CCDs, so these divisors must all come out CCD-aligned."""
    plan = allocate(topo, "96-191", n_instances=n)
    assert plan.ccd_aligned, plan.warnings
    assert len(plan.instances) == n
    assert all(cs.n_physical_cores == cores_each for cs in plan.instances)
    # Instances must not overlap, or "isolated" instances silently share cores.
    seen: set[int] = set()
    for cs in plan.instances:
        assert not (seen & set(cs.cpus))
        seen |= set(cs.cpus)
        assert cs.nodes == [1]
        assert cs.membind == [1]


def test_non_dividing_split_is_flagged_not_hidden(topo):
    plan = allocate(topo, "96-191", n_instances=8)
    assert not plan.ccd_aligned
    assert any("do not divide evenly" in w for w in plan.warnings)


def test_uneven_core_counts_are_warned(topo):
    plan = allocate(topo, "96-191", n_instances=7)
    sizes = {cs.n_physical_cores for cs in plan.instances}
    assert len(sizes) > 1
    assert any("not directly comparable" in w for w in plan.warnings)


def test_reserve_rounds_up_to_whole_ccd_to_preserve_alignment(topo):
    """A 4-core reserve off a 12-CCD budget would leave 92 cores, which divides evenly by
    nothing. Rounding to a whole CCD keeps the remainder CCD-quantised."""
    plan = allocate(topo, "96-191", n_instances=1, reserve=4)
    assert len(plan.reserved) == 8
    assert plan.ccd_aligned
    assert any("rounded up to 8" in w for w in plan.warnings)


def test_reserve_exact_when_ccd_align_disabled(topo):
    plan = allocate(topo, "96-191", n_instances=1, reserve=4, ccd_align=False)
    assert len(plan.reserved) == 4


def test_smt_include_doubles_threads_but_not_cores(topo):
    plan = allocate(topo, "96-191", n_instances=2, smt="include")
    cs = plan.instances[0]
    assert cs.n_physical_cores == 48
    assert cs.n_threads == 96
    assert set(cs.cpus) == set(parse_cpu_list("96-143,288-335"))


def test_smt_only_siblings_selects_the_non_primary_thread(topo):
    plan = allocate(topo, "96-103", n_instances=1, smt="only-siblings")
    assert set(plan.instances[0].cpus) == set(parse_cpu_list("288-295"))


def test_cross_socket_allocation_is_warned(topo):
    plan = allocate(topo, "0-191", n_instances=1)
    assert plan.instances[0].cross_socket
    assert any("spans sockets" in w for w in plan.warnings)


def test_cores_per_instance_leaves_remainder_and_says_so(topo):
    plan = allocate(topo, "96-191", n_instances=2, cores_per_instance=16)
    assert [cs.n_physical_cores for cs in plan.instances] == [16, 16]
    assert any("unused" in w for w in plan.warnings)


def test_membind_explicit_overrides_auto(topo):
    plan = allocate(topo, "96-191", n_instances=1, membind="0")
    assert plan.instances[0].membind == [0]


def test_membind_none_disables_binding(topo):
    plan = allocate(topo, "96-191", n_instances=1, membind="none")
    assert plan.instances[0].membind == []
    assert plan.instances[0].membind_arg == ""


@pytest.mark.parametrize("kwargs,match", [
    (dict(n_instances=0), "must be >= 1"),
    (dict(n_instances=200), "cannot split"),
    (dict(n_instances=1, reserve=96), "entire budget"),
    (dict(n_instances=4, cores_per_instance=40), "only 96 are available"),
    (dict(n_instances=1, smt="bogus"), "unknown smt policy"),
])
def test_allocation_errors(topo, kwargs, match):
    with pytest.raises(AllocationError, match=match):
        allocate(topo, "96-191", **kwargs)


def test_budget_outside_host_is_rejected(topo):
    with pytest.raises(AllocationError, match="no usable cpus"):
        allocate(topo, "1000-1010", n_instances=1)


def test_budget_partially_outside_host_warns_and_continues(topo):
    plan = allocate(topo, "96-103,1000-1001", n_instances=1)
    assert plan.instances[0].n_physical_cores == 8
    assert any("not present on this host" in w for w in plan.warnings)


def test_budget_of_only_smt_siblings_is_rejected(topo):
    with pytest.raises(AllocationError, match="SMT siblings only"):
        allocate(topo, "288-295", n_instances=1)


# --- auxiliary placement (nginx / client) ---


def _shares_physical_core(topo, cpus, plan) -> bool:
    inst_phys = {min(topo.cpus[c].siblings) for cs in plan.instances for c in cs.cpus}
    return bool({min(topo.cpus[c].siblings) for c in cpus} & inst_phys)


def test_aux_never_shares_a_physical_core_with_inference(topo):
    """A hyperthread is not a free resource. It shares execution units with its sibling, and
    ggml barriers every worker at each graph compute -- so a helper on an instance core's
    sibling can stall whole decode steps. Measured on this host: moving nginx off the
    siblings took tg32 TTFT from 144ms to 62ms."""
    plan = allocate(topo, "0-31", n_instances=2)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "same-node-idle-cores"
    assert aux.membind == [0]
    assert len(aux.cpus) == 4
    assert not _shares_physical_core(topo, aux.cpus, plan)


def test_aux_prefers_same_node_idle_cores_over_another_socket(topo):
    plan = allocate(topo, "0-31", n_instances=1)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert topo.nodes_of(aux.cpus) == [0]
    assert not _shares_physical_core(topo, aux.cpus, plan)


def test_aux_crosses_sockets_rather_than_sharing_an_inference_core(topo):
    """When the budget is a whole node there is no on-node core left, and the correct
    trade-off is a socket hop for the proxy rather than SMT contention with decode."""
    plan = allocate(topo, "0-95", n_instances=2)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "other-node-idle-cores"
    assert not _shares_physical_core(topo, aux.cpus, plan)
    assert "crosses a socket" in aux.note


def test_aux_reserved_ccd_beats_crossing_a_socket(topo):
    """The clean answer when the budget covers a whole node: give the helper its own CCD."""
    plan = allocate(topo, "0-95", n_instances=2, reserve=8)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "reserved"
    assert topo.nodes_of(aux.cpus) == [0]
    assert not _shares_physical_core(topo, aux.cpus, plan)


def test_aux_uses_reserved_cores_when_present(topo):
    plan = allocate(topo, "96-191", n_instances=1, reserve=8)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "reserved"
    assert set(aux.cpus) <= set(plan.reserved)


def test_aux_explicit_overlap_is_reported_not_silently_accepted(topo):
    plan = allocate(topo, "96-191", n_instances=1)
    aux = auxiliary_cpus(topo, plan, want=2, explicit="96-97")
    assert aux.source == "explicit"
    assert "overlaps instance cpus" in aux.note


def test_aux_last_resort_shares_smt_but_says_so(topo):
    """Instances own every physical core on the machine (both sockets), so the only cpus left
    are siblings of in-use cores. Falling back there is then unavoidable, but it must be
    reported along with the fix rather than passed off as free."""
    plan = allocate(topo, "0-191", n_instances=1)          # smt=exclude: siblings stay free
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "smt-siblings-of-instances"
    assert "stall whole decode steps" in aux.note
    assert "cpu.reserve" in aux.note


def test_aux_absolute_last_resort_when_even_siblings_are_taken(topo):
    """smt=include over both sockets leaves literally nothing; the overlap must still be
    stated so the run is not silently self-contending."""
    plan = allocate(topo, "0-191", n_instances=1, smt="include")
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "shared-with-instances"
    assert "overlaps instance cpus" in aux.note
    assert "cpu.reserve" in aux.note


def test_aux_explicit_still_wins_over_everything(topo):
    plan = allocate(topo, "0-31", n_instances=1)
    aux = auxiliary_cpus(topo, plan, want=2, explicit="64-65")
    assert aux.source == "explicit"
    assert aux.cpus == [64, 65]


def test_topology_serialises_for_the_manifest(topo):
    d = topo.to_dict()
    assert d["n_physical"] == 192 and d["cores_per_ccd"] == 8
    assert d["nodes"]["1"] == "96-191,288-383"
    assert d["node_distances"]["0"][1] == 32


def test_aux_explicit_cpus_are_used_verbatim_not_truncated(topo):
    """`want` is how many cpus to *find*, not a cap on what the operator asked for. Silently
    keeping the first `lb.n_cpus` of an explicit `lb.cpus` would quietly halve a deliberate
    placement and leave nginx with a quarter of the cores its config named."""
    plan = allocate(topo, "0-31", n_instances=1)
    aux = auxiliary_cpus(topo, plan, want=4, explicit="88-95")
    assert aux.cpus == list(range(88, 96))


def test_aux_tops_a_partial_reserve_up_instead_of_discarding_it(topo):
    """A reserve smaller than the helper needs used to be skipped outright -- and, being
    inside the budget, was then invisible to the next tier too, so cores held back for exactly
    this purpose went unused while the helper was placed elsewhere."""
    plan = allocate(topo, "0-31", n_instances=1, reserve=2, ccd_align=False)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert len(aux.cpus) == 4
    assert set(plan.reserved) <= set(aux.cpus)
    assert aux.source == "reserved+same-node-idle-cores"
    assert not _shares_physical_core(topo, aux.cpus, plan)


def test_aux_can_use_budget_cores_no_instance_was_given(topo):
    """cores_per_instance can leave part of the budget unallocated. Those cores are idle and
    on-node -- the best possible place for the helper -- but an in-budget check used to
    exclude them, pushing nginx onto another socket while they sat empty."""
    plan = allocate(topo, "0-31", n_instances=1, cores_per_instance=8)
    aux = auxiliary_cpus(topo, plan, want=4)
    assert aux.source == "same-node-idle-cores"
    assert set(aux.cpus) <= set(range(8, 32))
    assert not _shares_physical_core(topo, aux.cpus, plan)
