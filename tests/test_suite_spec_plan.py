"""Spec parsing/validation and expansion into a plan."""
from __future__ import annotations

import pytest

from llmbench.suite.plan import build_plan
from llmbench.suite.spec import SpecError, SuiteSpec
from tests.test_suite_topology import make_topology


def base_spec(**overrides) -> dict:
    data = {
        "name": "t",
        "out_dir": "/tmp/x",
        "mode": "online",
        "cpu": {"budget": "96-191"},
        "lb": {"kind": "client"},
        "backends": {
            "llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server",
                         "offline_bin": "/b/llama-bench", "batched_bin": "/b/batched"},
            "vllm": {"model": "/m/hf", "server_bin": "/b/vllm", "offline_bin": "/b/vllm"},
        },
        "deployment": {"backend": ["llamacpp"], "instances": [1]},
        "workload": {"n_prompt": [64], "n_gen": [32], "reps": 3},
    }
    data.update(overrides)
    return data


# --- spec validation ---


def test_scalars_are_promoted_to_axes():
    spec = SuiteSpec.from_dict(base_spec(deployment={"backend": "llamacpp", "instances": 4,
                                                     "n_ctx": 8192}))
    assert spec.deployment.instances == [4]
    assert spec.deployment.n_ctx == [8192]
    assert spec.deployment.backend == ["llamacpp"]


def test_range_syntax_is_accepted_on_int_axes():
    spec = SuiteSpec.from_dict(base_spec(deployment={"backend": ["llamacpp"],
                                                     "instances": "1-8*2"}))
    assert spec.deployment.instances == [1, 2, 4, 8]


def test_unknown_key_is_an_error_not_a_silent_typo():
    with pytest.raises(SpecError, match="instaces"):
        SuiteSpec.from_dict(base_spec(deployment={"backend": ["llamacpp"], "instaces": 4}))


def test_unknown_top_level_key_is_an_error():
    with pytest.raises(SpecError, match="unknown key"):
        SuiteSpec.from_dict(base_spec(objectiv={"metric": "x"}))


def test_backend_without_matching_backends_section_is_rejected():
    data = base_spec()
    data["backends"].pop("vllm")
    data["deployment"]["backend"] = ["llamacpp", "vllm"]
    with pytest.raises(SpecError, match="no matching `backends:` section"):
        SuiteSpec.from_dict(data)


def test_server_bin_required_for_online():
    data = base_spec()
    data["backends"]["llamacpp"].pop("server_bin")
    with pytest.raises(SpecError, match="server_bin is required"):
        SuiteSpec.from_dict(data)


def test_offline_requires_an_offline_tool():
    data = base_spec(mode="offline")
    data["backends"]["llamacpp"].pop("offline_bin")
    data["backends"]["llamacpp"].pop("batched_bin")
    with pytest.raises(SpecError, match="offline_bin"):
        SuiteSpec.from_dict(data)


def test_multi_instance_without_a_load_balancer_is_rejected():
    """Otherwise the client would silently address instance 0 only and report it as a fleet."""
    with pytest.raises(SpecError, match="cannot serve a multi-instance"):
        SuiteSpec.from_dict(base_spec(
            lb={"kind": "none"}, deployment={"backend": ["llamacpp"], "instances": [4]},
        ))


def test_constraint_needs_a_bound():
    with pytest.raises(SpecError, match="at least one of"):
        SuiteSpec.from_dict(base_spec(constraints=[{"metric": "ttft_ms_p99"}]))


def test_bad_enum_is_rejected():
    with pytest.raises(SpecError, match="not one of"):
        SuiteSpec.from_dict(base_spec(cpu={"budget": "96-191", "smt": "sometimes"}))


def test_objective_describes_itself():
    spec = SuiteSpec.from_dict(base_spec(
        objective={"metric": "total_token_throughput", "goal": "max"},
        constraints=[{"metric": "ttft_ms_p99", "max": 2000}],
    ))
    text = spec.objective.describe()
    assert "maximise total_token_throughput" in text
    assert "ttft_ms_p99 <= 2000" in text


# --- plan expansion ---


@pytest.fixture
def topo():
    return make_topology()


def test_deployment_is_the_outer_loop(topo):
    """Workloads must be grouped under one deployment so servers relaunch as rarely as
    possible -- the whole reason the two axis groups are separate."""
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [1, 2], "n_parallel": [1, 8]},
        workload={"n_prompt": [16, 64], "n_gen": [32], "concurrency": [1, 4], "reps": 2},
    ))
    plan = build_plan(spec, topo)
    assert plan.n_deployments == 4                      # 2 instances x 2 n_parallel
    assert all(len(ws) == 6 for _, ws in plan.groups)   # (2 pp + 1 tg) x 2 concurrency
    assert plan.n_online_trials == 24


def test_pp_tg_pg_are_parallel_loops_not_nested(topo):
    spec = SuiteSpec.from_dict(base_spec(
        workload={"n_prompt": [16, 32], "n_gen": [64], "pg": ["128,32"], "reps": 1},
    ))
    _, workloads = build_plan(spec, topo).groups[0]
    assert [w.test_name() for w in workloads] == ["pp16", "pp32", "tg64", "pp128+tg32"]


def test_zero_valued_axis_entries_are_skipped(topo):
    spec = SuiteSpec.from_dict(base_spec(workload={"n_prompt": [0, 64], "n_gen": [0], "reps": 1}))
    _, workloads = build_plan(spec, topo).groups[0]
    assert [w.test_name() for w in workloads] == ["pp64"]


def test_ports_are_assigned_per_instance_from_base_port(topo):
    spec = SuiteSpec.from_dict(base_spec(deployment={"backend": ["llamacpp"], "instances": [4]}))
    dep = build_plan(spec, topo).groups[0][0]
    assert [i.port for i in dep.instances] == [8100, 8101, 8102, 8103]
    assert dep.urls[2] == "http://127.0.0.1:8102"


def test_backends_get_distinct_port_ranges(topo):
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp", "vllm"], "instances": [1]}))
    ports = {d.backend: d.instances[0].port for d, _ in build_plan(spec, topo).groups}
    assert ports == {"llamacpp": 8100, "vllm": 8200}


def test_single_instance_client_lb_collapses_to_none(topo):
    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "client"}, deployment={"backend": ["llamacpp"], "instances": [1, 2]}))
    kinds = {d.n_instances: d.lb_kind for d, _ in build_plan(spec, topo).groups}
    assert kinds == {1: "none", 2: "client"}


def test_uniform_nginx_routes_even_a_single_instance_through_the_proxy(topo):
    """So 1-instance and N-instance latencies stay comparable: either both pay the proxy hop
    or neither does."""
    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "nginx", "uniform": True},
        deployment={"backend": ["llamacpp"], "instances": [1, 2]}))
    for dep, _ in build_plan(spec, topo).groups:
        assert dep.lb_kind == "nginx"
        assert dep.client_url == "http://127.0.0.1:18081"


def test_non_uniform_nginx_skips_the_proxy_for_one_instance(topo):
    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "nginx", "uniform": False},
        deployment={"backend": ["llamacpp"], "instances": [1, 2]}))
    kinds = {d.n_instances: d.lb_kind for d, _ in build_plan(spec, topo).groups}
    assert kinds == {1: "none", 2: "nginx"}


def test_nginx_deployment_warns_that_the_proxy_is_measured(topo):
    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "nginx"}, deployment={"backend": ["llamacpp"], "instances": [2]}))
    dep = build_plan(spec, topo).groups[0][0]
    assert any("inside the measured wire-to-wire window" in w for w in dep.warnings)
    assert dep.lb_placement is not None


def test_concurrency_below_instance_count_is_warned(topo):
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [4]},
        workload={"n_prompt": [64], "n_gen": [0], "concurrency": [1], "reps": 1}))
    dep = build_plan(spec, topo).groups[0][0]
    assert any("stay idle" in w for w in dep.warnings)


def test_unsatisfiable_deployment_is_skipped_not_fatal(topo):
    """One impossible point must not abort a sweep that has many valid ones."""
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [2],
                    "cores_per_instance": [8, 900]}))
    plan = build_plan(spec, topo)
    assert plan.n_deployments == 1
    assert any("skipped deployment" in w for w in plan.warnings)


# --- offline planning ---


def test_offline_tool_selection_by_batch_size(topo):
    """llama-bench cannot express a request batch, so batch_size > 1 must route to
    llama-batched-bench."""
    spec = SuiteSpec.from_dict(base_spec(
        mode="offline", deployment={"backend": ["llamacpp"]},
        offline={"batch_size": [1, 8], "n_prompt": [128], "n_gen": [32]}))
    tools = {o.batch_size: o.tool for o in build_plan(spec, topo).offline}
    assert tools == {1: "llama-bench", 8: "llama-batched-bench"}


def test_offline_vllm_uses_latency_for_static_batch(topo):
    spec = SuiteSpec.from_dict(base_spec(
        mode="offline", deployment={"backend": ["vllm"]},
        offline={"batch_size": [1, 4], "n_prompt": [128], "n_gen": [32]}))
    assert {o.tool for o in build_plan(spec, topo).offline} == {"vllm-latency"}


def test_offline_takes_the_whole_budget_as_one_core_set(topo):
    spec = SuiteSpec.from_dict(base_spec(
        mode="offline", deployment={"backend": ["llamacpp"], "instances": [4]},
        offline={"batch_size": [1], "n_prompt": [128], "n_gen": [32]}))
    plan = build_plan(spec, topo)
    assert all(o.cores.n_physical_cores == 96 for o in plan.offline)


def test_throughput_prompts_only_apply_to_vllm(topo):
    spec = SuiteSpec.from_dict(base_spec(
        mode="offline", deployment={"backend": ["llamacpp"]},
        offline={"batch_size": [1], "n_prompt": [128], "n_gen": [32],
                 "throughput_prompts": [100]}))
    plan = build_plan(spec, topo)
    assert not any(o.tool == "vllm-throughput" for o in plan.offline)
    assert any("only applies to vllm" in w for w in plan.warnings)


def test_mode_both_produces_online_and_offline(topo):
    spec = SuiteSpec.from_dict(base_spec(
        mode="both", deployment={"backend": ["llamacpp"], "instances": [1]},
        workload={"n_prompt": [64], "n_gen": [0], "reps": 1},
        offline={"batch_size": [1], "n_prompt": [128], "n_gen": [32]}))
    plan = build_plan(spec, topo)
    assert plan.n_online_trials == 1 and plan.n_offline_trials == 1


def test_plan_serialises_fully(topo):
    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "nginx"}, deployment={"backend": ["llamacpp"], "instances": [2]}))
    d = build_plan(spec, topo).to_dict()
    dep = d["deployments"][0]["deployment"]
    assert dep["instances"][0]["cpus"] == "96-143"
    assert dep["lb"]["kind"] == "nginx"
    assert d["topology"]["cores_per_ccd"] == 8


def test_reps_close_to_concurrency_is_warned(topo):
    """reps barely above concurrency makes ramp-up and drain the measurement. Measured here:
    reps=6 at concurrency=4 gave 26-46% rsd where reps=40 gave 3-8%."""
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [1], "n_parallel": [8]},
        workload={"n_prompt": [64], "n_gen": [0], "concurrency": [4], "reps": 6}))
    dep = build_plan(spec, topo).groups[0][0]
    assert any("ramp-up and drain" in w for w in dep.warnings)


def test_ample_reps_is_not_warned(topo):
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [1], "n_parallel": [8]},
        workload={"n_prompt": [64], "n_gen": [0], "concurrency": [4], "reps": 40}))
    dep = build_plan(spec, topo).groups[0][0]
    assert not any("ramp-up and drain" in w for w in dep.warnings)


def test_concurrency_one_is_never_warned_for_reps(topo):
    """At concurrency 1 there is no ramp-up transient to dilute."""
    spec = SuiteSpec.from_dict(base_spec(
        deployment={"backend": ["llamacpp"], "instances": [1], "n_parallel": [8]},
        workload={"n_prompt": [64], "n_gen": [0], "concurrency": [1], "reps": 3}))
    dep = build_plan(spec, topo).groups[0][0]
    assert not any("ramp-up and drain" in w for w in dep.warnings)


# --- axis values that would produce a broken server command ---


@pytest.mark.parametrize("axis", ["n_parallel", "n_ctx", "batch", "ubatch",
                                  "cores_per_instance", "threads_per_instance"])
def test_non_positive_deployment_axis_is_rejected(axis):
    """Each of these is passed straight to a server flag. Caught at parse time the message
    names the spec key; caught at launch time it is a backend usage error an hour into a
    sweep. `n_parallel: 0` was additionally silent -- it disabled the capacity pre-flight's
    slot-count guard, so every trial ran unchecked."""
    with pytest.raises(SpecError, match=axis):
        SuiteSpec.from_dict(base_spec(
            deployment={"backend": ["llamacpp"], "instances": [1], axis: [0]}))


def test_lb_port_colliding_with_the_instance_range_is_rejected():
    """Instance ports are base_port + index, so nginx on 8101 and instance 1 on 8101 both
    try to bind the same socket -- the second one loses, mid-sweep."""
    with pytest.raises(SpecError, match="collides"):
        SuiteSpec.from_dict(base_spec(
            deployment={"backend": ["llamacpp"], "instances": [4]},
            lb={"kind": "nginx", "port": 8101},
        ))


def test_base_port_too_high_for_the_instance_count_is_rejected():
    with pytest.raises(SpecError, match="base_port"):
        SuiteSpec.from_dict(base_spec(
            deployment={"backend": ["llamacpp"], "instances": [8]},
            backends={"llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/s",
                                   "base_port": 65534}},
        ))


def test_thread_count_above_the_allocation_is_warned(topo):
    """`-t` more than the cpuset holds buys nothing: numactl pins the surplus threads into the
    same cpus, and ggml's per-compute barrier makes the extra context switches strictly
    negative."""
    spec = SuiteSpec.from_dict(base_spec(
        cpu={"budget": "96-103"},
        deployment={"backend": ["llamacpp"], "instances": [1], "threads_per_instance": [32]},
    ))
    dep, _ = build_plan(spec, topo).groups[0]
    assert any("oversubscribe" in w for w in dep.warnings)


def test_thread_count_below_the_allocation_is_warned(topo):
    spec = SuiteSpec.from_dict(base_spec(
        cpu={"budget": "96-103"},
        deployment={"backend": ["llamacpp"], "instances": [1], "threads_per_instance": [4]},
    ))
    dep, _ = build_plan(spec, topo).groups[0]
    assert any("stay idle" in w for w in dep.warnings)


def test_matching_thread_count_is_not_warned(topo):
    spec = SuiteSpec.from_dict(base_spec(
        cpu={"budget": "96-103"},
        deployment={"backend": ["llamacpp"], "instances": [1], "threads_per_instance": [8]},
    ))
    dep, _ = build_plan(spec, topo).groups[0]
    assert not any("threads_per_instance" in w for w in dep.warnings)


# --- prefix-cache experiments ---


def test_shared_prefix_below_the_cache_block_is_warned(topo):
    """A prefix cache matches whole blocks (16 tokens on vLLM), so a shorter shared prefix
    can never produce a hit -- the experiment measures nothing and reads as 'prefix caching
    does not help here'."""
    spec = SuiteSpec.from_dict(base_spec(
        workload={"n_prompt": [512], "n_gen": [32], "shared_prefix": [8], "reps": 5}))
    dep, _ = build_plan(spec, topo).groups[0]
    assert any("below the 16-token block" in w for w in dep.warnings)


def test_shared_prefix_at_or_above_the_block_is_not_warned(topo):
    spec = SuiteSpec.from_dict(base_spec(
        workload={"n_prompt": [512], "n_gen": [32], "shared_prefix": [0, 256], "reps": 5}))
    dep, _ = build_plan(spec, topo).groups[0]
    assert not any("block a prefix cache" in w for w in dep.warnings)


def test_a_fully_shared_prompt_is_warned_as_degenerate(topo):
    """shared_prefix >= n_prompt leaves nothing unique, so every request is a total hit and
    the row times cache retrieval rather than prefill."""
    spec = SuiteSpec.from_dict(base_spec(
        workload={"n_prompt": [128], "n_gen": [32], "shared_prefix": [128], "reps": 5}))
    dep, _ = build_plan(spec, topo).groups[0]
    assert any("no prefill left to measure" in w for w in dep.warnings)
