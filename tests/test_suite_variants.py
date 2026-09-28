"""Named backend variants: several installations of one engine in a single sweep.

`backends:` keys used to double as the engine type, so a spec could hold one llama.cpp and
one vLLM and nothing else -- no ZenDNN build next to a stock build, no BF16 next to Q8 in one
table. A key is now a name, and `type:` says which engine it is. These pin that the type,
never the name, decides how a server is launched, which client talks to it and which offline
tool runs; and that the name is what reports and rankings show.
"""
from __future__ import annotations

import os

import pytest

from llmbench.backends.llamacpp import LlamaCppBackend
from llmbench.backends.vllm import VllmBackend
from llmbench.suite.deploy import build_server_command
from llmbench.suite.lb import make_backend
from llmbench.suite.plan import build_plan
from llmbench.suite.spec import SpecError, SuiteSpec
from tests.test_suite_topology import make_topology

VARIANTS = {
    "llamacpp-zendnn-q8": {"type": "llamacpp", "model": "/m/q8.gguf",
                           "server_bin": "/zendnn/llama-server",
                           "offline_bin": "/zendnn/llama-bench"},
    "llamacpp-q8": {"type": "llamacpp", "model": "/m/q8.gguf", "server_bin": "/stock/llama-server",
                    "offline_bin": "/stock/llama-bench"},
    "vllm-w8a8": {"type": "vllm", "model": "/m/w8a8", "server_bin": "/v/vllm",
                  "offline_bin": "/v/vllm"},
}


def spec_with(backends, *, mode="online", **overrides) -> SuiteSpec:
    data = {
        "name": "t", "out_dir": "/tmp/unused", "mode": mode,
        "cpu": {"budget": "96-103"}, "lb": {"kind": "client"},
        "backends": backends,
        "deployment": {"backend": list(backends), "instances": [1]},
        "workload": {"n_prompt": [64], "n_gen": [0], "reps": 3, "no_warmup": True},
        "offline": {"batch_size": [1], "n_prompt": [64], "n_gen": [16], "reps": 1},
    }
    data.update(overrides)
    return SuiteSpec.from_dict(data)


def test_each_variant_is_its_own_deployment_launched_from_its_own_binary():
    plan = build_plan(spec_with(VARIANTS), make_topology())
    by_name = {dep.backend: dep for dep, _ in plan.groups}
    assert set(by_name) == set(VARIANTS)

    argv, _ = build_server_command(by_name["llamacpp-zendnn-q8"],
                                   by_name["llamacpp-zendnn-q8"].instances[0], plan.spec.cpu)
    assert "/zendnn/llama-server" in argv and "-np" in argv
    argv, _ = build_server_command(by_name["llamacpp-q8"],
                                   by_name["llamacpp-q8"].instances[0], plan.spec.cpu)
    assert "/stock/llama-server" in argv
    argv, env = build_server_command(by_name["vllm-w8a8"],
                                     by_name["vllm-w8a8"].instances[0], plan.spec.cpu)
    assert "serve" in argv and "--max-num-seqs" in argv and "VLLM_CPU_OMP_THREADS_BIND" in env


def test_the_client_is_chosen_by_type_not_by_name():
    plan = build_plan(spec_with(VARIANTS), make_topology())
    classes = {dep.backend: type(make_backend(dep, plan.spec)) for dep, _ in plan.groups}
    assert classes == {"llamacpp-zendnn-q8": LlamaCppBackend, "llamacpp-q8": LlamaCppBackend,
                       "vllm-w8a8": VllmBackend}


def test_the_offline_tool_is_chosen_by_type_not_by_name():
    plan = build_plan(spec_with(VARIANTS, mode="offline"), make_topology())
    tools = {o.backend: o.tool for o in plan.offline}
    assert tools == {"llamacpp-zendnn-q8": "llama-bench", "llamacpp-q8": "llama-bench",
                     "vllm-w8a8": "vllm-latency"}


def test_rows_and_config_labels_carry_the_variant_name():
    """Two llama.cpp variants must not collapse into one `llamacpp` config in the ranking."""
    plan = build_plan(spec_with(VARIANTS), make_topology())
    assert {dep.axes()["backend"] for dep, _ in plan.groups} == set(VARIANTS)
    assert {dep.to_dict()["backend_type"] for dep, _ in plan.groups} == {"llamacpp", "vllm"}


def test_a_key_that_is_a_type_needs_no_type_field():
    """Every spec written before variants existed keeps working unchanged."""
    spec = spec_with({"llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server"}})
    assert spec.backends["llamacpp"].type == "llamacpp"
    assert spec.backends["llamacpp"].base_port == 8100


def test_a_variant_gets_its_type_s_default_port():
    spec = spec_with({"vllm-bf16": {"type": "vllm", "model": "/m/hf", "server_bin": "/v/vllm"}})
    assert spec.backends["vllm-bf16"].base_port == 8200


def test_an_unknown_name_without_a_type_says_how_to_declare_a_variant():
    with pytest.raises(SpecError, match="add `type: llamacpp`"):
        spec_with({"zendnn": {"model": "/m/x.gguf", "server_bin": "/b/llama-server"}})


def test_an_unknown_type_is_refused():
    with pytest.raises(SpecError, match="unknown backend type 'sglang'"):
        spec_with({"x": {"type": "sglang", "model": "/m", "server_bin": "/b"}})


def test_a_name_that_is_not_path_safe_is_refused():
    with pytest.raises(SpecError, match="not a usable name"):
        spec_with({"llama cpp/zendnn": {"type": "llamacpp", "model": "/m", "server_bin": "/b"}})


def test_deployment_backend_must_name_a_configured_variant():
    with pytest.raises(SpecError, match="no matching `backends:`"):
        spec_with(VARIANTS, deployment={"backend": ["llamacpp-zendnn-bf16"], "instances": [1]})


def test_offline_mode_checks_each_variant_by_its_type():
    with pytest.raises(SpecError, match="backends.vllm-w8a8.offline_bin"):
        spec_with({"vllm-w8a8": {"type": "vllm", "model": "/m", "server_bin": "/v/vllm"}},
                  mode="offline")


def test_a_home_relative_path_reaches_the_server_expanded():
    """argv goes to the server without a shell, so a literal `~` used to reach llama-server."""
    spec = spec_with({"llamacpp": {"model": "~/models/gguf/x.gguf", "server_bin": "~/bin/llama-server"}})
    home = os.path.expanduser("~")
    assert spec.backends["llamacpp"].model == os.path.join(home, "models", "gguf", "x.gguf")
    assert spec.backends["llamacpp"].server_bin.startswith(home)


def test_a_backend_s_own_axis_values_replace_the_global_ones_rather_than_multiply_them():
    """llama.cpp at -c 32000 and vLLM at --max-model-len 8192 is two deployments, not four."""
    backends = {
        "llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server",
                     "deployment": {"n_ctx": 32000, "ubatch": [256, 512]}},
        "vllm": {"model": "/m/hf", "server_bin": "/v/vllm"},
    }
    plan = build_plan(spec_with(backends, deployment={
        "backend": ["llamacpp", "vllm"], "instances": [1], "n_ctx": [8192], "ubatch": [4096],
    }), make_topology())

    points = sorted((d.backend, d.n_ctx, d.ubatch) for d, _ in plan.groups)
    assert points == [("llamacpp", 32000, 256), ("llamacpp", 32000, 512), ("vllm", 8192, 4096)]
    llama = next(d for d, _ in plan.groups if d.backend == "llamacpp")
    argv, _ = build_server_command(llama, llama.instances[0], plan.spec.cpu)
    assert argv[argv.index("-c") + 1] == "32000"
    assert llama.axes()["n_ctx"] == 32000, "the row must record the value actually launched"


def test_only_server_axes_can_be_overridden_per_backend():
    with pytest.raises(SpecError, match="backends.llamacpp.deployment: unknown key"):
        spec_with({"llamacpp": {"model": "/m", "server_bin": "/b",
                                "deployment": {"instances": [2]}}})


def test_x_prefixed_top_level_keys_hold_anchors_and_are_ignored():
    spec = spec_with({"llamacpp": {"model": "/m/x.gguf", "server_bin": "/b/llama-server"}},
                     **{"x-shared-env": {"OMP_NUM_THREADS": "32"}})
    assert spec.name == "t"
    with pytest.raises(SpecError, match="unknown key"):
        spec_with({"llamacpp": {"model": "/m", "server_bin": "/b"}}, xshared={"a": 1})


def test_the_turin_spec_in_the_repo_matches_the_reference_runs():
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "sweep.turin-32c-8b.yaml"
    spec = SuiteSpec.from_yaml(path)
    variants = ["llamacpp-zendnn-bf16", "llamacpp-zendnn-q8", "llamacpp-bf16", "llamacpp-q8",
                "vllm-bf16", "vllm-w8a8"]
    assert set(spec.deployment.backend) == {f"{m}-{v}" for m in ("llama31", "qwen36")
                                            for v in variants}
    assert spec.cpu.budget == "160-191" and spec.cpu.membind == "5"       # pod-5's cpuset
    assert spec.deployment.instances == [1] and spec.deployment.n_parallel == [1]
    assert spec.workload.reps == 3 and spec.workload.warmup_fixed == 1
    assert spec.workload.n_prompt == [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768,
                                      1024, 1536, 2048, 3072, 4096]      # -npp
    assert spec.workload.n_gen == [128]
    assert len({b.base_port for b in spec.backends.values()}) == 12

    for name, b in spec.backends.items():
        if b.type == "llamacpp":
            assert b.deployment == {"n_ctx": [32000]}                    # -c 32000
            assert b.extra_args == ["-fa", "on", "--load-mode", "mlock"]
            assert b.env["OMP_NUM_THREADS"] == "32" and "libomp.so.5" in b.env["LD_PRELOAD"]
            assert ("ZENDNNL_MATMUL_ALGO" in b.env) == ("zendnn" in name)
            # The copied binaries' RUNPATH points into sacsharm's live build tree; each build
            # must load its own libraries instead (see the comment on x-llamacpp-env).
            lib_dirs = b.env["LD_LIBRARY_PATH"].split(":")
            assert lib_dirs[0] == str(Path(b.server_bin).parent)
            assert any("zendnnl" in d for d in lib_dirs) == ("zendnn" in name)
            assert not any("/staff/sacsharm/" in d for d in lib_dirs)
        else:
            assert not b.deployment                                      # global 8192 applies
            assert b.env["VLLM_CPU_KVCACHE_SPACE"] == "90"
            assert "--enforce-eager" in b.extra_args
            # Qwen3.6's HF checkpoints are multimodal; only its variants skip the vision tower.
            assert ("--language-model-only" in b.extra_args) == name.startswith("qwen36-")

    # Both models get the same engine settings: each qwen36 variant matches its llama31 twin
    # in everything but the weights, the served name, the port and --language-model-only.
    for v in variants:
        llama, qwen = spec.backends[f"llama31-{v}"], spec.backends[f"qwen36-{v}"]
        assert (qwen.type, qwen.server_bin, qwen.env, qwen.deployment) == \
               (llama.type, llama.server_bin, llama.env, llama.deployment)
        assert [a for a in qwen.extra_args if a != "--language-model-only"] == llama.extra_args
    assert spec.deployment.n_ctx == [8192]                               # --max-model-len
    assert spec.deployment.batch == [4096] and spec.deployment.ubatch == [4096]
