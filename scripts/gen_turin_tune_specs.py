#!/usr/bin/env python3
"""Write the Turin tuning sweeps: one knob at a time from each engine's baseline, at pp4096 + tg128.

    python scripts/gen_turin_tune_specs.py        # rewrites the four spec files below

  sweep.turin-tune-llamacpp.yaml        stock + ZenDNN llama.cpp, 8 bases, 92 deployments
  sweep.turin-tune-vllm.yaml            ZenDNN vLLM, 4 bases, 52 deployments
  sweep.turin-tune-{llamacpp,vllm}-smoke.yaml
                                        every non-baseline knob once, on one base, at pp16/tg16

Bases are {Llama 3.1 8B, Qwen3.6-35B-A3B} x {16-bit, 8-bit} weights, the same files as
sweep.turin-32c-8b.yaml. A variant is named <base>-<knob> and differs from <base>-base in
that one knob only. <base>-base-end is a second copy of the baseline, run after everything
else, to show drift during the run. The generated YAML is the thing to review; edit this
script, not the output, so the four files stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

CPU = {"budget": "160-191", "smt": "exclude", "ccd_align": True, "membind": "5",
       "numa_policy": "membind"}

LLAMACPP = "/proj/rdi/staff/sohroy/llama.cpp-sacsharm"
VLLM_BIN = "/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm"
TCMALLOC = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# ---------------------------------------------------------------- llama.cpp

LLAMACPP_ENV = {
    "LD_PRELOAD": f"{TCMALLOC}:/proj/rdi/staff/sohroy/lib/libomp.so.5",
    "LD_LIBRARY_PATH": f"{LLAMACPP}/build/bin",
    "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "OMP_NUM_THREADS": "32",
}
LLAMACPP_ZENDNN_ENV = {
    **LLAMACPP_ENV,
    "LD_LIBRARY_PATH": f"{LLAMACPP}/build_zendnn/bin:{LLAMACPP}/zendnnl/lib",
    "ZENDNNL_MATMUL_ALGO": "1",
}
BUILDS = {  # name part -> (server_bin, env)
    "llamacpp-zendnn": (f"{LLAMACPP}/build_zendnn/bin/llama-server", LLAMACPP_ZENDNN_ENV),
    "llamacpp": (f"{LLAMACPP}/build/bin/llama-server", LLAMACPP_ENV),
}
GGUF = {  # (model, quant) -> (path, served name)
    ("llama31", "bf16"): ("/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-BF16.gguf",
                          "llama3.1-8b-bf16"),
    ("llama31", "q8"): ("/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-Q8_0.gguf",
                        "llama3.1-8b-q8_0"),
    ("qwen36", "bf16"): ("/proj/rdi/staff/sacsharm/models/gguf/Qwen3.6-35B-A3B-BF16.gguf",
                         "qwen3.6-35b-a3b-bf16"),
    ("qwen36", "q8"): ("/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-Q8_0.gguf",
                       "qwen3.6-35b-a3b-q8_0"),
}
LLAMACPP_ARGS = ["-fa", "on", "--load-mode", "mlock"]

# knob -> (extra args appended, args replaced {old: new}, env set, env unset, deployment
#          overrides, ZenDNN build only). The harness passes `-t N -tb N` before extra_args,
#          and llama.cpp keeps the last value, so an appended `-t 16` leaves `-tb 32`.
LLAMACPP_KNOBS = {
    "base":       ([], {}, {}, [], {"ubatch": [512, 1024, 2048, 4096]}, False),
    "ctx32k":     ([], {}, {}, [], {"n_ctx": [32000]}, False),
    "t16":        (["-t", "16"], {}, {}, [], {}, False),
    "t24":        (["-t", "24"], {}, {}, [], {}, False),
    "lm-none":    ([], {"mlock": "none"}, {}, [], {}, False),
    "lm-dio":     ([], {"mlock": "dio"}, {}, [], {}, False),
    "algo-unset": ([], {}, {}, ["ZENDNNL_MATMUL_ALGO"], {}, True),
    "algo2":      ([], {}, {"ZENDNNL_MATMUL_ALGO": "2"}, [], {}, True),
    "algo8":      ([], {}, {"ZENDNNL_MATMUL_ALGO": "8"}, [], {}, True),
}

# ---------------------------------------------------------------- vLLM

VLLM_ENV = {
    "LD_PRELOAD": f"{TCMALLOC}:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so",
    "OMP_NUM_THREADS": "32",
    "VLLM_CPU_KVCACHE_SPACE": "90",
    "ZENDNNL_MATMUL_ALGO": "1",
    "ZENDNNL_MATMUL_WEIGHT_CACHE": "1",
    "ZENDNNL_LRU_CACHE_CAPACITY": "1024",
}
VLLM_ARGS = ["--dtype", "auto", "--kv-cache-dtype", "auto", "--enforce-eager",
             "--distributed-executor-backend", "mp"]
HF = {
    ("llama31", "bf16"): ("/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct",
                          "llama3.1-8b-bf16"),
    ("llama31", "w8a8"): ("/proj/rdi/staff/sohroy/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
                          "llama3.1-8b-w8a8"),
    ("qwen36", "bf16"): ("/proj/rdi/staff/sacsharm/models/hf/Qwen3.6-35B-A3B",
                         "qwen3.6-35b-a3b-bf16"),
    ("qwen36", "w8a8"): ("/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-w8a8-llmcompressor",
                         "qwen3.6-35b-a3b-w8a8"),
}
VLLM_KNOBS = {
    "base":      ([], {}, {}, [], {}, False),
    # torch.compile (inductor) instead of eager; env from the zentorch vLLM README.
    # (compile and kvf16 edit the base args; see _vllm_args.)
    "compile":   ([], {},
                  {"TORCHINDUCTOR_FREEZING": "1", "VLLM_USE_AOT_COMPILE": "0",
                   "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}, [], {}, False),
    # Leave cores to the API server / scheduler; numactl still spans 160-191.
    "omp30":     ([], {}, {"VLLM_CPU_OMP_THREADS_BIND": "160-189", "OMP_NUM_THREADS": "30"},
                  [], {}, False),
    "omp31":     ([], {}, {"VLLM_CPU_OMP_THREADS_BIND": "160-190", "OMP_NUM_THREADS": "31"},
                  [], {}, False),
    "lru-unset": ([], {}, {}, ["ZENDNNL_LRU_CACHE_CAPACITY"], {}, False),
    "wcache2":   ([], {}, {"ZENDNNL_MATMUL_WEIGHT_CACHE": "2"}, [], {}, False),
    "kvspace8":  ([], {}, {"VLLM_CPU_KVCACHE_SPACE": "8"}, [], {}, False),
    "algo8":     ([], {}, {"ZENDNNL_MATMUL_ALGO": "8"}, [], {}, False),
    "noprefix":  (["--no-enable-prefix-caching"], {}, {}, [], {}, False),
    # Same KV storage type as llama.cpp's f16 cache. `--dtype auto` already resolves to
    # bfloat16 for these checkpoints (server log: dtype=torch.bfloat16), so that stays.
    "kvf16":     ([], {}, {}, [], {}, False),
    "bs32":      (["--block-size", "32"], {}, {}, [], {}, False),
    "bs64":      (["--block-size", "64"], {}, {}, [], {}, False),
}


def _vllm_args(model: str, knob: str) -> list[str]:
    args = list(VLLM_ARGS)
    if knob == "kvf16":
        args[args.index("--kv-cache-dtype") + 1] = "float16"
    elif knob == "compile":
        args.remove("--enforce-eager")
    args += VLLM_KNOBS[knob][0]
    if model == "qwen36":
        args.append("--language-model-only")  # text-only, like the GGUFs
    return args


def _llamacpp_args(knob: str) -> list[str]:
    extra, replace, *_ = LLAMACPP_KNOBS[knob]
    return [replace.get(a, a) for a in LLAMACPP_ARGS] + extra


def _env(base: dict, knobs: dict, knob: str) -> dict:
    _, _, set_, unset, _, _ = knobs[knob]
    env = {k: v for k, v in base.items() if k not in unset}
    env.update(set_)
    return env


def llamacpp_backends() -> dict[str, dict]:
    """Grouped by base so a model file stays in page cache across its knobs."""
    out: dict[str, dict] = {}
    port = 8500
    for model in ("llama31", "qwen36"):
        for quant in ("bf16", "q8"):
            for build, (server_bin, env) in BUILDS.items():
                path, served = GGUF[(model, quant)]
                for knob, (*_, overrides, zendnn_only) in LLAMACPP_KNOBS.items():
                    if zendnn_only and build != "llamacpp-zendnn":
                        continue
                    b = {"type": "llamacpp", "model": path, "served_model_name": served,
                         "server_bin": server_bin, "base_port": port,
                         "extra_args": _llamacpp_args(knob),
                         "env": _env(env, LLAMACPP_KNOBS, knob)}
                    if overrides:
                        b["deployment"] = overrides
                    out[f"{model}-{build}-{quant}-{knob}"] = b
                    port += 1
    return out


def vllm_backends() -> dict[str, dict]:
    out: dict[str, dict] = {}
    port = 8700
    for model in ("llama31", "qwen36"):
        for quant in ("bf16", "w8a8"):
            path, served = HF[(model, quant)]
            for knob in VLLM_KNOBS:
                out[f"{model}-vllm-{quant}-{knob}"] = {
                    "type": "vllm", "model": path, "served_model_name": served,
                    "server_bin": VLLM_BIN, "base_port": port,
                    "extra_args": _vllm_args(model, knob),
                    "env": _env(VLLM_ENV, VLLM_KNOBS, knob)}
                port += 1
    return out


def with_end_controls(backends: dict[str, dict]) -> dict[str, dict]:
    """Append <base>-base-end: the baseline again at ub4096 only, run last."""
    out = dict(backends)
    port = max(b["base_port"] for b in backends.values()) + 1
    for name, b in backends.items():
        if name.endswith("-base"):
            end = {k: v for k, v in b.items() if k != "deployment"}
            end["base_port"] = port
            out[f"{name}-end"] = end
            port += 1
    return out


def spec(name: str, header: str, backends: dict[str, dict], order: list[str], *,
         smoke: bool) -> str:
    doc = {
        "name": name,
        "out_dir": f"out/{name}",
        "mode": "online",
        "cpu": CPU,
        "lb": {"kind": "client"},
        "backends": backends,
        "deployment": {"backend": order, "instances": [1], "n_parallel": [1],
                       "n_ctx": [8192], "batch": [4096], "ubatch": [4096]},
        "workload": ({"n_prompt": [16], "n_gen": [16], "concurrency": [1], "reps": 1,
                      "warmup_fixed": 1} if smoke else
                     {"n_prompt": [4096], "n_gen": [128], "concurrency": [1], "reps": 3,
                      "warmup_fixed": 1}),
        "objective": {"metric": "tps_mean", "goal": "max", "src": "client"},
        "startup_timeout_s": 3600,
        "request_timeout_s": 900,
        "settle_s": 10,
        "continue_on_error": True,
        "core_sample_interval_s": 0.25,
    }
    body = yaml.dump(doc, Dumper=_Dumper, sort_keys=False, width=100)
    return header + "\n" + body


class _Dumper(yaml.SafeDumper):
    """Plain copies instead of &id anchors; lists on one line, mappings in block style."""

    def ignore_aliases(self, data) -> bool:
        return True


_Dumper.add_representer(
    list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
_Dumper.add_representer(
    dict, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v.items(), flow_style=False))


HEADER_COMMON = """\
# GENERATED by scripts/gen_turin_tune_specs.py -- edit that, not this file.
#
# One knob at a time, from each engine's baseline, at pp4096 (TTFT, prefill t/s) and tg128
# (decode t/s), 3 reps + 1 warm-up, concurrency 1, cores 160-191 / NUMA node 5 (pod-5).
# Knobs come from the 2026-09-28 tuning notes for this pod. Variants are <base>-<knob>;
# <base>-base-end repeats the baseline after everything else, as a drift check (the
# 2026-09-28 long-prompt run measured ~11% below the main sweep with identical settings).
# All bases use -c / --max-model-len 8192; <base>-ctx32k is the main sweep's -c 32000.
"""

LLAMACPP_HEADER = HEADER_COMMON + """\
#
# llama.cpp knobs (every base; algo-* on the ZenDNN build only):
#   base        main-sweep settings, -c 8192, swept over -ub 512/1024/2048/4096 (-b 4096)
#   ctx32k      -c 32000 (the main sweep's value), -ub 4096
#   t16, t24    decode threads -t 16 / 24, prefill threads -tb 32
#   lm-none     --load-mode none: weights read into the process's own (node-5) memory
#   lm-dio      --load-mode dio: same, via direct I/O (bypasses the page cache)
#   algo-unset  no ZENDNNL_MATMUL_ALGO (the user's offline baseline); algo2, algo8
# Not here: --poll / --cpu-strict need a GGML_OPENMP=OFF build (both builds link libgomp).
"""

VLLM_HEADER = HEADER_COMMON + """\
#
# vLLM knobs (ZenDNN / zentorch vLLM, sacsharm's venv):
#   base        main-sweep settings (--enforce-eager, 32 OMP threads, ALGO=1, cache env)
#   compile     no --enforce-eager (torch.compile/inductor) + TORCHINDUCTOR_FREEZING=1
#   omp30/31    OMP workers on 160-189 / 160-190, leaving cores to the API server
#   lru-unset   ZENDNNL_LRU_CACHE_CAPACITY unset (unlimited)
#   wcache2     ZENDNNL_MATMUL_WEIGHT_CACHE=2
#   kvspace8    VLLM_CPU_KVCACHE_SPACE=8
#   algo8       ZENDNNL_MATMUL_ALGO=8
#   noprefix    --no-enable-prefix-caching
#   kvf16       --kv-cache-dtype float16 (llama.cpp's KV type)
#   bs32, bs64  --block-size 32 / 64 (default 128)
# Not here: VLLM_ZENTORCH_WEIGHT_PREPACK already defaults to 1 in this vLLM build; the MoE
# ZENDNNL_GRP_MATMUL_* settings come after this sweep, on the winning Qwen config.
"""


def main() -> None:
    lc = llamacpp_backends()
    vl = vllm_backends()
    lc_all, vl_all = with_end_controls(lc), with_end_controls(vl)

    smoke_lc = {k: v for k, v in lc.items()
                if k.startswith("llama31-llamacpp-zendnn-q8-") and not k.endswith("-base")}
    smoke_vl = {k: v for k, v in vl.items()
                if (k.startswith("llama31-vllm-w8a8-") and not k.endswith("-base"))
                or k in ("qwen36-vllm-w8a8-compile", "qwen36-vllm-w8a8-kvf16")}
    for b in (*smoke_lc.values(), *smoke_vl.values()):
        b.pop("deployment", None)

    files = {
        "sweep.turin-tune-llamacpp.yaml":
            spec("turin-tune-llamacpp", LLAMACPP_HEADER, lc_all, list(lc_all), smoke=False),
        "sweep.turin-tune-vllm.yaml":
            spec("turin-tune-vllm", VLLM_HEADER, vl_all, list(vl_all), smoke=False),
        "sweep.turin-tune-llamacpp-smoke.yaml":
            spec("turin-tune-llamacpp-smoke", LLAMACPP_HEADER, smoke_lc, list(smoke_lc),
                 smoke=True),
        "sweep.turin-tune-vllm-smoke.yaml":
            spec("turin-tune-vllm-smoke", VLLM_HEADER, smoke_vl, list(smoke_vl), smoke=True),
    }
    for name, text in files.items():
        (REPO / name).write_text(text)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
