#!/usr/bin/env python3
"""Write the Turin tuning sweeps: one knob at a time from each engine's baseline, at pp4096 + tg128.

    python scripts/gen_turin_tune_specs.py        # rewrites the eight spec files below

  sweep.turin-tune-llamacpp.yaml        stage 1: stock + ZenDNN llama.cpp, 8 bases, 92 deployments
  sweep.turin-tune-vllm.yaml            stage 1: ZenDNN vLLM, 4 bases, 52 deployments
  sweep.turin-tune-{llamacpp,vllm}-smoke.yaml
                                        every non-baseline knob once, on one base, at pp16/tg16
  sweep.turin-tune2-{llamacpp,vllm}[-smoke].yaml
                                        stage 2: the combined winners (`tuned`) vs stage 1's
                                        baseline, plus what stage 1 left open; see
                                        STAGE2_HEADER

Bases are {Llama 3.1 8B, Qwen3.6-35B-A3B} x {16-bit, 8-bit} weights, the same files as
sweep.turin-32c-8b.yaml. A variant is named <base>-<knob> and differs from <base>-base in
that one knob only. <base>-base-end is a second copy of the baseline, run after everything
else, to show drift during the run. The generated YAML is the thing to review; edit this
script, not the output, so the four files stay consistent.
"""

from __future__ import annotations

import copy
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
         smoke: bool, n_ctx: int = 8192, n_prompt: tuple[int, ...] = (4096,)) -> str:
    doc = {
        "name": name,
        "out_dir": f"out/{name}",
        "mode": "online",
        "cpu": CPU,
        "lb": {"kind": "client"},
        "backends": backends,
        "deployment": {"backend": order, "instances": [1], "n_parallel": [1],
                       "n_ctx": [n_ctx], "batch": [4096], "ubatch": [4096]},
        "workload": ({"n_prompt": [16], "n_gen": [16], "concurrency": [1], "reps": 1,
                      "warmup_fixed": 1} if smoke else
                     {"n_prompt": list(n_prompt), "n_gen": [128], "concurrency": [1],
                      "reps": 3, "warmup_fixed": 1}),
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


# ---------------------------------------------------------------- stage 2
#
# Stage 1 (turin-tune-*) varied one knob at a time. Stage 2 runs the combination of its
# winners as `tuned` -- the recommended command -- next to `ref` (stage 1's baseline), and
# probes what stage 1 left open. pp4096 + pp8192 + tg128, so both engines get 16384 context.

TUNED_UB = {("llama31", "bf16", "llamacpp-zendnn"): 1024}   # every other base: 512
STAGE2_UB = [256, 512, 1024, 2048]
STAGE2_BATCH = [1024, 2048, 4096, 8192]                    # vLLM --max-num-batched-tokens

VLLM_TUNED_ENV = {
    "LD_PRELOAD": VLLM_ENV["LD_PRELOAD"],
    "OMP_NUM_THREADS": "32",
    "VLLM_CPU_KVCACHE_SPACE": "8",          # ~2 GiB needed at 16384 context, batch 1
    "ZENDNNL_MATMUL_ALGO": "1",
    "ZENDNNL_MATMUL_WEIGHT_CACHE": "1",
    # ZENDNNL_LRU_CACHE_CAPACITY left unset: ZenDNN's default is unlimited, so all of
    # Qwen's expert weights stay reordered (1024 entries evicts them every request).
    "TORCHINDUCTOR_FREEZING": "1",
    "VLLM_USE_AOT_COMPILE": "0",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
}
VLLM_TUNED_ARGS = ["--dtype", "auto", "--kv-cache-dtype", "float16",
                   "--distributed-executor-backend", "mp"]


def _with(b: dict, *, env: dict | None = None, unset: tuple = (),
          deployment: dict | None = None) -> dict:
    v = copy.deepcopy(b)
    v["env"] = {k: val for k, val in v["env"].items() if k not in unset} | (env or {})
    v.pop("deployment", None)
    if deployment:
        v["deployment"] = deployment
    return v


def stage2_llamacpp() -> tuple[dict[str, dict], dict[str, dict]]:
    """(backends in run order, ends appended last) for both llama.cpp builds."""
    out: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for model in ("llama31", "qwen36"):
        for quant in ("bf16", "q8"):
            for build, (server_bin, env) in BUILDS.items():
                path, served = GGUF[(model, quant)]
                name = f"{model}-{build}-{quant}"
                ref = {"type": "llamacpp", "model": path, "served_model_name": served,
                       "server_bin": server_bin, "extra_args": list(LLAMACPP_ARGS),
                       "env": dict(env)}
                # --load-mode mlock only ever failed (ulimit -l 8 MB); none/dio/mmap measured
                # the same, so tuned uses the default.
                tuned = {**copy.deepcopy(ref), "extra_args": ["-fa", "on"]}
                ub = TUNED_UB.get((model, quant, build), 512)
                out[f"{name}-ref"] = ref
                out[f"{name}-tuned"] = _with(tuned, deployment={"ubatch": STAGE2_UB})
                if build == "llamacpp-zendnn" and model == "qwen36":
                    for algo in (3, 4, 5):
                        out[f"{name}-grp{algo}"] = _with(
                            tuned, env={"ZENDNNL_GRP_MATMUL_ALGO": str(algo)},
                            deployment={"ubatch": [ub]})
                ends[f"{name}-tuned-end"] = _with(tuned, deployment={"ubatch": [ub]})
    return _ports(out | ends, 9000), ends


def stage2_vllm() -> tuple[dict[str, dict], dict[str, dict]]:
    out: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for model in ("llama31", "qwen36"):
        for quant in ("bf16", "w8a8"):
            path, served = HF[(model, quant)]
            name = f"{model}-vllm-{quant}"
            tail = ["--language-model-only"] if model == "qwen36" else []
            ref = {"type": "vllm", "model": path, "served_model_name": served,
                   "server_bin": VLLM_BIN, "extra_args": VLLM_ARGS + tail,
                   "env": dict(VLLM_ENV)}
            tuned = {**copy.deepcopy(ref), "extra_args": VLLM_TUNED_ARGS + tail,
                     "env": dict(VLLM_TUNED_ENV)}
            out[f"{name}-ref"] = ref
            out[f"{name}-tuned"] = _with(tuned, deployment={"batch": STAGE2_BATCH})
            out[f"{name}-algo2"] = _with(tuned, env={"ZENDNNL_MATMUL_ALGO": "2"})
            out[f"{name}-algo-unset"] = _with(tuned, unset=("ZENDNNL_MATMUL_ALGO",))
            # 8 = auto_tuner (ZenDNN runtime_env.md: `auto` and 8 are the same setting).
            out[f"{name}-algo-auto"] = _with(tuned, env={"ZENDNNL_MATMUL_ALGO": "8"})
            if model == "qwen36":
                for algo in (3, 4, 5):
                    out[f"{name}-grp{algo}"] = _with(
                        tuned, env={"ZENDNNL_GRP_MATMUL_ALGO": str(algo)})
                out[f"{name}-lru1024"] = _with(
                    tuned, env={"ZENDNNL_LRU_CACHE_CAPACITY": "1024"})
                if quant == "bf16":
                    out[f"{name}-wcache2"] = _with(
                        tuned, env={"ZENDNNL_MATMUL_WEIGHT_CACHE": "2"})
            ends[f"{name}-tuned-end"] = _with(tuned)
    return _ports(out | ends, 9200), ends


def _ports(backends: dict[str, dict], first: int) -> dict[str, dict]:
    for i, b in enumerate(backends.values()):
        b["base_port"] = first + i
    return backends


STAGE2_HEADER = """\
# GENERATED by scripts/gen_turin_tune_specs.py -- edit that, not this file.
#
# Stage 2 of the Turin tuning. `<base>-tuned` is the recommended command, combining the
# stage-1 winners (sweep.turin-tune-*); `<base>-ref` is stage 1's baseline, so tuned-vs-ref
# is measured within one run. Other variants are one change from tuned. `<base>-tuned-end`
# repeats tuned after everything else, as a drift check. pp4096, pp8192 and tg128, 3 reps +
# 1 warm-up, concurrency 1, cores 160-191 / NUMA node 5; -c / --max-model-len 16384.
"""

STAGE2_LLAMACPP_HEADER = STAGE2_HEADER + """\
#
# llama.cpp (stock and ZenDNN builds):
#   ref         stage-1 baseline: -ub 4096, --load-mode mlock (which fails, ulimit -l 8 MB)
#   tuned       no --load-mode, swept over -ub 256/512/1024/2048 (-b 4096)
#   grp3/4/5    ZenDNN Qwen only: ZENDNNL_GRP_MATMUL_ALGO 3 (N-tile), 4 (multilevel CCD),
#               5 (per-expert), vs the default 0 (auto); at the tuned -ub
"""

STAGE2_VLLM_HEADER = STAGE2_HEADER + """\
#
# vLLM (ZenDNN / zentorch):
#   ref         stage-1 baseline: --enforce-eager, KV 90 GiB, LRU 1024, bf16 KV cache
#   tuned       torch.compile + inductor env, --kv-cache-dtype float16, KV 8 GiB, LRU unset;
#               swept over --max-num-batched-tokens 1024/2048/4096/8192
#   algo2       ZENDNNL_MATMUL_ALGO=2 (onednn_blocked)
#   algo-unset  ZENDNNL_MATMUL_ALGO unset (library default selection)
#   algo-auto   ZENDNNL_MATMUL_ALGO=8 (auto_tuner), now without the 1024-entry LRU cap
#   grp3/4/5    Qwen only: ZENDNNL_GRP_MATMUL_ALGO 3/4/5 vs the default 0 (auto)
#   lru1024     Qwen only: the stage-1 LRU cap, to isolate its effect within tuned
#   wcache2     Qwen BF16 only: ZENDNNL_MATMUL_WEIGHT_CACHE=2 (in-place reorder, less memory)
"""


def main() -> None:
    lc = llamacpp_backends()
    vl = vllm_backends()
    lc_all, vl_all = with_end_controls(lc), with_end_controls(vl)

    # Copies: the full specs share these dicts, and a smoke edit must not reach them.
    smoke_lc = {k: copy.deepcopy(v) for k, v in lc.items()
                if k.startswith("llama31-llamacpp-zendnn-q8-") and not k.endswith("-base")}
    smoke_vl = {k: copy.deepcopy(v) for k, v in vl.items()
                if (k.startswith("llama31-vllm-w8a8-") and not k.endswith("-base"))
                or k in ("qwen36-vllm-w8a8-compile", "qwen36-vllm-w8a8-kvf16")}
    for b in (*smoke_lc.values(), *smoke_vl.values()):
        b.pop("deployment", None)

    lc2, _ = stage2_llamacpp()
    vl2, _ = stage2_vllm()
    # Smoke: everything new at one deployment each -- every tuned base plus the one-off
    # variants of one base per engine; ref and the ends ran in stage 1 already.
    smoke_lc2 = {k: _with(v) for k, v in lc2.items()
                 if k.endswith("-tuned") or "-grp" in k}
    smoke_vl2 = {k: _with(v) for k, v in vl2.items()
                 if k.endswith("-tuned")
                 or (k.startswith("llama31-vllm-w8a8-") and "-algo" in k)
                 or (k.startswith("qwen36-") and ("-grp" in k or "-lru" in k or "-wcache" in k))}
    stage2 = dict(smoke=False, n_ctx=16384, n_prompt=(4096, 8192))

    files = {
        "sweep.turin-tune2-llamacpp.yaml":
            spec("turin-tune2-llamacpp", STAGE2_LLAMACPP_HEADER, lc2, list(lc2), **stage2),
        "sweep.turin-tune2-vllm.yaml":
            spec("turin-tune2-vllm", STAGE2_VLLM_HEADER, vl2, list(vl2), **stage2),
        "sweep.turin-tune2-llamacpp-smoke.yaml":
            spec("turin-tune2-llamacpp-smoke", STAGE2_LLAMACPP_HEADER, smoke_lc2,
                 list(smoke_lc2), smoke=True, n_ctx=16384),
        "sweep.turin-tune2-vllm-smoke.yaml":
            spec("turin-tune2-vllm-smoke", STAGE2_VLLM_HEADER, smoke_vl2, list(smoke_vl2),
                 smoke=True, n_ctx=16384),
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
