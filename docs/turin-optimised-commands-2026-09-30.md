# Turin: optimised server commands per configuration (boost off, 2026-09-30)

The best measured command for each of the twelve configurations on `turin-xcovoid0021-pod-5`
(cores 160-191, NUMA node 5, one server instance), with CPU boost off, which the infra team set
on purpose at the 2026-09-28 11:37 UTC reboot. Evidence is in `docs/turin-findings-2026-09-30.md`,
finding 9.

## How these were chosen

- **Baseline**: `sweep.turin-32c-8b-rerun.yaml` → `out/turin-32c-8b-rerun/` (09-30, boost off).
- **Candidate setups**:
  - new-command: `out/turin-32c-newcmd/`, with Qwen BF16 llama.cpp from
    `out/turin-32c-newcmd-qwen-bf16-rerun/`;
  - compile/`-ub 512`: `out/turin-32c-compile-ub512/`.
- **Rule**: each configuration gets whichever of baseline, new-command and compile/`-ub 512` was
  fastest for it.
  - Parts of different setups were **not** combined, because those mixes are unmeasured.
  - Differences under ~3% are within run-to-run noise; boost-off runs reproduce within 1-3%.

| configuration | setup | vs baseline: pp≤1024 / pp1536-4096 / tg128 |
|---|---|---|
| Llama llama.cpp ZenDNN BF16 | **A** baseline (`-ub 4096`) | 1.00 / 1.00 / 1.00 (new setups 0.96-0.97) |
| Llama llama.cpp ZenDNN Q8_0 | **B** new-command | ~1.00 / 1.02 / 0.99 (all setups tie) |
| Llama llama.cpp stock BF16 | **B** new-command | 1.00-1.07 / 1.13 / 1.00 |
| Llama llama.cpp stock Q8_0 | **B** new-command | ~1.00 / 1.06 / 0.99 |
| Qwen llama.cpp ZenDNN BF16 | **B** new-command | 1.00-1.08 / 1.18 / 0.98 |
| Qwen llama.cpp ZenDNN Q8_0 | **B** new-command | 1.00-1.10 / 1.29 / 0.99 |
| Qwen llama.cpp stock BF16 | **B** new-command | 1.00-1.03 / 1.18 / 0.97 |
| Qwen llama.cpp stock Q8_0 | **B** new-command | 1.00-1.09 / 1.24 / 0.98 |
| Llama vLLM BF16 | **C** vLLM compile | 1.06-1.14 / 1.07 / 1.04 |
| Llama vLLM W8A8 | **C** vLLM compile | 1.10-1.18 / 1.09 / 1.06 |
| Qwen vLLM BF16 | **C** vLLM compile, KV 32 GiB | 1.3-7.1 / 1.94 / 1.21 |
| Qwen vLLM W8A8 | **D** vLLM new-command | 1.2-3.2 / 1.75 / 1.25 |

## A: llama.cpp baseline (Llama ZenDNN BF16 only)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sohroy/lib/libomp.so.5 \
LD_LIBRARY_PATH=/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build_zendnn/bin:/proj/rdi/staff/sohroy/llama.cpp-sacsharm/zendnnl/lib \
KMP_AFFINITY=granularity=fine,compact,1,0 \
OMP_NUM_THREADS=32 \
ZENDNNL_MATMUL_ALGO=1 \
numactl --physcpubind=160-191 --membind=5 -- \
/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build_zendnn/bin/llama-server \
    -m /proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-BF16.gguf \
    --host 127.0.0.1 --port 8100 -t 32 -tb 32 -c 32000 -np 1 -b 4096 -ub 4096 \
    --metrics --alias llama3.1-8b-bf16 -fa on --load-mode mlock
```

## B: llama.cpp new-command (the other seven llama.cpp configurations)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sohroy/lib/libomp.so.5 \
LD_LIBRARY_PATH=<LIBS> \
KMP_AFFINITY=granularity=fine,compact,1,0 \
KMP_BLOCKTIME=1 \
KMP_TPAUSE=0 \
KMP_FORKJOIN_BARRIER_PATTERN=dist,dist \
KMP_PLAIN_BARRIER_PATTERN=dist,dist \
KMP_REDUCTION_BARRIER_PATTERN=dist,dist \
OMP_NUM_THREADS=32 \
OMP_DYNAMIC=FALSE \
OMP_WAIT_POLICY=ACTIVE \
ZENDNNL_MATMUL_ALGO=1 \
numactl --physcpubind=160-191 --membind=5 -- \
<BIN> \
    -m <GGUF> \
    --host 127.0.0.1 --port <PORT> -t 32 -tb 32 -c 32000 -np 1 -b 4096 -ub 512 \
    --metrics --alias <NAME> -fa on -ctk f16 -ctv f16
```

Leave out `ZENDNNL_MATMUL_ALGO=1` for the stock build.

| build | `<BIN>` | `<LIBS>` |
|---|---|---|
| ZenDNN | `/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build_zendnn/bin/llama-server` | `/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build_zendnn/bin:/proj/rdi/staff/sohroy/llama.cpp-sacsharm/zendnnl/lib` |
| stock | `/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build/bin/llama-server` | `/proj/rdi/staff/sohroy/llama.cpp-sacsharm/build/bin` |

| model | `<GGUF>` |
|---|---|
| Llama Q8_0 | `/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-Q8_0.gguf` |
| Llama BF16 (stock build only) | `/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-BF16.gguf` |
| Qwen BF16 | `/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-BF16.gguf` (our copy, not sacsharm's shared file) |
| Qwen Q8_0 | `/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-Q8_0.gguf` |

## C: vLLM compile (Llama BF16, Llama W8A8, Qwen BF16)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sacsharm/vllm/.venv/lib/libiomp5.so \
OMP_NUM_THREADS=32 \
VLLM_CPU_KVCACHE_SPACE=90 \
ZENDNNL_MATMUL_WEIGHT_CACHE=1 \
VLLM_CPU_OMP_THREADS_BIND=160-191 \
numactl --physcpubind=160-191 --membind=5 -- \
/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm serve <MODEL> \
    --host 127.0.0.1 --port <PORT> \
    --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 4096 \
    --served-model-name <NAME> \
    --dtype auto --kv-cache-dtype auto --distributed-executor-backend mp
```

- No `--enforce-eager`, `ZENDNNL_LRU_CACHE_CAPACITY`, `ZENDNNL_MATMUL_ALGO` or `--block-size`:
  library defaults.
- **Llama BF16**: `<MODEL>` = `/proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct`.
- **Llama W8A8**: `<MODEL>` = `/proj/rdi/staff/sohroy/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`.
- **Qwen BF16**:
  - `<MODEL>` = `/proj/rdi/staff/sacsharm/models/hf/Qwen3.6-35B-A3B`;
  - add `--language-model-only`;
  - use **`VLLM_CPU_KVCACHE_SPACE=32`**: with the default LRU cache, 90 is OOM-killed at the
    pod's 256 GiB limit; 32 peaked at 222 GB;
  - use `--max-model-len 16384` for prompts over 8K tokens.

## D: vLLM new-command (Qwen W8A8)

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/aigstaff/sacsharm/vllm/.venv/lib/libiomp5.so \
VLLM_CPU_KVCACHE_SPACE=32 \
VLLM_CPU_OMP_THREADS_BIND=160-190 \
VLLM_ZENTORCH_WEIGHT_PREPACK=1 \
OMP_NUM_THREADS=32 \
OMP_DYNAMIC=FALSE \
OMP_WAIT_POLICY=ACTIVE \
TORCHINDUCTOR_FREEZING=1 \
VLLM_USE_AOT_COMPILE=0 \
TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
numactl --physcpubind=160-191 --membind=5 -- \
/proj/aigstaff/sacsharm/vllm/.venv/bin/vllm serve /proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-w8a8-llmcompressor \
    --host 127.0.0.1 --port 8410 \
    --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 4096 \
    --served-model-name qwen3.6-35b-a3b-w8a8 \
    --dtype bfloat16 --kv-cache-dtype float16 --distributed-executor-backend mp \
    --language-model-only
```

`OMP_NUM_THREADS=32` is included because the harness added it in the measured run.

## Caveats

- **Per-configuration picks.** The winning pieces were never combined, for example weight
  prepacking with the Llama compile settings, or the `KMP_*` environment with `-ub 4096`. A
  combination might beat these, but it is unmeasured.
- **Llama Q8_0 ties.** ZenDNN and stock Q8_0 score the same on all setups (within 2%).
- **llama.cpp decode trade-off.** Setup B costs 1-3% decode on Qwen for +18-29% long-prompt
  prefill. If decode matters most, setup A is 1-3% faster at decode.
- **Qwen W8A8 beyond 4096 tokens.** Setup C was 2% faster than D there. D's mid-range gain is much
  larger, so D is still the pick.
- **Picks come from different runs.** A single run with exactly these twelve commands against the
  baseline would confirm them.
