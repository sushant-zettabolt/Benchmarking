# Turin runs, 2026-09-29/30: observations, findings and hypotheses

All runs on `turin-xcovoid0021-pod-5`, cores 160-191 (NUMA node 5), memory bound to node 5,
one server instance, concurrency 1, 3 reps + 1 warm-up. Numbers are client-side `tps_mean`
(t/s) from each run's `report.csv`. Earlier runs are summarised in `docs/turin-status.md`.

## Runs

| run | spec | when (UTC) | result | output |
|---|---|---|---|---|
| New-command smoke | `sweep.turin-newcmd-smoke.yaml` | 09-29 20:33-20:57 | 30/30 ok | `out/turin-newcmd-smoke/` |
| New-command full | `sweep.turin-32c-8b-newcmd.yaml` | 09-29 21:01-23:04 | 264/264 ok; `qwen36-llamacpp-zendnn-bf16` invalid (remote pages, below) | `out/turin-32c-newcmd/` |
| Qwen BF16 llama.cpp rerun | `sweep.turin-32c-newcmd-qwen-bf16-rerun.yaml` | 09-30 01:04-01:38 | 50/50 ok | `out/turin-32c-newcmd-qwen-bf16-rerun/` |
| Compile/`-ub 512` smoke | `sweep.turin-compile-ub512-smoke.yaml` | 09-30 01:38-02:05 | 30/30 ok | `out/turin-compile-ub512-smoke/` |
| Compile/`-ub 512` full | `sweep.turin-32c-8b-compile-ub512.yaml` | 09-30 02:05-04:06 | 264/264 ok, all checks clean | `out/turin-32c-compile-ub512/` |

Reference runs used for comparison:

- **Baseline (current)**: `sweep.turin-32c-8b-rerun.yaml` → `out/turin-32c-8b-rerun/`,
  09-30 05:30-06:33 UTC, boost off (finding 9). Use this one from now on.
- **Old baseline**: `sweep.turin-32c-8b.yaml` → `out/turin-32c-llama31-qwen36/`, 09-28 04:27-05:17
  UTC, **before the host reboot**, with boost on (see finding 1).
- **Post-reboot baseline**: the `<variant>-base` rows of `out/turin-tune-llamacpp/` and
  `out/turin-tune-vllm/` (09-28 14:33-20:49 UTC). Identical commands to the baseline, pp4096 and
  tg128 only.

### What each spec runs

- **New-command** (`sweep.turin-32c-8b-newcmd.yaml`): the user's 2026-09-29 command patterns.
  - llama.cpp: `KMP_BLOCKTIME=1 KMP_TPAUSE=0 KMP_*_BARRIER_PATTERN=dist,dist OMP_DYNAMIC=FALSE
    OMP_WAIT_POLICY=ACTIVE`, tcmalloc + libomp preload, `-c 32000 -np 1 -b 4096 -ub 512 -fa on
    -ctk f16 -ctv f16`, no `--load-mode`.
  - vLLM: `VLLM_CPU_KVCACHE_SPACE=32 VLLM_CPU_OMP_THREADS_BIND=160-190 VLLM_ZENTORCH_WEIGHT_PREPACK=1`,
    torch.compile, `--max-num-seqs 16 --dtype bfloat16 --kv-cache-dtype float16`.
- **Compile/`-ub 512`** (`sweep.turin-32c-8b-compile-ub512.yaml`): the baseline's commands with
  - vLLM: no `--enforce-eager`, `ZENDNNL_LRU_CACHE_CAPACITY` and `ZENDNNL_MATMUL_ALGO` unset, default
    block size; `VLLM_CPU_KVCACHE_SPACE=32` for `qwen36-vllm-bf16` only (90 elsewhere);
  - llama.cpp: `-ub 512` instead of 4096.
- Both give the Qwen variants the long-context prompt list (pp1-pp12288, `--max-model-len 16384`
  on vLLM), via the per-backend `workload:` override added in commit `7b9e022`, and (since
  `bb45873`) load our own copy of the Qwen BF16 GGUF.

## Findings

### 1. The "regression" against the baseline is the host losing CPU boost (confirmed cause, indirect proof)

**Boost off is intentional** (user, 2026-09-30): the infra team disabled it for run-to-run
consistency. The new baseline is a rerun of `sweep.turin-32c-8b.yaml` under these conditions:
`sweep.turin-32c-8b-rerun.yaml` → `out/turin-32c-8b-rerun/` (started 2026-09-30 05:30 UTC). It
differs from the original spec only in loading our local Qwen BF16 GGUF copy (finding 3). A
root-level APERF/MPERF sampler logged ~2710 MHz on 160-191 during its measured trials.

- The host was rebooted at **2026-09-28 11:37:53 UTC** and came back with
  `/sys/devices/system/cpu/cpufreq/boost = 0` (driver `amd-pstate-epp`, governor and EPP
  `performance`, `scaling_max_freq` 4.12 GHz).
- Measured 2026-09-30 with APERF/MPERF (MSR 0xE8/0xE7, readable as root in the pod) under an
  all-core load on 160-191: **2733 MHz**, ratio 1.01 against the 2701 MHz reference, i.e. the
  EPYC 9755's 2.7 GHz base clock, no boost.
- The baseline ran **before** the reboot; every later run ran after it.
- The tuning sweep ran the baseline's exact commands after the reboot: **0.67-0.90x at pp4096,
  0.95-1.01x at tg128**, with `-base` and `-base-end` agreeing within ~1%. Same binaries, same
  models, same flags, so the drop is the host, not the settings.
- Compute-bound prefill scales with clock; memory-bound decode does not. A ~0.78x prefill ratio
  implies the baseline ran at ~3.4-3.5 GHz all-core, which is plausible for this part with boost.
- **Not proven directly** for the old run: no clock data was recorded on 09-28. The harness's
  new `MHz` column reads a constant 2102 on this host, so it cannot show this either.

### 2. On equal clocks, the new settings help

New compile/`-ub 512` run ÷ post-reboot baseline (`<variant>-base`), both with boost off:

| variant | pp4096 | tg128 |
|---|---|---|
| Llama llama.cpp stock BF16 / Q8_0 | 1.08 / 1.09 | 0.98 / 0.98 |
| Llama llama.cpp ZenDNN BF16 / Q8_0 | 0.98 / 1.28 | 0.98 / 0.98 |
| Llama vLLM BF16 / W8A8 | 1.06 / 1.08 | 1.03 / 1.03 |
| Qwen llama.cpp stock BF16* / Q8_0 | 1.15 / 1.17 | 0.98 / 0.98 |
| Qwen llama.cpp ZenDNN BF16* / Q8_0 | 1.14 / 1.25 | 0.98 / 0.97 |
| Qwen vLLM BF16 / W8A8 | 1.48 / 1.42 | 1.18 / 1.18 |

\* The tuning run's Qwen BF16 llama.cpp variants read sacsharm's shared GGUF, whose page cache
may have been remote (finding 3), so these two ratios may overstate the gain.

- vLLM torch.compile (no `--enforce-eager`): large gain on Qwen, small on Llama. Against the
  pre-reboot baseline Qwen vLLM is still 2-6x faster at pp32-pp512 (the eager runs had a deep dip
  there) and 1.3x at pp4096, despite the lower clock.
- llama.cpp `-ub 512`: +8-28% at pp4096, most for ZenDNN Q8_0. For prompts <= 512 tokens `-ub`
  makes no difference (one micro-batch either way).
- llama.cpp decode is 2-3% lower than the post-reboot baseline.

### 3. mmap'd GGUFs can run from another pod's NUMA node

- In the new-command full run, `qwen36-llamacpp-zendnn-bf16` ran at **0.4-0.6x** on every test
  (tg128 5.9 t/s vs ~15), with all 32 cores busy, no I/O wait and no foreign load.
- Its GGUF was sacsharm's shared file. It loaded in **2 s** (already in the host page cache)
  instead of ~80 s. A probe (mincore + move_pages on resident pages) found **~70% of that file's
  page cache on NUMA node 1**, another pod's node.
- llama.cpp mmaps weights; `numactl --membind=5` only governs pages the process allocates, not
  page cache that already exists. `qwen36-llamacpp-bf16` (same file, same run) loaded cold (77 s),
  got node-5 pages and was fine.
- Fix applied: copied the file to `/proj/rdi/staff/sohroy/models/Qwen3.6-35B-A3B-BF16.gguf` with
  `dd iflag=direct oflag=direct` (sha256 `d748123e9f300489e895aeb4e3f3dbc97a97cdd42c96eeecd306e7072703d685`,
  identical). The rerun on the copy loaded in 80 s and ran normally.
- Only llama.cpp on shared files is exposed. Our own GGUF copies are only cached on node 5; vLLM
  copies weights into memory it allocates itself. In the compile/`-ub 512` run the new
  `numa_maps` check (finding 7) reported 99.9-100% of every server's memory on node 5.

### 4. Engine and weight comparisons (compile/`-ub 512` run, boost off)

- **Winners.** Llama: llama.cpp Q8_0 up to pp8 and tg128, vLLM W8A8 from pp16 (peak 1015 t/s at
  pp768). Qwen: llama.cpp Q8_0 up to pp4 and tg128, vLLM BF16 pp8-pp512, vLLM W8A8 from pp768.
- **ZenDNN vs stock llama.cpp.** Decode equal everywhere.

  | | up to pp128 | pp256 and up |
  |---|---|---|
  | Llama BF16 | equal | 1.30-1.39x |
  | Llama Q8_0 | equal | ~3x |
  | Qwen BF16 | equal | 1.02-1.04x (confirmed by the rerun: 1.03-1.04x) |
  | Qwen Q8_0 | equal | 1.16-1.24x |

  So ZenDNN does accelerate Q8_0, but only once prompts reach ~256 tokens.
- **16-bit vs 8-bit.**
  - Decode: 8-bit is 1.8x on Llama (both engines), 1.5x on Qwen llama.cpp, only 1.06x on Qwen vLLM.
  - vLLM W8A8 prefill is 1.4-1.7x BF16 on Llama from pp64. On Qwen it is 1.07-1.27x from pp768 but
    slower below that.
  - Stock llama.cpp Q8_0 prefill is slower than BF16 from pp32 on Llama and from pp256 on Qwen.
    On Llama, llama.cpp Q8_0 is flat at ~200 t/s between pp32 and pp128 on both builds.

### 5. Qwen vLLM W8A8 dips at small prompts (reproducible, cause unknown)

pp8-pp32 run at 14-72 t/s with ~0.4-0.5 s time to first token, versus 58-134 t/s for BF16. Seen in
the new-command smoke and full runs and in the compile/`-ub 512` smoke and full runs.

**Hypothesis:** the int8 MoE expert path (`ZenCPUExpertsInt8`) is inefficient at small M. The
new-command settings (weight prepack, `--max-num-seqs 16`) were 10-25% better than
compile/`-ub 512` at pp32-pp512.

### 6. The new-command settings vs compile/`-ub 512` (different days, same boost-off state)

Median ratio, compile/`-ub 512` ÷ new-command, where below 1 means the new-command run was faster:

- **Llama llama.cpp**: equal (0.94-1.02).
- **Qwen llama.cpp**: 0.91-0.95 from pp32. **Hypothesis:** the `KMP_*` barrier/blocktime and
  `OMP_WAIT_POLICY=ACTIVE` environment helps the MoE model's many small parallel regions.
- **Llama vLLM**: compile/`-ub 512` better at long prompts (W8A8 1.12x).
- **Qwen vLLM W8A8**: new-command better at pp32-pp512 (0.74-0.92).

Too many knobs differ between the two setups (prepack, max-num-seqs, KV size, 31 vs 32 threads,
dtype flags) to attribute any of these to one knob; that needs a single-run A/B.

### 7. Harness state and one bug in uncommitted code

- **The compile/`-ub 512` full run is not reproducible from committed code.** Another session's
  uncommitted changes landed at 01:55 UTC 09-30, after its smoke started and before the full run
  started (02:05), so the full run used them. They add:
  - `deploy.verify_memory_placement` (per-node resident memory from `numa_maps`);
  - a per-trial `MHz` column (reads a constant 2102 on this host, so it is not informative);
  - an optional `workload.prompt_text`, unset here, so prompts were the usual random ids.

  Specs are committed (`7b9e022`, `bb45873`); the code changes are not.
- **Bug** in that uncommitted `execute.py`: the per-row "memory not on its bound node" warning is
  under `elif local and min(local) < ...` directly after `if local:`, so it can never fire. The
  `events.jsonl` warning path works.

### 8. Other observations

- **The "only 1 core busy" view** during a run was server startup (weight load plus vLLM
  compile), which is largely single-threaded. During measured trials every deployment used
  25-32 cores (vLLM 31 when bound to 160-190), per `cores/<deployment>.csv`.
- **Qwen BF16 on vLLM** with LRU unset:
  - fits at a 32 GiB KV cache (peak 222 GB resident, under the pod's 256 GiB cgroup limit);
  - OOM-killed at 90 GiB in the 09-28 tuning run.
- **Qwen W8A8 on vLLM** at a 90 GiB KV cache: 191 GB resident.

### 9. Against the new baseline (`out/turin-32c-8b-rerun/`, 2026-09-30 05:30-06:33 UTC)

The rerun finished with 456/456 rows ok and no warnings. Pinning was verified and memory was
99.9-100% on node 5 for all 12 deployments. Clock during measured trials: mean 2727 MHz (range
2680-2737; `clock-aperf-mperf.csv` in the run directory). It matches the 09-28 tuning `-base`
rows within 1-3% at pp4096 and tg128 (the exception is Llama ZenDNN Q8_0 pp4096, 0.81 on 09-28),
so boost-off runs reproduce across days.

Median ratio against the new baseline per prompt-length group. Qwen pp5120-pp12288 have no
baseline, because the spec stops at pp4096.

| variant | old 09-28, boost on: pp256-4096 / tg128 | compile/`-ub 512`: pp1536-4096 / tg128 | new-command: pp1536-4096 / tg128 |
|---|---|---|---|
| Llama llama.cpp ZenDNN BF16 | 1.23-1.26 / 1.01 | 0.96 / 1.00 | 0.97 / 1.00 |
| Llama llama.cpp ZenDNN Q8_0 | 1.15-1.19 / 1.02 | 1.01 / 1.00 | 1.02 / 0.99 |
| Llama llama.cpp stock BF16 | 1.22-1.29 / 1.01 | 1.06 / 1.00 | 1.13 / 1.00 |
| Llama llama.cpp stock Q8_0 | 1.32-1.34 / 1.02 | 1.05 / 1.00 | 1.06 / 0.99 |
| Llama vLLM BF16 | 1.23-1.25 / 1.01 | 1.07 / 1.04 | 0.99 / 1.03 |
| Llama vLLM W8A8 | 1.19-1.21 / 1.06 | 1.09 / 1.06 | 1.00 / 1.06 |
| Qwen llama.cpp ZenDNN BF16 | 1.19-1.29 / 1.06 | 1.12 / 1.00 | 1.18 / 0.98 |
| Qwen llama.cpp ZenDNN Q8_0 | 1.26-1.31 / 1.10 | 1.18 / 1.01 | 1.29 / 0.99 |
| Qwen llama.cpp stock BF16 | 1.14-1.26 / 1.06 | 1.12 / 1.00 | 1.18 / 0.97 |
| Qwen llama.cpp stock Q8_0 | 1.27-1.33 / 1.09 | 1.13 / 1.01 | 1.24 / 0.98 |
| Qwen vLLM BF16 | 1.10-1.11 / 1.12 | 1.94 / 1.21 | 1.89 / 1.16 |
| Qwen vLLM W8A8 | 1.11-1.12 / 1.20 | 1.75 / 1.24 | 1.75 / 1.25 |

- **Boost** was worth 15-34% on llama.cpp and Llama vLLM prefill from pp256, 10-12% on Qwen vLLM,
  and 1-20% on decode.
- **llama.cpp, both new setups:** equal to the baseline up to pp1024, where `-ub 512` changes
  little.
  - The new-command setup (`-ub 512` plus the `KMP_*`/`OMP_WAIT_POLICY` environment) gives the
    most at long prompts: Qwen +18-29%, Llama stock BF16 +13%.
  - Compile/`-ub 512` (`-ub 512` alone) gives Qwen +12-18%.
  - So on Qwen the `KMP_*` environment adds roughly 5-10% on top of `-ub 512`.
  - Llama ZenDNN BF16 is 3-4% lower with either setup.
- **vLLM:** compile is 1.5-7x on Qwen prefill (largest at pp32-pp1024) and +16-25% on Qwen
  decode.
  - On Llama the compile/`-ub 512` settings give +6-18%, while the new-command settings are flat
    at long prompts (W8A8 0.86-1.00 at pp256-1024).
  - The new-command settings are better for Qwen W8A8 at pp32-pp1024 (2.6-3.2x vs 2.0-2.6x).

## Open questions and next steps

1. **Use `out/turin-32c-8b-rerun/` as the baseline** from now on, not the 09-28 boost-on sweep.
2. **Check boost is still off** (`cat /sys/devices/system/cpu/cpufreq/boost` = 0) before
   comparing runs, in case the infra setting changes.
3. **Run a single-run A/B** of the baseline, new-command and compile/`-ub 512` settings per
   variant (pp128/pp512/pp4096/tg128). That separates setting effects from drift, which has been
   10-25% between runs.
4. **Make the clock sampler useful:** read APERF/MPERF (as `turbostat` does) instead of
   `cpuinfo_avg_freq`, so every row records the real clock.
5. **Check for remote page cache at launch:** any llama.cpp run on a file other pods may read
   should either use a private copy or be checked with `numa_maps` (now automatic, finding 7)
   before its numbers are used.
6. **Qwen W8A8 small-prompt dip:** profile pp8-pp32 on vLLM W8A8, or compare with prepack on and
   off in one run.
