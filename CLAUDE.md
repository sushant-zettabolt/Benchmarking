# llmbench — instructions for the agent on the Turin machine

This repo is `llmbench`, a benchmark harness for llama.cpp and vLLM servers. `llmbench sweep`
starts the servers itself, pins them to cores, runs a grid of workloads against them, tears
them down, and ranks the configurations. Read `docs/sweep.md` before changing anything in
`llmbench/suite/`, and `docs/contract.md` before changing anything that measures.

The code was written on a Windows machine that cannot run it. **Nothing in the latest changes
has been executed.** They were only syntax-checked. You are the first to run them, so treat
every test failure as a real finding, not noise.

## Your task

Run the sweep defined in `sweep.turin-32c-8b.yaml` and report the results.

It compares six server configurations of Llama 3.1 8B, each a single instance on 32 cores
(logical CPUs 192-223, memory on NUMA node 6):

**Expanded 2026-09-28 (user request):** the spec now runs the same six configurations for
**both** Llama 3.1 8B and Qwen3.6-35B-A3B, so 12 variants named `llama31-<name>` and
`qwen36-<name>` (e.g. `qwen36-llamacpp-zendnn-q8`), 228 trials, `out_dir:
out/turin-32c-llama31-qwen36`. The file keeps its old name. The Qwen weights and their provenance
are in the spec header; the Qwen vLLM variants add `--language-model-only`.

| backend name (prefix `llama31-` / `qwen36-`) | engine | weights |
|---|---|---|
| `llamacpp-zendnn-bf16` | llama.cpp, ZenDNN build | BF16 GGUF |
| `llamacpp-zendnn-q8` | llama.cpp, ZenDNN build | `~/models/gguf/Llama-3.1-8B-Instruct-Q8_0.gguf` |
| `llamacpp-bf16` | llama.cpp, stock build | BF16 GGUF |
| `llamacpp-q8` | llama.cpp, stock build | same Q8_0 GGUF |
| `vllm-bf16` | vLLM | `/tmp/models/Meta-Llama-3.1-8B-Instruct` |
| `vllm-w8a8` | vLLM | `/tmp/models/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` |

Each configuration runs these tests with 3 reps and 1 warm-up, at concurrency 1:

- 18 prefill tests: pp1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048,
  3072, 4096 (the `-npp` list of the verified reference below)
- one decode test: tg128

That is 114 trials per model, 228 in all. The settings mirror the user's earlier offline runs, which the spec file's
header comment quotes in full: `llama-batched-bench` and `vllm bench latency`, with the same
numactl binding, `LD_PRELOAD`, OpenMP and `ZENDNNL_*` environment. Do not "tidy" those
settings. They are the point of comparison.

## Rules on this machine

- **Never kill a process by name** (`pkill`, `killall`, `kill $(pgrep ...)`). Other people's
  servers may be running. To stop servers a sweep left behind, run
  `llmbench sweep cleanup <out_dir>`. It only signals processes recorded in that run's
  `pids.json` whose PID and kernel start time both match.
- **Do not change the ports or cores of anything you did not start.** If a port is busy, pick
  another `base_port` in the spec.
- **Do not edit tests just to make them pass.** If a test fails, work out whether the code or
  the test is wrong, fix that, and say which in your report.
- **Ask the user before** converting or downloading models, installing system packages,
  deleting anything under `out/`, or changing what is measured (the prompt list, reps, core
  budget, server flags).

## Step 1 — install and run the tests

```bash
git pull
python -m venv .venv && . .venv/bin/activate      # or reuse an existing venv
pip install -e '.[test]'
python -m pytest -q
```

Everything should pass on Linux. These are new and have never run:

- `tests/test_suite_reliability.py`: signal handling, resume, the PID ledger, `cleanup`,
  and the ranking excluding contaminated rows. Some tests start real `sh`/`sleep` processes
  and read `/proc`.
- `tests/test_suite_variants.py`: named backend variants and per-backend overrides. It also
  checks that `sweep.turin-32c-8b.yaml` still matches the reference settings.

If something fails, fix it before going further. A bug in teardown or the PID ledger can
leave servers holding cores for the rest of the night.

## Step 2 — fill in the spec

`grep -n REPLACE sweep.turin-32c-8b.yaml` lists what is missing:

1. **The llama.cpp checkout.** The ZenDNN server is `<checkout>/build_zendnn/bin/llama-server`
   (the reference ran `./build_zendnn/bin/llama-batched-bench`). The stock server is assumed to
   be `<checkout>/build/bin/llama-server`. Find them, for example with
   `ls -d ~/*/build_zendnn/bin ~/*/*/build_zendnn/bin 2>/dev/null`. If there is no non-ZenDNN
   build, ask the user; do not build one unasked.
2. **The BF16 GGUF.** The user says one exists. Search with
   `find ~ /tmp -iname '*llama*3.1*8b*bf16*.gguf' 2>/dev/null`. llama.cpp cannot load the HF
   directory `/tmp/models/Meta-Llama-3.1-8B-Instruct`; only vLLM uses that. If you find none,
   or more than one, ask.
3. **The vLLM binary.** It is set to `/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm`, inferred
   from the reference's libiomp5 path. Check that it exists.

Then check the flags the reference used that nobody has confirmed `llama-server` accepts:

```bash
<checkout>/build_zendnn/bin/llama-server --help | grep -E -- '--load-mode|-fa|--flash-attn'
<checkout>/build/bin/llama-server        --help | grep -E -- '-fa|--flash-attn'
```

- **`--load-mode mlock`** is on all four llama.cpp variants (`x-llamacpp-args`). Both
  `llama-server` builds on pod-5 list `-lm, --load-mode MODE` (auto|none|mmap|mlock|mmap+mlock).
- **`-fa on`**: some builds want `-fa` or `--flash-attn on`. Use whatever that build's
  `--help` shows.

Also confirm every file in the spec exists: both GGUFs, both HF directories, and both
`LD_PRELOAD` libraries.

### Findings for turin-xcovoid0021-pod-5 / pod-7 (this install)

These pods only expose two of the eight NUMA nodes: pod-5 is hard cpuset-restricted to cores
160-191 (NUMA node 5, confirmed via `taskset -pc $$` and `numactl -H` -- `numactl -H` shows all
8 nodes but the process's actual affinity mask does not span them), pod-7 to 224-255 (node 7).
Node 6 / cores 192-223 from the original spec is not reachable from either pod. Use pod-5's
range and `membind: "5"` unless told otherwise; re-check `numactl -H` and `taskset -pc $$`
before assuming these values still apply on a different pod.

1. **llama.cpp checkout.** No checkout was found in `~` or the readable staff dirs searched at
   first pass. Built one at `/proj/rdi/staff/sohroy/llama.cpp` from `ggml-org/llama.cpp`
   master: stock via `cmake -B build -DCMAKE_BUILD_TYPE=Release`, ZenDNN via
   `cmake -B build_zendnn -DGGML_ZENDNN=ON -DCMAKE_BUILD_TYPE=Release` (this auto-downloads and
   builds ZenDNN plus its deps -- oneDNN, libxsmm, AOCL DLP/utils -- and takes 30-45 minutes,
   not the 5-10 the upstream doc claims). A second, independently-built copy already existed,
   world-readable, at `/proj/rdi/staff/sacsharm/llama.cpp/{build_release,build_zendnn}`, and
   was copied to `/proj/rdi/staff/sohroy/llama.cpp-sacsharm/{build,build_zendnn}` for
   comparison -- both perform identically. sacsharm's copied binaries resolve some shared libs
   via rpath back to `/proj/rdi/staff/sacsharm/...`; keep that tree in place if using the copy.
2. **BF16 / Q8_0 GGUF and HF dir.** Found on **pod-7's local `/tmp/models/`** (container-local
   storage, not NFS-shared -- invisible from pod-5). Copied to shared storage at
   `/proj/rdi/staff/sohroy/models/`: `Llama-3.1-8B-Instruct-BF16.gguf`,
   `Llama-3.1-8B-Instruct-Q8_0.gguf`, `Llama-3.1-8B-Instruct/` (HF dir -- note: no `Meta-`
   prefix, unlike the spec's assumed `Meta-Llama-3.1-8B-Instruct`), and
   `Meta-Llama-3.1-8B-Instruct-quantized.w8a8/`.
3. **vLLM binary.** `/proj/rdi/staff/sacsharm/vllm/.venv/bin/vllm` exists and runs
   (`vllm --help` succeeds).
4. **`LD_PRELOAD` libraries.** `/usr/lib/x86_64-linux-gnu/libomp.so.5` does **not** exist on
   this host. The `libomp5` apt package is an empty transitional package with no library in
   it -- the real file ships in `libomp5-18` (`apt-get download libomp5-18` works unprivileged,
   no root needed; extract with `dpkg-deb -x` and take `usr/lib/llvm-18/lib/libomp.so.5`,
   resolving the symlink with `cp -L`). A copy now lives at
   `/proj/rdi/staff/sohroy/lib/libomp.so.5` -- confirmed via `nm -D` that it exports the
   `GOMP_*` compat symbols needed to intercept `libgomp` through `LD_PRELOAD`. Without it, both
   llama.cpp builds here link GNU `libgomp.so.1` (check with `ldd <bin> | grep omp`), which
   **silently ignores `KMP_AFFINITY`** -- confirm which OpenMP runtime a build actually uses
   before relying on `KMP_AFFINITY` for pinning; preload the copied `libomp.so.5` if you need
   it to actually take effect.
5. **Verified offline reference (user, 2026-09-28).** This ran without problems on pod-5 and is
   the llama.cpp reference the sweep mirrors (`/proj/rdi/staff/sohroy/llama.cpp-sacsharm` is a
   symlink to `/proj/aigstaff/sohroy/llama.cpp-sacsharm`):
   ```bash
   export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/proj/rdi/staff/sohroy/lib/libomp.so.5"
   export KMP_AFFINITY="granularity=fine,compact,1,0"
   export OMP_NUM_THREADS=32
   # ZENDNNL_MATMUL_ALGO left unset: the user's baseline, to compare against ALGO=1
   numactl --physcpubind=160-191 --membind=5 \
     /proj/rdi/staff/sohroy/llama.cpp-sacsharm/build_zendnn/bin/llama-batched-bench \
     -m /proj/rdi/staff/sohroy/models/Llama-3.1-8B-Instruct-Q8_0.gguf \
     -c 32000 -b 4096 -ub 4096 -t 32 -fa on --load-mode mlock \
     -npp 1,2,4,8,16,32,64,96,128,256,384,512,768,1024,1536,2048,3072,4096 -ntg 1 -npl 1,1
   ```
   The user says this whole combination is what makes it work, the `libomp.so.5` preload in
   particular (it takes over libgomp's `GOMP_*` entry points, so `KMP_AFFINITY` applies).
   Spec changes that followed: the llama.cpp `LD_PRELOAD` is tcmalloc + the copied
   `libomp.so.5` for both builds; `--load-mode mlock` (was `none`, ZenDNN-only) is on all
   llama.cpp variants; the prompt list gained 384, 768, 1536; the
   `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT` debugging env was removed. The ZenDNN variants keep
   `ZENDNNL_MATMUL_ALGO=1` from the original reference; the user confirmed the same command
   also works with `ZENDNNL_MATMUL_ALGO=1` set. `ulimit -l` for sohroy in the pod is 8192 KB, so mlock of a
   multi-GB model probably fails with a warning and falls back -- check the server log.
6. **Pod access from the workstation:** `/usr/local/bin/kubectl exec turin-xcovoid0021-pod-5
   -n zendnn -- runuser -l sohroy -c '<cmd>'` (`su` fails; HOME in the pod is
   `/proj/rdi/staff/sohroy`).

**Open issue, unresolved as of this pass:** every llama.cpp request so far (both builds, with
and without the `libomp.so.5` preload) takes a suspiciously constant ~117s regardless of
prompt length (pp16) or which build runs it. Ruled out so far: CFS CPU throttling (`nr_periods:
0` at every cgroup level from the leaf to `kubepods.slice`), page-cache thrashing (`pgmajfault`
does not move during a slow request), missing AVX512/BF16 (`-march=native` correctly resolves
to `znver5` with AVX512/AVX512BF16/FMA), and GNU OpenMP passive-wait (`OMP_WAIT_POLICY=ACTIVE` +
large `GOMP_SPINCOUNT`, confirmed reaching the process via `/proc/<pid>/environ`, changed
nothing). `/proc/stat` for the bound cores shows ~97% idle *during* an active request, and
per-thread jiffies confirm the process itself is only getting ~1 core's worth of real
execution across 32 threads. A cluster-wide advisory lock dir at `/proj/zendnn/k8/.locks` (one
`.lock` file per pod name, shared across multiple differently-named fleets e.g. `xcovoid0014`,
`xcovoid0021`, `xcovoid0023` all having their own `pod-5`) may be relevant if those map to
shared physical hardware, but this is not confirmed. Do not trust throughput numbers from this
pod pair until this is root-caused.

## Step 3 — check the plan and the machine

```bash
llmbench sweep plan --spec sweep.turin-32c-8b.yaml
```

- There should be 12 deployments, each with 19 workloads, and every one showing
  `cpus=192-223` and `membind=[6]`. If the core list differs, cores 192-223 are not 32
  physical cores on node 6 on this host (`lscpu -e`, `numactl -H`). Stop and ask.
- Read every `!` warning. None is expected.
- Check nobody else is using cores 192-223: `mpstat -P 192-223 1 5` (or `top`, press `1`).
  If they are busy, tell the user before starting. The results would be contaminated, and
  the ranking will refuse any row with 15% or more foreign load on its cores.

## Step 4 — smoke test (about 20 minutes)

Every variant starts its server at least once here, so a bad path or flag fails now rather
than at 3am. Create `sweep.turin-smoke.yaml` as a copy of the real spec, with three changes:

```yaml
name: turin-smoke
out_dir: out/turin-smoke
workload:
  n_prompt: [16]
  n_gen: [16]
  concurrency: [1]
  reps: 1
  warmup_fixed: 1
```

```bash
llmbench sweep run --spec sweep.turin-smoke.yaml
cat out/turin-smoke/report.md
```

There are 24 trials: pp16 and tg16 for each of the 12 variants. Every row must be `ok`; the
llama.cpp trials also produce a `src=server` row. For any row that is not `ok`, read `out/turin-smoke/logs/<deployment>/
server-i0.log`, fix the spec, and re-run with `--resume`, which retries only the failed
trials. Also check in the smoke logs:

- **ZenDNN is actually used.** Look for a ZenDNN line in the ZenDNN servers' logs
  (`grep -i zendnn out/turin-smoke/logs/*/server-i0.log`). If that build does not accelerate
  Q8_0, `llamacpp-zendnn-q8` will quietly run the plain CPU path and match `llamacpp-q8`.
  Report which it is.
- **vLLM loaded W8A8.** `vllm-w8a8`'s log should show a compressed-tensors / int8
  quantization method, not an error or a fallback.
- **Pinning took.** `out/turin-smoke/deployments.json` → `live.affinity[].verified` should be
  `true` for each deployment.

Stop and report if any variant cannot be made to start. Don't launch the overnight run with a
variant known to be broken without the user's go-ahead.

## Step 5 — the full run

```bash
nohup llmbench sweep run --spec sweep.turin-32c-8b.yaml > sweep.log 2>&1 &
```

SIGINT, SIGTERM and SIGHUP all stop it cleanly: the live servers are torn down and reports
are written from whatever was measured. To stop it, `kill <pid>` once, with the pid taken
from `out/turin-32c-llama31-qwen36/run.json`, and wait for it to finish tearing down.

**Monitoring:**

```bash
tail -f out/turin-32c-llama31-qwen36/events.jsonl        # every launch, trial, warning
jq '{status, error, attempt, counts}' out/turin-32c-llama31-qwen36/run.json
```

**If it dies:**

1. Read `status` in `run.json`. `interrupted` and `failed` both come with a reason.
   `running` with no live process means it was killed too hard to record anything, for
   example SIGKILL or the OOM killer.
2. Run `llmbench sweep cleanup out/turin-32c-llama31-qwen36 --dry-run`. If it lists anything,
   run it again without `--dry-run`.
3. Continue with `llmbench sweep run --spec sweep.turin-32c-8b.yaml --resume`. Finished trials
   are kept; failed and unfinished ones are redone.

You may change only the timeouts, `settle_s`, `continue_on_error` or the objective between
attempts. Anything else changes the plan fingerprint and `--resume` will refuse it. That is
deliberate, because it would mix two experiments into one table.

## Step 6 — report back

Results are in `out/turin-32c-llama31-qwen36/`: `report.md`, `report.html`, `report.csv` (full
precision) and `best.json`. Give the user:

1. **A table of `tps_mean`**, one row per test (pp1 … pp4096, tg128) and one column per
   variant, from `report.csv` rows with `src=client`.
2. **The winner per test** (`best_per_test` in `best.json`), and where the winner changes as
   pp grows. Don't lead with the single `best` row; it hides that crossover.
3. **The two comparisons asked for:** ZenDNN vs stock llama.cpp, at BF16 and at Q8_0; and
   16-bit vs 8-bit weights for each engine.
4. **Anything that makes a number untrustworthy:**
   - rows excluded from the ranking (the notes list them, with the reason);
   - contention warnings;
   - `error` or `capacity` rows;
   - a warm-up that hit its cap;
   - whether ZenDNN was actually active for Q8_0.
5. **The comparison with the user's earlier offline runs.** These numbers go through the HTTP
   server, so they should be somewhat below `llama-batched-bench` and `vllm bench latency`,
   most visibly at small pp. A gap far larger than that, or a server number *above* the
   offline one, is worth flagging.

Say plainly what you changed in the spec or the code, and why.

## Code map, for when something breaks

| file | job |
|---|---|
| `llmbench/suite/cli.py` | subcommands, signal handling, final reports |
| `llmbench/suite/spec.py` | YAML schema and validation (backend variants, per-backend `deployment:` overrides, `x-` anchor keys) |
| `llmbench/suite/plan.py` | expands the spec into deployments × workloads; `fingerprint()` for resume |
| `llmbench/suite/execute.py` | `SweepRunner`: launch, measure, persist, resume, `events.jsonl`, `run.json` |
| `llmbench/suite/deploy.py` | server command lines, numactl, health wait, affinity check, teardown |
| `llmbench/suite/procs.py` | PID ledger (`pids.json`), die-with-parent, `cleanup` |
| `llmbench/suite/objective.py` | ranking; `quality_issues()` excludes contaminated rows |
| `llmbench/runner.py`, `llmbench/metrics.py` | request timing and statistics, shared with single-endpoint `llmbench` |
