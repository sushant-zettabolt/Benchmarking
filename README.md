# llmbench

An endpoint-driven inference benchmark harness for llama.cpp-server and vLLM.

**Start here: [docs/contract.md](docs/contract.md).** It defines the one measurement
contract this tool implements identically for both backends, and why matching
`llama-bench`'s or `vllm bench serve`'s own numbers directly is not the goal (see
"What this is not" in that document). The short version:

> `llmbench` measures an endpoint, not a decode loop. `llama-bench` times `llama_decode()`
> in-process -- no HTTP, no scheduler, no admission, no sampling. vLLM has no equivalent of
> that: it has exactly one execution path, and it always goes through the scheduler. So
> `llmbench` defines its own boundary (wire-to-wire: last byte of the request out, to last
> byte of the final content chunk in) and applies it identically to both backends, using
> `llama-bench` and `vllm bench serve` as **calibration anchors** that bound and explain the
> numbers -- never as targets to match.
>
> At concurrency 1 with random tokens, llama.cpp and vLLM are close to comparable. Above
> that they are not measuring the same system: llama.cpp uses a fixed slot pool with no
> preemption and slot-local(-ish) prefix caching; vLLM uses paged KV with preemption and
> global content-hashed prefix caching. `llmbench parity` checks and reports this divergence
> before any comparison is printed, rather than hiding it behind a single throughput number.

## Install

```bash
pip install -e .
```

Requires Python >=3.10. Dependencies: `httpx` (async streaming, no `requests`), `pyyaml`,
`rich`. No pandas -- `statistics`/hand-rolled percentile math covers everything in
`metrics.py`.

## Quick start

Point it at a running llama.cpp `llama-server` or vLLM OpenAI-compatible server:

```bash
llmbench --url http://127.0.0.1:8080 -m qwen2.5-0.5b-instruct -p 512 -n 128 -r 5
```

```
| model                 | size      | params | backend  | src    |  test | t/s              |
| --------------------- | --------: | -----: | -------- | ------ | ----: | ---------------: |
| qwen2.5-0.5b-instruct | N/A       | N/A    | llamacpp | server | pp512 | 141.96 ± 2.02     |
| qwen2.5-0.5b-instruct | N/A       | N/A    | llamacpp | client | pp512 | 138.24 ± 2.03     |
```

`src=client` (Path A) is wall-clock, black-box, timestamped from the requesting process --
comparable across backends, and the number your application will actually experience.
`src=server` (Path B) is backend-reported (llama.cpp's inline `timings` object; vLLM's
`/metrics` delta scrape) -- a **diagnostic**, not a cross-backend result (see
[docs/contract.md](docs/contract.md)'s "`src=server` numbers are diagnostics, not results").

Raw per-request records are always written to `--out-dir` (JSONL + SQLite) regardless of
stdout format -- results are never aggregated at collection time; `metrics.py` computes
every statistic later, from those raw records.

## Two ways to use it

**Attach mode** (`llmbench ...`) points at a server you started yourself and measures it.

**Sweep mode** (`llmbench sweep ...`) owns the servers: it decides their CPU/NUMA placement,
launches N instances, load-balances across them, drives the workload, tears everything down,
and reports which configuration was best against an objective you declare. It also drives each
backend's native *offline* tools (`llama-bench`, `llama-batched-bench`, `vllm bench
latency/throughput`) including static batch size.

```bash
llmbench sweep plan --spec sweep.yaml     # resolve and print; launches nothing
llmbench sweep run  --spec sweep.yaml     # execute, then write report.{html,md,csv,json}
llmbench sweep report out/sweep-8b        # re-render reports from stored artifacts
llmbench sweep run  --spec sweep.yaml --resume   # continue an interrupted run in place
llmbench sweep cleanup out/sweep-8b       # stop servers a killed run left behind
```

```yaml
cpu:        { budget: "0-95", smt: exclude, ccd_align: true }
lb:         { kind: nginx, uniform: true }
deployment: { backend: [llamacpp, vllm], instances: [1, 2, 4], n_parallel: [1, 8] }
workload:   { n_prompt: [16, 32, 64, 1024], n_gen: [64], concurrency: [1, 8], reps: 5 }
offline:    { batch_size: [1, 4, 16] }
objective:  { metric: total_token_throughput, goal: max }
constraints: [{ metric: ttft_ms_p99, max: 5000 }]
```

See **[docs/sweep.md](docs/sweep.md)** for the full schema and
[sweep.example.yaml](sweep.example.yaml) for a worked example. Highlights:

- **CCD-aware core splitting.** 96 cores is 12 CCDs on an EPYC 9R14, so 1/2/3/4/6/12 instances
  align to L3 boundaries and 8 does not — the planner says which, rather than silently handing
  you a layout whose instances share cache.
- **Placement is verified, not assumed.** `taskset` lies for llama.cpp (ggml clears thread
  affinity after each compute); the suite reads `/proc/<pid>/task/*/status` instead.
- **Offline rows are labelled not-comparable across backends** — `llama-bench` times
  `llama_decode()` with no scheduler, `vllm bench latency` runs the full engine. Compare each
  backend's offline number to its own online number, not to the other backend's.
- **A constraint with no measurement fails.** An unmeasured SLO is not a met SLO.
- **Only PIDs it started are ever signalled**, and target ports are checked free before
  anything is torn down — this machine has other people's servers on it.

## Running the Turin sweep

[`sweep.turin-32c-8b.yaml`](sweep.turin-32c-8b.yaml) compares ZenDNN llama.cpp, stock
llama.cpp and vLLM at 16-bit and 8-bit weights (BF16 / Q8_0 / W8A8), for Llama 3.1 8B and
Qwen3.6-35B-A3B: 12 variants × 19 tests (pp1…pp4096, tg128) × 3 reps + 1 warm-up = 228 trials,
each a single server pinned to cores 160-191 / NUMA node 5 of `turin-xcovoid0021-pod-5`.
[`scripts/run_turin_sweep.sh`](scripts/run_turin_sweep.sh) runs it deterministically: it stops
at the first failed check and only starts the full run after a smoke run in which every variant
started and measured.

| step | check |
|---|---|
| 1 | this process's cpuset is exactly `160-191` (the right pod) |
| 2 | `pytest` passes |
| 3 | every model, server binary and `LD_PRELOAD` library in the spec exists |
| 4 | no server from an earlier run is alive; cores 160-191 are < 15% busy |
| 5 | the plan is exactly 12 deployments / 228 trials, all on `cpus=160-191 membind=[5]` |
| 6 | smoke run (`sweep.turin-smoke.yaml`, pp16 + tg16 per variant): all 24 rows `ok`, none < 1 t/s |
| 7 | the full run |

The checkout at `/proj/aigstaff/sohroy/Benchmarking` is on shared NFS: the workstation and
the pod see the same files, so update it (`git pull`) wherever you have GitHub access, then run.
The commit that ran is recorded in `run.json` (`host_info.llmbench_commit`).

**From the workstation** (non-interactive; `kubectl` is not on `PATH` there):

```bash
K="/usr/local/bin/kubectl exec turin-xcovoid0021-pod-5 -n zendnn --"
$K runuser -l sohroy -c 'cd /proj/aigstaff/sohroy/Benchmarking &&
  nohup scripts/run_turin_sweep.sh > sweep-turin.log 2>&1 < /dev/null & echo pid $!'
```

**Or inside the pod** (`kubectl exec -it turin-xcovoid0021-pod-5 -n zendnn -- login sohroy`):

```bash
cd /proj/aigstaff/sohroy/Benchmarking
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'     # first time only
nohup scripts/run_turin_sweep.sh > sweep-turin.log 2>&1 < /dev/null &
```

`scripts/run_turin_sweep.sh smoke` stops after step 6; `full` skips the smoke run.

**Watch it:**

```bash
tail -f sweep-turin.log                                        # the script's own steps
tail -f out/turin-32c-llama31-qwen36/events.jsonl              # every launch, trial, warning
jq '{status, error, attempt, counts}' out/turin-32c-llama31-qwen36/run.json
```

**Stop it:** `kill <pid>` once, with the pid from `out/turin-32c-llama31-qwen36/run.json`
(SIGINT/SIGTERM/SIGHUP tear the live server down and write reports from what was measured).
Never kill servers by name; other people's servers run on this machine.

**If it dies:** `llmbench sweep cleanup out/turin-32c-llama31-qwen36` (only signals PIDs this
run recorded), then `scripts/run_turin_sweep.sh resume` — finished trials are kept, failed and
unfinished ones redone. Only timeouts, `settle_s`, `continue_on_error`,
`core_sample_interval_s` and the objective may change between attempts.

**Results** (`out/turin-32c-llama31-qwen36/`):

| file | contents |
|---|---|
| `report.md`, `report.html` | ranking, per-test winners, full table, host/pod details, the exact server command of every deployment |
| `report.csv` | one row per trial and source (`src=client` is the comparable one): mean t/s, prefill t/s, decode t/s, TTFT, ITL, TPOT, e2e, … and the `server cmd` |
| `report_reps.csv` | every measured repetition of every trial (3 per test), not only the mean |
| `cores/<deployment>.csv` | per-core busy % of cores 160-191 every 250 ms, labelled by trial and phase (idle / warmup / rep N), with server / harness / foreign load split |
| `best.json` | the winner per test and overall |
| `run.json`, `deployments.json`, `records/`, `logs/` | host and software versions, launch commands, raw per-request records, server logs |

## Commands

- `llmbench [flags]` -- run a benchmark against one endpoint (the default/implicit command).
- `llmbench sweep {plan,run,report,cleanup}` -- the orchestrated framework above
  ([docs/sweep.md](docs/sweep.md)).
- `llmbench parity --a <url> --b <url>` -- probe both servers, classify every parity axis
  (weights, KV dtype, context capacity, KV memory budget, batch shaping, attention backend,
  prefix-cache state, sampling, warm state) as `matched` / `mismatched` / `unverifiable`,
  write `parity.json`, and refuse to endorse a cross-backend comparison unless every axis
  is `matched`. Run this before trusting any A/B number.
- `llmbench quality --a <url> --b <url> --model-a ... --model-b ...` -- greedy-decoding
  divergence check (exact-match rate, first-divergence position, top-k KL where available).
  Required before any `--parity-mode native` comparison is meaningful -- a throughput
  "speedup" without this is not a claim the tool will vouch for.
- `llmbench calibrate` -- runs native `llama-bench` and `llmbench` (src=server) against the
  same `llama-server`/model/machine, and reports the delta as the HTTP+scheduler tax. A
  diagnostic report, not a pass/fail gate (see "What this is not" in the contract).
- `llmbench compare <run_a> <run_b>` -- diffs two stored runs and flags regressions beyond a
  threshold.

## `--parity-mode {weights,native}`

Required for any cross-backend comparison. There is no configuration that is simultaneously
identical-weights and fair-to-both-backends -- see docs/contract.md and
docs/reference-notes.md for why. `weights` sends the same GGUF to both sides (vLLM's GGUF
path is an out-of-tree plugin, recorded as such); `native` lets each backend use its
preferred format and requires `llmbench quality` in the same run.

## Docs

- [docs/contract.md](docs/contract.md) -- the measurement contract. Read this first.
- [docs/reference-notes.md](docs/reference-notes.md) -- source-verified internals of
  `llama-bench`, the llama.cpp server, and vLLM's serving/scheduler code, with file:line
  citations and a claim-by-claim verdict against every assumption this tool depends on.
- [docs/validation.md](docs/validation.md) -- gates G1-G8 with real numbers from this
  machine.
- [bench.example.yaml](bench.example.yaml) -- example config; CLI flags override YAML.

## Development

```bash
pip install -e ".[test]"
pytest
```

`tests/test_integration_fake_server.py` is the load-bearing integration test: a hand-rolled
asyncio SSE server with injectable TTFT/ITL delays and four simulated stream shapes (empty
first chunk, mid-stream stall, truncated generation, cached-prefix response), asserting the
client-side timestamp logic recovers known values within tolerance. `metrics.py` is
unit-tested to guarantee it never branches on backend identity
(`tests/test_metrics_no_backend_branch.py`) -- that guarantee is the mechanical enforcement
of the contract's "one implementation, both backends" rule.
