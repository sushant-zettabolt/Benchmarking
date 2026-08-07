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

## Commands

- `llmbench [flags]` -- run a benchmark against one endpoint (the default/implicit command).
- `llmbench sweep {plan,run,report}` -- the orchestrated framework above
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
