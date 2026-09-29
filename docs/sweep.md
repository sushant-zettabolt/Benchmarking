# `llmbench sweep` — the orchestrated framework

The single-endpoint `llmbench` command attaches to a server you started yourself. `llmbench
sweep` inverts that: it **owns** the servers. It decides their CPU and memory placement,
launches however many instances a configuration calls for, puts a load balancer in front of
them when there is more than one, drives the workload, tears everything down, and then tells
you which configuration was best against an objective you declare.

```bash
llmbench sweep plan   --spec sweep.yaml          # resolve and print; launches nothing
llmbench sweep run    --spec sweep.yaml          # execute
llmbench sweep run    --spec sweep.yaml --dry-run
llmbench sweep report out/sweep-8b               # re-render reports from artifacts
llmbench sweep run    --spec sweep.yaml --resume # continue an interrupted run in place
llmbench sweep cleanup out/sweep-8b [--dry-run]  # stop servers a killed run left behind
```

Start from [`sweep.example.yaml`](../sweep.example.yaml).

---

## The two axis groups

Everything in the spec is one of two kinds of knob, and they differ by cost:

| | `deployment:` | `workload:` |
|---|---|---|
| changes | how servers are **launched** | what the client **sends** |
| e.g. | backend, instances, n_parallel, n_ctx, batch/ubatch | n_prompt, n_gen, concurrency, request_rate |
| cost of changing | full fleet relaunch — minutes for an 8B model on CPU | nothing; runs against the live, warm fleet |

The planner nests **deployment outermost**, so every workload that can share a live fleet
does. This is the same rationale as `config.NESTING_ORDER` in the single-endpoint path
(after `llama-bench.cpp:1294-1435`), applied one level up.

Both accept scalars, lists, or llama-bench range syntax: `instances: 4`, `instances: [1,2,4]`
and `instances: "1-8*2"` are all valid.

### Backend variants

A key under `backends:` is a **name**; `type:` says which engine it is. Leave `type` out and
the key must itself be `llamacpp` or `vllm`, which is every spec written before variants. Name
several installations of one engine to compare builds or weights in one run and one table:

```yaml
backends:
  llamacpp-zendnn-q8: {type: llamacpp, server_bin: /opt/zendnn/llama-server, model: ~/m/q8.gguf}
  llamacpp-q8:        {type: llamacpp, server_bin: /opt/stock/llama-server,  model: ~/m/q8.gguf}
  vllm-w8a8:          {type: vllm,     server_bin: .venv/bin/vllm, model: /tmp/m/w8a8}
deployment:
  backend: [llamacpp-zendnn-q8, llamacpp-q8, vllm-w8a8]
```

The type decides the launcher, the HTTP client and the offline tool. The name is what rows,
config labels and the ranking show, so two llama.cpp builds never merge into one `llamacpp`
config. `~` in `model` and `*_bin` paths is expanded; argv reaches the server without a
shell, so nothing else would expand it. [`sweep.turin-32c-8b.yaml`](../sweep.turin-32c-8b.yaml)
is a worked example with six variants.

A backend can give its own values for the server axes `n_ctx`, `n_parallel`, `batch`,
`ubatch` and `threads_per_instance`. These *replace* the global `deployment:` values for that
backend rather than being crossed with them. Use this for settings with no common meaning
across engines, such as llama.cpp at `-c 32000` next to vLLM at `--max-model-len 8192`:

```yaml
backends:
  llamacpp-q8: {type: llamacpp, ..., deployment: {n_ctx: 32000}}
deployment:
  n_ctx: [8192]        # everyone else
```

Rows record the value each server was actually launched with. The test list works the same
way: a backend's `workload: {n_prompt: [...], n_gen: [...]}` replaces the global lists for that
backend only, for instance to give a long-context model prompts the others cannot hold. Reps,
warm-up and concurrency stay global. Tests only some backends ran get their winner from those
backends, and the ranking notes that configs covered different numbers of tests.

Top-level keys starting with
`x-` are ignored, as in docker-compose. They exist to hold YAML anchors shared by several
backends (`x-llamacpp-env: &env {...}`, then `env: *env`).

---

## CPU placement

```yaml
cpu:
  budget: "0-95"       # cores the inference servers may use (NUMA node 0)
  smt: exclude         # exclude | include | only-siblings
  ccd_align: true
  reserve: 0
  membind: auto        # bind memory to the node the cores live on
  numa_policy: membind # membind | interleave | none
```

The budget is split into `instances` disjoint, contiguous core sets. Three things this gets
right that a hand-written `numactl` line usually does not:

**CCD alignment.** On EPYC, L3 is private per CCD (8 physical cores sharing 32 MiB on a
9R14). An instance straddling a CCD boundary gets two partial L3 slices instead of one whole
one. The allocator splits on CCD boundaries where the arithmetic allows and **warns when it
cannot** rather than silently handing you a misaligned layout:

```
96 cores = 12 CCDs  ->  1, 2, 3, 4, 6, 12 instances all align
                    ->  8 instances does not; you get a warning
```

**`--physcpubind`, not `--cpunodebind`.** `cpunodebind` restricts to a node but still permits
every logical CPU on it, including the SMT siblings `smt: exclude` was meant to exclude.

**Splitting by physical core, then attaching siblings.** An instance never gets one half of a
physical core while the other half goes to a different instance — which would make two
"isolated" instances silently share execution resources.

`reserve` holds back cores from the instance budget. With `ccd_align: true` it rounds **up to
a whole CCD**, because a 4-core reserve off a 12-CCD budget leaves 92 cores, which divides
evenly by nothing.

### Verifying placement actually took

`taskset -cp <pid>` is misleading for llama.cpp. ggml calls `clear_numa_thread_affinity()`
after each graph compute, which resets the calling thread's mask to the whole machine — so
the main thread reads back as `0-383` even when every worker is correctly pinned. The suite
reads `/proc/<pid>/task/*/status` instead and classifies each thread as *within allocation*,
*affinity cleared* (expected, harmless) or *on other CPUs* (a real failure). Observed on this
host: 401 of 402 threads inside the allocation, one cleared. Results land in
`deployments.json`.

---

## Multi-instance and load balancing

```yaml
lb:
  kind: client        # client | nginx | none
  strategy: least-outstanding
  uniform: true
  n_cpus: 4
```

**`client`** (default) — llmbench fans requests out to the instances itself. No proxy process
exists, so nothing is added to the measured wire-to-wire window and no cores are spent
forwarding bytes. More accurate for measurement.

**`nginx`** — a real reverse proxy in front of the fleet. The proxy hop **is inside the
measured window**; that is the point, since it is the production topology. Rows are labelled
so they are never silently compared against `client` rows. Requires nginx on `PATH`
(`sudo apt-get install -y nginx-light`); it runs unprivileged on a high port with a generated
config, writing its pid/temp/logs under the run directory.

`uniform: true` routes even a 1-instance layout through nginx, so 1-instance and N-instance
latencies stay comparable — either both pay the proxy hop or neither does.

Details in the generated config:

- `proxy_read_timeout` is tied to `request_timeout_s`. A CPU-backend prefill of a long prompt
  can exceed nginx's 60s default, and the resulting 504 would be recorded as a backend
  failure when it was really the proxy giving up. This one matters in practice here.
- `proxy_next_upstream off` — a silent retry onto another instance would be timed as one slow
  request and would hide a failing backend.
- `proxy_buffering off` / `postpone_output 0` are the standard SSE settings and are kept as
  insurance, **but they were measured to make no difference on nginx 1.18 here**: sweeping a
  fake SSE upstream from 40ms down to 0.2ms inter-token intervals, buffering on and off gave
  identical median ITL to two decimal places. nginx forwards `text/event-stream` as it arrives
  so long as the client keeps up. Keep them (other versions, or a gzip/`sub_filter` in the
  chain, do behave differently) but do not credit them for a latency result without measuring.

nginx version compatibility: `-e` (command-line error-log path) only exists from **1.19.5**;
older builds reject it outright with `invalid option: "e"`. The launcher probes `nginx -v` and
only passes it when supported. On older nginx you will see one harmless startup alert about
the unwritable compiled-in `/var/log/nginx/error.log`; it still starts and binds.

### Where nginx's own CPUs come from

`topology.auxiliary_cpus()` picks them, and it will **never place the helper on an SMT sibling
of a core running inference** unless there is nowhere else on the machine. Order of
preference:

1. `lb.cpus`, if you named them explicitly. Used **verbatim** — all of it, not the first
   `n_cpus` of it. `n_cpus` is how many cpus to go and find, not a cap on what you asked for.
2. `cpu.reserve` cores — the clean answer, and the only one that stays on-node when the budget
   covers a whole socket. `reserve: 8` gives nginx a CCD of its own.
3. Any cpu on the instances' NUMA node whose physical core is running no inference. That
   includes budget cores no instance was given (a `cores_per_instance` remainder), not just
   cores outside the budget. Primaries first, then siblings of *idle* cores.
4. The same, on the other socket — a hop for the proxy, but no interference.
5. SMT siblings of instance cores — last resort, and warned about.

Tiers accumulate rather than being all-or-nothing: a `reserve` smaller than `n_cpus` is used
and topped up from the next tier, and the placement's `source` names every tier it drew from
(`reserved+same-node-idle-cores`).

A hyperthread looks free and is not. It shares execution units with its sibling, and ggml
synchronises every worker thread on a barrier at each graph compute, so one thread descheduled
by a busy sibling stalls the whole decode step. Measured here with a 32-core budget and 2
instances, moving nginx off the instance cores' siblings onto free physical cores:

| nginx placement | tg32 TTFT p50 | tg32 t/s |
|---|---:|---:|
| SMT siblings of instance cores | 144 ms | 15.3 |
| free physical cores | 62 ms | 18.7 |
| *(no proxy, direct)* | 65 ms | 19.2 |

With the budget set to a whole socket (`0-95`) there is no free on-node core, so nginx crosses
to the other socket. If that socket is also busy, set `cpu.reserve: 8` instead.

**The proxy hop itself is not the cost.** Measured against a deterministic SSE upstream over 30
requests, nginx adds **0.78 ms to TTFT and 0.04 ms to ITL**; against a real `llama-server` on
separate cores, ITL was 31.85 ms direct versus 31.84 ms proxied. Any large nginx-vs-client gap
you see is placement or contention, not the proxy — check `provenance.contention` first.

### Things that quietly invalidate a measurement

All are detected and reported at plan time or on the row:

0. **`reps` too close to `concurrency`.** A closed-loop measurement needs enough requests that
   steady state outweighs the ramp-up at the start and the drain at the end. Measured here:
   the same workload gave 26-46% relative stddev at `reps: 6, concurrency: 4`, and 3-8% at
   `reps: 40`. Use `reps >= 10x concurrency`; below 4x you get a warning.
1. **Concurrency below instance count.** With `concurrency: 1` against 4 instances, 3 sit
   idle for the whole measurement — the row describes one instance on a quarter of the cores.
   No load balancer can fix this; there is no second request to place.
2. **Tie-break bias.** `least-outstanding` with a naive lowest-index tie-break sends *every*
   request to instance 0 at concurrency 1. The implementation breaks ties on fewest
   dispatches, so ties round-robin. Actual distribution is recorded per trial in
   `provenance.lb_distribution`.

---

## Offline mode

`mode: offline` (or `both`) drives each backend's **own** benchmark tool as a subprocess and
parses its output into the same table.

| concept | llama.cpp | vLLM |
|---|---|---|
| **static batch size** | `llama-batched-bench -npl B` | `vllm bench latency --batch-size B` |
| prompt / gen tokens | `-npp` / `-ntg` | `--input-len` / `--output-len` |
| shared prefix | `-pps` | *(n/a)* |
| continuous-batch throughput | *(n/a)* | `vllm bench throughput --num-prompts` |
| single stream | `llama-bench -p -n` | `vllm bench latency --batch-size 1` |

Tool choice is automatic: `batch_size: 1` uses `llama-bench` (the canonical, most-cited
llama.cpp number); anything larger **must** use `llama-batched-bench`, because `llama-bench`
drives a single sequence and has no request-batch concept at all — its `-b` is the token batch
fed to one decode call, a different quantity that is already a deployment axis.

> **Offline rows are not comparable across backends, and are labelled as such.**
> `llama-bench` times `llama_decode()` in-process with no HTTP, no scheduler and no admission
> control. `vllm bench latency` runs the full vLLM engine including its scheduler. They are
> not the same measurement. What they *are* good for is comparing each backend to **itself**
> online: offline minus online is that backend's HTTP-plus-scheduler tax.

Repetition semantics differ, which changes what a stddev means. `llama-bench` takes `-r N` and
repeats in-process (within-process variance). `llama-batched-bench` has no repetition flag, so
N reps means N process invocations, each reloading the model (variance includes process start
and page cache). Each row records which in `raw.reps_mechanism`.

---

## Declaring what "best" means

```yaml
objective:
  metric: total_token_throughput
  goal: max                    # max | min
  src: client                  # which measurement path to rank on
constraints:
  - { metric: ttft_ms_p99, max: 5000 }
  - { metric: itl_ms_p95,  max: 200 }
```

Any metric on a trial row is addressable: `tps_mean`, `total_token_throughput`,
`request_throughput`, `ttft_ms_{mean,p50,p95,p99}`, `itl_ms_{mean,p50,p95,p99}`,
`tpot_ms_mean`, `e2e_ms_{mean,p50,p99}`, `prefill_tps_mean`, `decode_tps_mean`,
`overhead_ms_mean`. Name one that does not exist and the report says so, listing what *is*
available, rather than returning an empty ranking.

**A constraint with no measurement fails.** An unmeasured SLO is not a met SLO — treating a
missing p99 as passing would promote exactly the configurations whose latency data is missing
because they fell over.

Three answers come out, because they are different questions:

- **`best`** — the single highest-scoring feasible row.
- **`best_per_test`** — the winner for each workload separately. A config that wins `pp1024`
  routinely loses `tg64`; on this hardware the llama.cpp/vLLM crossover sits just under
  `pp64`. A single global winner hides that.
- **`best_overall`** — each configuration averaged across the workload mix. Values are
  normalised to each workload's winner *before* averaging, so a workload measured in the
  hundreds of t/s cannot outvote one measured in the tens purely by scale.

Plus a **Pareto front** over the objective and every constrained metric, which is what you
want when the constraints turn out too tight or too loose to be interesting. If nothing is
feasible, the report says so and still shows the front and per-test winners so the trade-off
stays visible.

**Untrustworthy rows are not ranked.** An `ok` row is excluded from every answer above --
it stays in the results table, and a note names it and why -- when any of these holds:

- foreign CPU load on its cores reached `CONTENTION_WARN_PCT` (15%);
- a server thread was pinned outside the deployment's allocation;
- it lost more than `objective.max_error_pct` of its requests (default `0`: any lost request
  disqualifies, because throughput computed from the requests a config kept is not
  throughput it delivered).

These are read from each row's `provenance`, so `sweep report` applies them to old runs too.
`objective.rank_flagged: true` ranks them anyway, for diagnosing a noisy box.

---

## Artifacts

```
out/<name>/
  run.json          manifest: status, spec, env, topology, objective, counts, timings
  plan.json         the fully-resolved plan — every port, core list and membind node
  deployments.json  what actually launched, incl. per-thread affinity verification
  trials.jsonl      one JSON row per measured result, appended as it completes
  events.jsonl      timestamped log of every launch, trial, warning, stop and resume
  pids.json         every process the run started and has not yet seen exit
  offline.jsonl     native-tool rows with their argv, env and parsed output
  records/<dep>.jsonl   raw per-request records
  llmbench.db       the same raw records in SQLite
  logs/<dep>/       server stdout+stderr, nginx logs, offline tool output
  report.html       self-contained: no CDN, no external assets
  report.md
  report.csv        raw precision, for re-sorting in a spreadsheet
  report.json
  best.json         the machine-readable answer
```

Rows are appended **as they complete**, so a sweep killed at trial 90 of 126 still leaves 89
fully analysable results. A truncated final line from a hard kill is skipped and counted, not
allowed to sink the whole report.

Raw per-request records are kept for every trial and are never discarded in favour of the
aggregates. But note what `llmbench sweep report <dir>` does by default: it **replays** the
metrics already flattened into `trials.jsonl` rather than recomputing them. That is right for
a formatting change and wrong for anything else — a fix or an addition in `metrics.py` would
silently not reach a finished run. Pass `--from-records` to recompute every statistic from
`records/` instead:

```bash
llmbench sweep report out/my-sweep --from-records
```

Use it after changing `metrics.py`, and whenever a number looks wrong: it is the difference
between re-rendering a stored answer and re-deriving it. The one thing it cannot repair is a
value that was already wrong *in* the raw record — for that the run has to be repeated.

Pointing a second run at an `out_dir` that already holds results renames the old
`trials.jsonl`/`offline.jsonl`/`deployments.json` to `<file>.<previous-run-id>.<ext>` first,
so the report describes one sweep rather than silently splicing two. Server logs are never
truncated either: a relaunch into an existing `server-i0.log` moves it to `server-i0.1.log`.
Every row also carries its own `run_id`.

`run.json`'s `status` records how the run ended:

| status | meaning |
|---|---|
| `running` | in progress -- or, if nothing is writing to the directory, killed too hard to record anything (SIGKILL, power loss). Check `pids.json` / `sweep cleanup` |
| `finished` | every planned unit ran (some may still be `error` rows) |
| `interrupted` | stopped by a signal; `stop_reason` says which. Servers were torn down and reports written |
| `failed` | an exception ended it (`continue_on_error: false`, or a harness bug); `error` and `traceback` say what. Reports were still written |

## Stopping, resuming, and cleaning up

**SIGINT, SIGTERM and SIGHUP all stop the run in an orderly way:** the trial in flight is
abandoned, the live fleet is torn down, and the reports are written from whatever was
measured. SIGHUP matters most -- it is what an overnight run gets when the SSH session it
was started from drops. A second signal does not interrupt the teardown the first one
started.

**Resume** continues the run recorded in `out_dir`, under the same `run_id`:

```bash
llmbench sweep run --spec sweep.yaml --resume
```

Units that finished (`ok` or `capacity`) are kept, and a deployment whose workloads are all
done is not relaunched. Units with an `error` or `skipped` row, and the one the run died
inside, are measured again. The previous `trials.jsonl` is archived as
`trials.<run-id>.attempt<N>.jsonl` before its superseded rows are dropped. Raw records of
later attempts carry an attempt-qualified `run_id`, so `report --from-records` never pools
a failed attempt's requests with the retry's.

Resume is refused if the spec now describes different measurements: `run.json` stores a
fingerprint of every trial, port, core list, the host topology, and every launch and
workload setting. Timeouts, `settle_s`, `continue_on_error`, the objective and the name may
change between attempts. Raising `startup_timeout_s` after a slow load failed is the usual
reason to resume.

**If the sweep process itself is killed** (SIGKILL, OOM killer), no `finally` runs. Two
things still cover it. On Linux every server and offline tool is started with
`PR_SET_PDEATHSIG`, so the kernel sends it SIGTERM when the sweep dies. That reaches the
direct child only, not grandchildren such as vLLM's engine core. So every launch is also
recorded in `pids.json`, with its kernel start time, and

```bash
llmbench sweep cleanup out/sweep-8b --dry-run   # list what is still alive
llmbench sweep cleanup out/sweep-8b             # SIGTERM, then SIGKILL after 30s
```

stops exactly those processes. A process counts as ours only if it has the recorded PID
*and* start time, or belongs to the session the run created and started after its leader,
so a reused PID is never signalled. A new run in the same `out_dir` refuses to start while a
previous run's processes are alive, or while another sweep is still running there.

### What a row's `status` means

| status | meaning |
|---|---|
| `ok` | measured; at least one request succeeded (a partial failure adds a row warning naming the count) |
| `capacity` | the configuration did not fit — a server that failed to start on an allocation marker, or a concurrency the fleet has no slots for. A result, not a bug |
| `error` | the trial ran but produced nothing usable, including the case where *every* request failed. Such a row would otherwise aggregate to a plausible-looking 0 t/s and rank as if it were real |
| `skipped` | `--dry-run` |

---

## Safety on a shared machine

This host carries a long-running `llama-server` on port 18080 belonging to another session.
Two rules keep sweeps away from it:

1. **Only PIDs we started are ever signalled.** `sweep cleanup` works from the `pids.json` ledger and
   verifies each PID's start time, never a process name. Teardown goes through a real `Popen` handle
   and signals the process *group* (vLLM's engine core is a separate process; signalling only
   the leader orphans it and leaves the port bound). There is no `pkill`-by-name anywhere in
   the codebase — `pkill -f llama-server` would kill the other session's server. This applies
   to the offline tools too: they are run under `Popen`, not `subprocess.run(timeout=...)`,
   which on a timeout kills only the direct child and would strand a `vllm bench` engine
   holding cores and KV memory for the rest of the sweep.
2. **Port pre-flight.** Every port a deployment needs is checked free *before* the previous
   deployment is torn down. If something else is listening, the run refuses to start and tells
   you to change `base_port`/`lb.port`. llmbench will not stop a process it did not start.

Servers that fail to start with an allocation-failure marker in their log are recorded as
`capacity` — "this configuration did not fit" is a legitimate result, not a harness bug — and
the sweep continues.

## Contention detection

Pinning guarantees our servers stay on their cores. It guarantees nothing about who *else* is
using those cores, and on a shared box an unpinned job from another user silently halves your
throughput while the numbers still look plausible and reproducible.

So every trial samples it. `/proc/stat` gives per-CPU busy time over the trial; subtracting
what our own managed PIDs consumed leaves the foreign load. It lands in each row's
`provenance.contention` and, above `CONTENTION_WARN_PCT` (15% of the allocated CPUs), becomes
a warning on the row and in the report:

```
!! d000: 8.5 core(s) of foreign CPU load (53% of the 16 allocated cpus) were active during
   this measurement -- processes this run does not own are competing for the same cores.
   Treat these numbers as contaminated.
```

This is not hypothetical. It was added after an unpinned `llama-mtmd-cli` from another session
ran across all 384 logical CPUs and made an nginx-vs-client comparison look like a 10x
regression that was entirely contention — caught only because the numbers were implausible
enough to go looking. Check this field before believing any comparison from this machine.
