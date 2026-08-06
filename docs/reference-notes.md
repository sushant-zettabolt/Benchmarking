# Reference notes: llama.cpp and vLLM source verification

Milestone 1 deliverable. All claims below are verified against the pinned clones in
`reference/` (see `reference/PINNED.md` for exact commit SHAs and clone dates). No
application code has been written yet. Paths are relative to `reference/llama.cpp/` or
`reference/vllm/` as noted per section. **We do not import from or vendor either repo.**

This document is organized as: (1) llama-bench internals, (2) llama.cpp server
internals, (3) vLLM `benchmarks serve` internals, (4) vLLM scheduler/metrics internals,
(5) a claim-by-claim verdict on every assertion in the task spec's §4, (6) a consolidated
list of spec corrections that should change how `llmbench` is built.

---

## 1. `llama-bench` internals (`tools/llama-bench/llama-bench.cpp`)

### `cmd_params` / `cmd_params_defaults` (llama-bench.cpp:322–408)

Full field list confirmed at `llama-bench.cpp:322` (`struct cmd_params {`). Every field
except `numa, reps, prio, delay, verbose, progress, no_warmup, output_format,
output_format_stderr` is a `std::vector<T>` — Cartesian-multi-valued. README confirms
(`tools/llama-bench/README.md:90`): "With the exception of `-r`, `-o` and `-v`, all
options can be specified multiple times." Defaults confirmed to match the spec's §6
Group defaults (`-p 512`, `-n 128`, `-r 5`, `-b 2048`, `-ub 512`, `-ngl -1`, `-o md`).

### Range syntax — `parse_int_range()` (llama-bench.cpp:280–320)

Regex-based: `first[-last[(+|*)step]]`, comma-separated for multiple ranges in one
string. Loop applies the op (`+step` or `*mult`, default `+1`) starting at `first` while
`i <= last`; **guards against non-increasing sequences** (`+0`, `*1`, `*0`) by throwing
`std::invalid_argument`. `*` steps can overshoot and stop without ever landing exactly on
`last`. `allow_negative` (only for `-ngl`, since `-1` = "all layers") permits a negative
`first`; `last`/`step` are never negative. NOT used for `-pg` (literal `pp,tg` pair) or
enum flags (`-sm`, `-fa`, `-lm`, `-ctk`, `-ctv` — comma-separated name lookups only, no
ranges). `llmbench`'s range parser must replicate this exactly, including the
non-increasing-sequence guard.

### `get_cmd_params_instances()` — Cartesian nesting (llama-bench.cpp:1294–1435)

Explicit comment (llama-bench.cpp:1297): *"this ordering minimizes the number of times
that each model needs to be reloaded."* Nesting, outermost (varies slowest) → innermost
(varies fastest): `model → fit_params_target → fit_params_min_ctx → n_gpu_layers →
n_cpu_moe → split_mode → load_mode → main_gpu → devices → tensor_split →
tensor_buft_overrides → no_host → embeddings → no_op_offload → n_batch → n_ubatch →
type_k → type_v → no_kv_offload → flash_attn → n_threads → cpu_mask → cpu_strict →
n_depth → poll`, and *inside* that, three parallel (not nested against each other) inner
loops: `n_prompt`, `n_gen`, `n_pg` (each skips a zero/zero-pair value).

**Direct implication for Milestone 5** (`--server-mode manage`): Group-2 (server-affecting)
params should be the outer loop and Group-1 (request-shaping) params the inner loop
against one live server, mirroring this nesting to minimize restarts. One nuance:
`cmd_params_instance::to_llama_cparams()` (llama-bench.cpp:1276–1291) sets `cparams.n_ctx
= n_prompt + n_gen + n_depth` — llama-bench **resizes context per instance**, something we
cannot do against an already-running HTTP server with fixed `n_ctx`. This is precisely
why the capacity preflight (spec §4.1) is a requirement for us but a non-issue for native
llama-bench.

### `test` struct — statistics formulas (llama-bench.cpp:1437–1683)

`get_fields()` (1557–1570) is the canonical CSV/JSON/JSONL/SQL field order. **Field name
gotcha**: struct methods are named `stdev_ns()`/`stdev_ts()` but the external/output
field names are `"stddev_ns"`/`"stddev_ts"` (double-d) — a Python port's *output* field
names must use the double-d form.

`avg()`/`stdev()` (llama-bench.cpp:102–118), the canonical formulas:

```cpp
template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) { return 0; }
    return std::accumulate(v.begin(), v.end(), T(0)) / (T) v.size();
}
template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) { return 0; }   // explicit early return, not NaN
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    return std::sqrt(sq_sum / (T)(v.size()-1) - mean*mean*(T)v.size()/(T)(v.size()-1));
}
```

- `stdev` is **sample** stddev (Bessel-corrected, N−1 denominator).
- **`v.size() <= 1` → returns exactly `0.0`**, an explicit early return — not
  `statistics.stdev` (which raises on N<2) and not NaN. `llmbench`'s Python port must
  special-case this rather than call `statistics.stdev` directly.

`get_ts()` (llama-bench.cpp:1523–1529), the `t/s` formula:

```cpp
std::vector<double> get_ts() const {
    int n_tokens = n_prompt + n_gen;
    return samples_ns | transform([n_tokens](t){ return 1e9 * n_tokens / t; });
}
```

Per-repetition `ts_i = 1e9 * n_tokens / t_ns_i`; reported `t/s = avg(ts) ± stdev(ts)` is
computed **over the per-repetition throughput values themselves**, not as
`avg(tokens)/avg(time)`. This order of operations must be replicated exactly. Formatting
is `"%.2f ± %.2f"` (llama-bench.cpp:2043) — two decimals, U+00B1 surrounded by single
spaces.

### `test_prompt()` / `test_gen()` — confirmed pure in-process (llama-bench.cpp:2112–2160)

Both operate directly on `llama_decode()`; **no HTTP, no tokenizer, no real sampler**.
`test_gen()` draws the next token as `rand() % n_vocab` — llama-bench does not run a
sampler over logits at all. `test_prompt()` batches `n_prompt` random token IDs (BOS
first only if the vocab wants one and it's the first chunk) into `n_batch`-sized
`llama_decode()` calls, syncing once at the end; `test_gen()` decodes one token at a time
and calls `llama_synchronize()` **every token** (fully serialized, no pipelining).
README.md:98–99 confirms explicitly: *"The measurements with llama-bench do not include
the times for tokenization and for sampling."* This directly underwrites §8's requirement
to send/measure raw random token IDs.

### The five printers (llama-bench.cpp:1700–2104)

CSV/JSON/JSONL/SQL are field-complete/schema-identical (`get_fields()`/`get_values()`);
only markdown selects a display subset.

- **CSV**: every field double-quote-wrapped via `escape_csv` (doubles embedded `"`), even
  numeric fields.
- **JSON**: top-level array; each field type-switched via `get_field_type()` (STRING →
  quoted+escaped via `escape_json`, BOOL → bare `true`/`false`, INT/FLOAT → bare numeral);
  **also emits `samples_ns` and `samples_ts` arrays** not present in CSV/SQL — the only
  place per-repetition raw data appears in stdout. `fflush()` after each object.
- **JSONL**: same fields/typing as JSON, one object per line, no enclosing array,
  `fflush()` per line (streaming/tail-friendly).
- **SQL**: `CREATE TABLE IF NOT EXISTS llama_bench (...)` DDL (types via
  `get_sql_field_type()`: STRING→TEXT, BOOL|INT→INTEGER, FLOAT→REAL) then one `INSERT`
  per test row — **all values are single-quoted string literals regardless of declared
  column type** (llama-bench.cpp:2100). No `samples_ns`/`samples_ts`.
- **Markdown** (1806–2065): always shows `model, size, params, backend`; conditionally
  appends a column via the pattern
  `if (params.<X>.size() > 1 || params.<X> != cmd_params_defaults.<X>) fields.push(<X>)`
  — **checked once globally against the whole run's parsed param vectors vs. compiled
  defaults**, not by diffing rendered per-row values. `n_gpu_layers`/`n_threads` have an
  extra backend-type-gated force-show rule. Column widths and left/right justification
  are hardcoded per field (`get_field_width()`); right-justified columns get a trailing
  `:` in the markdown separator row, left-justified get none. `size`: MiB below 1 GiB else
  GiB (binary, 1024³ threshold); `params`: M below 1e9 else B (**decimal** threshold,
  inconsistent with `size`'s binary threshold — both must be replicated as-is). Test name:
  `pp{n}` / `tg{n}` / `pp{n}+tg{n}`, **`" @ d{n}"` suffix with a leading space** when
  `n_depth > 0` — i.e. `pp512+tg128 @ d64`, **not** `pp512+tg128@d64` as the task spec's
  example implies. The `t/s` column gets a documented `+1` width hack to compensate for
  `±` being a 2-byte/1-column UTF-8 character. Footer: `"\nbuild: %s (%d)\n"`.

---

## 2. llama.cpp server internals (`tools/server/`)

### `/completion` request fields

- `cache_prompt` (README.md:523) — **default `true`**.
- `timings_per_token` (README.md:529) — default `false`.
- `n_predict` (README.md:459) — native field name (not `max_tokens`, which is the
  OAI-compat alias on `/v1/completions`); default `-1` (infinite).
- `ignore_eos` (README.md:511) — default `false`.
- `prompt` accepts string / array of strings / array of token IDs (README.md:427). BOS is
  auto-inserted only on the *string* path if the model wants one — **a pure token-ID array
  bypasses auto-BOS-insertion**, so `llmbench` must add BOS itself when required (§8
  Path 1).

### `timings` response object — exact fields (`tools/server/server-task.h:262–280`)

```cpp
struct result_timings {
    int32_t cache_n = -1;
    int32_t prompt_n = -1;             double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;  double prompt_per_second = 0.0;
    int32_t predicted_n = -1;          double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0; double predicted_per_second = 0.0;
    int32_t draft_n = 0; int32_t draft_n_accepted = 0;   // only serialized if > 0
};
```

**Correction to spec §3**: the field list the spec guesses (`prompt_n, prompt_ms,
prompt_per_second, predicted_n, predicted_ms, predicted_per_second`) is correct but
incomplete — missing `cache_n`, `prompt_per_token_ms`, `predicted_per_token_ms`. `cache_n`
is directly the field §8's n_depth-verification step needs (tokens served from cache) —
use it instead of inferring from `prompt_n` alone.

### Endpoints

- `/health` (README.md:406–417) — GET, public. 503 while loading, 200 `{"status":"ok"}`
  when ready. `/v1/health` alias.
- `/tokenize` / `/detokenize` (README.md:613–663) — POST `{content, add_special, ...}` →
  `{"tokens":[...]}`; reverse for detokenize.
- `/props` — **GET `/props`** (README.md:766) returns `default_generation_settings.n_ctx`
  (**nested, already the per-slot budget** — README.md:777) and `total_slots`
  (README.md:855, "defined by `--parallel`"). **Correction to spec §4.1**: there is no
  separate raw total-`n_ctx`/`n_parallel` pair to divide ourselves — the server already
  reports the per-slot context directly at `default_generation_settings.n_ctx`. The
  capacity preflight is simpler than the spec implies: read that field and `total_slots`,
  no division needed.
- `/slots` (README.md:910–914) — GET, **enabled by default** (disable with `--no-slots`).
  Array of per-slot objects (`id`, `id_task`, `n_ctx`, `is_processing`, `params`).
  `?fail_on_no_slot=1` → 503 if none free.
- `/metrics` (README.md:1058–1079) — GET, **only present if `--metrics` was passed**
  (disabled by default — unlike vLLM, where /metrics is normally on). Prefix `llamacpp:`.
  Full metric list: `llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:prompt_tokens_seconds` (Gauge), `llamacpp:tokens_predicted_total`,
  `llamacpp:tokens_predicted_seconds_total`, `llamacpp:predicted_tokens_seconds` (Gauge),
  `llamacpp:requests_processing` (Gauge), `llamacpp:requests_deferred` (Gauge),
  `llamacpp:n_tokens_max` (Counter), `llamacpp:n_decode_total`,
  `llamacpp:n_busy_slots_per_decode` (Gauge). **No preemption counter** (expected — see
  below) and **no TTFT/ITL histograms** — this endpoint is much thinner than vLLM's.
  `requests_deferred` is the closest available queueing signal.

### Launch flags — corrections to spec §1/§4.1/§4.2

- `-np, --parallel N` (README.md:177) — **default `-1` (auto)**, not a value the operator
  must always set explicitly. `llmbench`'s capacity preflight must read the *resolved*
  value back from `/props.total_slots` after the server is up, not assume the launch flag
  value is authoritative.
- `-c, --ctx-size N` (README.md:50) — default `0` (loaded from model); this is the
  **total** context before division across slots.
- `--cache-ram, -cram N` (README.md:168) — **default `8192` MiB**, not off. This is why
  the spec's §4.2 recommendation of `--cache-ram 0` is necessary — the out-of-the-box
  behavior already has an 8 GiB cross-slot RAM cache active.
- `--cache-idle-slots` (README.md:170) — default **enabled** (requires `--cache-ram`).
- `--context-shift` / `--no-context-shift` (README.md:171) — **default: disabled** (i.e.
  context shift is opt-in, not opt-out). See below — this is a load-bearing correction to
  spec §4.1/§11.2.

### Slot selection / prompt-cache reuse — `get_available_slot()` (server-context.cpp:1574–1661)

1. If the task pins `id_slot`, use it directly.
2. Else, if `slot_prompt_similarity != 0` (default **0.1**, `common/common.h:675`), scan
   **all currently-idle slots**, compute longest-common-prefix fraction between the
   incoming prompt and each idle slot's retained previous prompt, pick the best match
   above threshold.
3. Else fall back to plain LRU (oldest `t_last_used` idle slot).
4. If the chosen slot is losing over half its retained context, or was chosen by LRU, its
   old prompt is first spilled to the RAM-backed cross-slot cache (`--cache-ram`), and
   that cache is probed for a better match for the incoming prompt.

**Correction to spec §4 table / §4.2**: prefix-cache matching is **not** strictly
"slot-local, compared only against that slot's own previous prompt." It's best-match over
*all* idle slots' single retained prompt each, **plus** a cross-slot RAM-backed LRU
(on by default, 8 GiB) that can resurrect a prompt into a different slot than the one that
originally computed it. There are effectively `n_slots + cache_ram_capacity` prior
prompts a new request can match against, not just one. The spec's mitigation (unique
random prefix + `cache_prompt: false` + `--cache-ram 0`) is still correct and sufficient
— per-request uniqueness defeats LCP matching regardless of how many prior prompts are
searched — but the mental model needs updating for any documentation/comments we write.

### Context shift — exact behavior, both modes (`pre_decode()`, server-context.cpp:2891–2944)

Triggers only mid-generation, when `slot.prompt.n_tokens() + 1 >= slot.n_ctx`
(server-context.cpp:2895).

- **`--no-context-shift` (current default)**: caught earlier in the token loop
  (server-context.cpp:1907–1913) — generation stops cleanly with `STOP_TYPE_LIMIT`,
  `truncated: true` is reported. Not silent, but `n_gen_actual < n_gen_target`.
- **`--context-shift` (opt-in)**: `pre_decode()` actually shifts — discards
  `n_left/2` tokens from the middle via `seq_rm`/`seq_add` and continues. **This is the
  silent hazard** the spec describes — no response flag indicates discarded tokens;
  `timings` still look normal.
- At submission (not mid-gen), an oversized prompt is rejected outright either way (`ERROR_TYPE_EXCEED_CONTEXT_SIZE`, server-context.cpp:3177–3194).

**Correction to spec §4.1/§11.2**: the silent-discard hazard is real but conditional on
`--context-shift` being explicitly passed — **not the current default**. `llmbench`'s
recommendation should be "do not pass `--context-shift`" (stating `--no-context-shift`
explicitly is harmless/future-proof but redundant against current defaults). Regardless of
the flag, `llmbench` must still preflight `n_prompt + n_depth + n_gen` against
`default_generation_settings.n_ctx` and refuse if it doesn't fit — because even the safe
default (`STOP_TYPE_LIMIT`) silently poisons `t/s` via a short generation, just via early
stop rather than mid-context discard. The existing verification step
(`usage.completion_tokens == n_gen`, §8) already catches this regardless of mechanism.

### No preemption — confirmed exactly (server-context.cpp:2408–2419)

```cpp
server_slot * slot = get_available_slot(task);
if (slot == nullptr) {
    queue_tasks.defer(std::move(task));   // deferred, not evicted from a busy slot
    break;
}
```

No code path in `update_slots()`/`get_available_slot()`/`pre_decode()` stops a
`SLOT_STATE_GENERATING` slot to make room for another task. A slot's occupancy ends only
via natural completion (EOS/`STOP_TYPE_LIMIT`/`STOP_TYPE_WORD`) or explicit client
cancellation. **Confirms the spec's "no preemption" claim exactly.**

### `llama-batched-bench` — path correction: `tools/batched-bench/`, not `tools/llama-batched-bench/`

Triple-nested static grid `for pp: for tg: for pl (=B)` (batched-bench.cpp:128–130), skips
combinations where `n_ctx_req > n_kv_max`. Lock-step: builds one batch covering all `pl`
sequences (optionally sharing a prompt), decodes+times the prefill phase, then the same
for `tg` decode steps. Exact formulas (batched-bench.cpp:229–235):

```cpp
speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
speed_tg = pl*tg / t_tg;
speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;
```

This is the correct reference for Gate D (`-c N` closed-loop throughput at matched
`B=concurrency, PP=n_prompt, TG=n_gen`).

### `examples/parallel/` ("llama-parallel") — path correction: `examples/parallel/`, not `tools/parallel/`

**No arrival-rate/Poisson concept exists in this tool at all.** It maintains a fixed pool
of `n_clients = n_parallel` client sequences; as soon as one finishes, it's immediately
assigned the next prompt from a corpus — a closed-loop pool, functionally similar to
`llmbench`'s own `-c N`, **not** an open-loop generator. Reports per-client speed and a
final aggregate (`Total prompt/gen tokens`, `Total speed (AVG)`, `Cache misses`).

**Correction to spec §4.5/§4.6**: `examples/parallel/` is a fine secondary reference for
closed-loop `-c N` behavior, but it cannot validate open-loop `--request-rate`/
`--burstiness` (§4.6/Gate E) — `vllm bench serve --request-rate` remains the only valid
reference for that mode, exactly as the spec already states for Gate E.

---

## 3. vLLM `benchmarks serve` internals

### Repo-layout correction

`benchmarks/backend_request_func.py` and `benchmarks/benchmark_serving.py` at repo root
are **dead code** — `benchmark_serving.py:5-16` is a deprecation shim that prints "moved
to the vLLM CLI... use: vllm bench serve" and `sys.exit(1)`. The live implementation is:

- `vllm/benchmarks/serve.py` (CLI, orchestration, `BenchmarkMetrics`, `calculate_metrics`,
  `get_request`)
- `vllm/benchmarks/lib/endpoint_request_func.py` (the actual async per-request functions)

Any future work should cite these paths, not `benchmarks/backend_request_func.py`.

### `BenchmarkMetrics` (serve.py:321–352)

Fields: `completed, failed, total_input, total_output, request_throughput,
request_goodput, output_throughput, total_token_throughput, mean/median/std_ttft_ms +
percentiles_ttft_ms, mean/median/std_tpot_ms + percentiles_tpot_ms, mean/median/std_itl_ms
+ percentiles_itl_ms, mean/median/std_e2el_ms + percentiles_e2el_ms,
max_output_tokens_per_s, max_concurrent_requests, rtfx`.

### `calculate_metrics()` (serve.py:556–765) — exact formulas

**TPOT** (serve.py:607–613): `tpot = (outputs[i].latency - outputs[i].ttft) /
(output_len - 1)`, guarded `output_len > 1`, computed once per request (not per token).
Confirms the spec's formula and the worked example in `docs/benchmarking/cli.md:150-151`
(`(180ms-100ms)/(5-1)=20ms/token`).

**ITL** is *not* recomputed here — it's the literal per-chunk gap list produced by the
request function. If a backend bundles multiple tokens per SSE chunk, ITL has fewer
samples than output tokens (TPOT still amortizes over all tokens; ITL does not).

**Stats convention — material divergence from llama-bench**: everything uses **numpy**;
`np.std(...)` is **population** stddev (ddof=0). llama-bench's `stdev()` is **sample**
stddev (ddof=1, N−1). `llmbench`'s `metrics.py` must not reuse one formula for both parity
targets — use llama-bench's sample-stdev formula for the `t/s` parity column, and be
explicit in the detailed report about which convention every other stat column uses.

**Percentiles**: `np.percentile`; default `--metric-percentiles` is `"99"` **only**
(serve.py:1770–1776) — out of the box vLLM reports p99 only (plus median, always computed
separately via `np.median`), not p50/p95/p99 together. `--percentile-metrics` defaults to
`"ttft,tpot,itl"` (generative) / `"e2el"` (pooling).

**Throughput** (serve.py:726–734): `request_throughput = completed/dur_s`,
`output_throughput = sum(actual_output_lens)/dur_s`, `total_token_throughput =
(total_input + sum(actual_output_lens))/dur_s` — aggregate rates over wall-clock run
duration, unlike llama-bench's per-repetition `t/s`.

**`actual_output_lens`** (serve.py:588–604): uses reported `usage.completion_tokens` if
present; **falls back to re-tokenizing the generated text locally** if the backend didn't
report usage, with an explicit comment that this "may inflate the output token count
slightly." If no local tokenizer either, falls back to `1`.

**`max_output_tokens_per_s` / `max_concurrent_requests`** (serve.py:654–724):
reconstructed by cumulative-summing `ttft` then `itl` per request into absolute
timestamps, bucketed into 1-second windows, peak reported. Confirms these are *derived*
metrics computed from raw per-chunk timestamps after the fact — consistent with
`llmbench`'s "never aggregate at collection time" design (non-negotiable #2); this
computation belongs in `metrics.py`, not the runner.

### TTFT timestamping and SSE parsing (`vllm/benchmarks/lib/endpoint_request_func.py`)

**Clock**: `time.perf_counter()` (seconds, not nanoseconds) for latency, `time.time()`
(wall clock) for arrival-pacing deadlines — two different clocks for two different
purposes.

**`async_request_openai_completions`** (`/v1/completions`, lines 158–268) — the important,
easy-to-get-wrong detail:

```python
if choices := data.get("choices"):
    # Note that text could be empty here e.g. for special tokens
    text = choices[0].get("text")
    timestamp = time.perf_counter()
    if not first_chunk_received:
        first_chunk_received = True
        ttft = time.perf_counter() - st
    ...
```

**vLLM's own reference client does NOT skip empty-content first chunks on
`/v1/completions`.** It stamps TTFT on the first message with a non-null `choices` key,
regardless of whether `choices[0]["text"]` is empty — the comment explicitly acknowledges
this. It only skips messages with no `choices` key, SSE comment lines (`:`-prefixed), and
`[DONE]`.

The **chat** variant (`async_request_openai_chat_completions`, lines 342–435) is worse for
our purposes: it gates first-token detection on `if ttft == 0.0` rather than content
non-emptiness, so a role-priming empty-delta chunk (`{"delta":{"role":"assistant"}}`, common
as the very first SSE event from OpenAI-style chat APIs) **will be misattributed as the
first token**, understating true TTFT by one round trip.

**Conclusion / deliberate divergence to document**: the task spec's §3 requirement ("not
the first byte, not an empty role/priming chunk") is *stricter and more correct* than
vLLM's own reference tool. `llmbench`'s Path A client should implement the stricter
behavior (skip empty/None content, stamp only on first non-empty content) and this must be
called out explicitly in our docs/code as a deliberate, documented departure from `vllm
bench serve` — our numbers will not bit-for-bit match vLLM's own tool's TTFT on the chat
endpoint even though both are "correct" by their own definitions. On `/v1/completions`
(our default endpoint per spec §11.5) this rarely bites since real first-token chunks
normally carry non-empty text.

**SSE framing** (`StreamedResponseHandler`, lines 23–62): raw `iter_any()` byte chunks (no
line-buffering assumption from the HTTP client), incremental UTF-8 decoding via
`codecs.getincrementaldecoder("utf-8")` (safe against multi-byte chars split across TCP
reads — a naive per-chunk `.decode("utf-8")` would crash/corrupt), split on `"\n\n"`, and
**a message is only ever yielded once its JSON body parses cleanly** — if `json.loads`
raises on a `data: `-prefixed partial buffer, it waits for more bytes rather than emitting
a malformed message. `[DONE]` is special-cased as always-complete. Both call sites also
`.strip()` raw bytes and drop whitespace-only reads before decoding. `llmbench`'s SSE
parser (a unit-test target per §14) should replicate all of this: incremental UTF-8
decode, `\n\n` framing, JSON-completeness gating, `[DONE]` handling, comment-line skip.

**`output.latency`** = time to the last chunk carrying a `choices` payload — a trailing
`usage`-only chunk does not advance `most_recent_timestamp`, so E2E is not inflated by a
usage trailer.

**Failure handling**: a stream that ends with zero content chunks marks the whole request
`success=False` with an explicit error — never silently recorded as a zero-latency
success.

### Arrival-rate / burstiness — `get_request()` (serve.py:393–504)

```python
theta = 1.0 / (current_request_rate * burstiness)
delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))
```

`shape=burstiness, scale=1/(rate*burstiness)` → mean inter-arrival is always `1/rate`
regardless of burstiness; `burstiness=1` degenerates to `Exponential(1/rate)` (pure
Poisson), `burstiness→∞` is special-cased to a literal constant `1/rate` delay (not
relying on a Gamma variance→0 limit). `request_rate==inf` → all delays are `0`.

**Rescaling step (easy to miss, required for Gate E parity)**: after generating raw
Gamma deltas and cumulative-summing them, if not in ramp-up mode, the *entire cumulative
delay array* is linearly rescaled so the last request's cumulative delay lands exactly on
`total_requests/request_rate` (serve.py:469–485) — raw Gamma sums have a documented "1-2%
gap" from the target total otherwise. Individual inter-arrival gaps in vLLM's tool are
therefore **not raw Gamma samples** — they're raw samples uniformly rescaled by a constant
factor. `llmbench`'s open-loop generator must replicate: sample → cumsum → rescale by
`target_total_delay_s / cumsum[-1]`.

**Pacing**: absolute wall-clock deadlines (`start_ts + delay_ts[i]`) via `time.time()`,
not chained relative sleeps — avoids compounding scheduler-jitter drift.

**Defaults**: `--request-rate` = `inf` (send-all-at-once open loop, not paced at all);
`--burstiness` = `1.0`.

### `--max-concurrency` — a third, hybrid mode not in the spec's binary framing

A semaphore (`serve.py:787,809-810,879-887`) gates in-flight execution to at most
`max_concurrency` **on top of** the open-loop arrival pacing from `get_request()` —
requests are still *dispatched* (created) per the Gamma/rate schedule, but additionally
blocked from *executing* past the semaphore limit. Also caps the underlying
`aiohttp.TCPConnector` pool (`limit=max_concurrency`). This is a genuine third mode
(open-loop arrivals + closed-loop execution cap) that the task spec's §4.6 binary
closed/open framing doesn't call out. **Flagging as a possible scope gap** — recommend
treating it as a stretch-goal composition of `-c` and `--request-rate` rather than adding
scope now; the spec's `mode` column (`closed|open`) should probably grow a third value
(`hybrid`) later if we implement it, but Milestone 6 as specified (pure `-c` xor pure
`--request-rate`+`--burstiness`) doesn't require it.

---

## 4. vLLM scheduler / KVCacheManager / BlockPool / Prometheus metrics

### Preemption — no silent data loss, full recompute (`vllm/v1/core/sched/scheduler.py`)

Trigger: `allocate_slots()` returns `None` (not enough free KV blocks) →
`scheduler.py:589-626` preempts a running request to free blocks and retries. Default FCFS
victim selection is `self.running.pop()` (LIFO — most recently added running request);
`PRIORITY` policy instead picks `max(running, key=(priority, arrival_time))`.
`_preempt_request()` (scheduler.py:1275–1316): frees KV blocks, sets
**`request.num_computed_tokens = 0`**, `status = PREEMPTED`, and **prepends** (not
appends) it to the front of the waiting queue.

Confirms **no silent data loss**: the full prompt + already-generated-output token
sequence lives in the `Request` object itself, not in the freed KV blocks — only the *KV
cache* needs recomputing on resumption. Every token the request already produced is still
delivered once it finishes; nothing is dropped from the output. Architecturally distinct
from llama.cpp's context shift, which discards tokens from the context itself (changes
what the model can attend to). vLLM preemption is a pure latency/performance cost
(bimodal TTFT/ITL for the preempted request), not a correctness hazard — confirms the
spec's framing exactly.

### Global, content-hashed prefix caching — confirmed cross-request

`hash_block_tokens()` (`vllm/v1/core/kv_cache_utils.py:576-602`) chains each block's hash
over `(parent_block_hash, curr_block_token_ids, extra_keys)` — block *k*'s hash commits to
the exact content of blocks `0..k`. `BlockPool.cached_block_hash_to_block`
(`vllm/v1/core/block_pool.py:184`) is **one pool-wide dict**, not per-request state;
`cache_full_blocks()` inserts into it, `get_cached_block()` looks up from it regardless of
which request originally produced the block — confirms genuinely **global** matching, any
request from any client can hit blocks another request produced. Block size:
`DEFAULT_BLOCK_SIZE = 16` (`vllm/config/cache.py:47`), confirmed. **On by default**:
`enable_prefix_caching: bool = True` (`vllm/config/cache.py:93`).

### Chunked prefill — explicit, on by default

`enable_chunked_prefill: bool = True` (`vllm/config/scheduler.py:74`), confirmed default
(can be forced off under specific conditions elsewhere, e.g. `scheduler.py:231` — verify
at implementation time if it matters for a given deployment, but the shipped default is
on).

### Prometheus metric names (`vllm/v1/metrics/loggers.py`)

**Wire-format nuance**: `prometheus_client` Counters are auto-suffixed `_total` on scrape
— a Python `name="vllm:num_preemptions"` appears on `/metrics` as
`vllm:num_preemptions_total`. This reconciles the spec's cited name with the source (not a
typo in either place — same metric, Python-name vs. wire-name).

| Purpose | Python `name=` | Wire name | Location |
|---|---|---|---|
| TTFT histogram | `vllm:time_to_first_token_seconds` | same (Histogram, no suffix) | loggers.py:797 |
| Inter-token latency (per-token) | `vllm:inter_token_latency_seconds` | same | loggers.py:830 |
| Time-per-output-token (per-request) | `vllm:request_time_per_output_token_seconds` | same | loggers.py:860 |
| E2E latency | `vllm:e2e_request_latency_seconds` | same | loggers.py:913 |
| Queue time | `vllm:request_queue_time_seconds` | same | loggers.py:923 |
| Prefill/decode time | `vllm:request_prefill_time_seconds` / `_decode_time_seconds` | same | loggers.py:943,953 |
| Prompt tokens | `vllm:prompt_tokens` | `vllm:prompt_tokens_total` | loggers.py:670 |
| Generation tokens | `vllm:generation_tokens` | `vllm:generation_tokens_total` | loggers.py:705 |
| Preemptions | `vllm:num_preemptions` | `vllm:num_preemptions_total` | loggers.py:661 |
| Local prefix-cache queries/hits | `vllm:prefix_cache_queries` / `_hits` | `..._total` | loggers.py:585,596 |
| External (KV-connector) prefix-cache queries/hits | `vllm:external_prefix_cache_queries` / `_hits` | `..._total` | loggers.py:609,621 |
| Cached prompt tokens | `vllm:prompt_tokens_cached` | `vllm:prompt_tokens_cached_total` | loggers.py:692 |

`vllm:prompt_tokens_cached` is the best signal for `llmbench`'s `cached_tokens` raw-record
field and for Gate F (shared-prefix divergence) — delta it across a repetition.
`vllm:request_queue_time_seconds` / `_prefill_time_seconds` / `_decode_time_seconds` are
useful for decomposing `overhead_ms` on the vLLM side (queueing vs. compute).

---

## 5. Claim-by-claim verdict on task spec §4

| # | Spec §4 claim | Verdict | Detail |
|---|---|---|---|
| 1 | llama.cpp KV cache: flat pre-allocated cells, contiguous, no paging | **Confirmed** (uncontested by any source read; not separately re-derived here, consistent with all server-context.cpp slot/context code read) | — |
| 2 | llama.cpp concurrency: fixed `server_slot` pool, `-np N` static at launch | **Confirmed, with nuance** | `-np` default is `-1` (auto-resolved), not always operator-set; resolved value must be read back from `/props.total_slots`, §2 above |
| 3 | llama.cpp memory ceiling: `n_ctx × n_seq_max` regardless of use → OOM/refusal at modest concurrency | **Confirmed at submission time**; not separately load-tested in this milestone | Oversized-prompt rejection confirmed at server-context.cpp:3177-3194; full OOM-at-launch behavior deferred to Milestone 5/manage-mode testing |
| 4 | llama.cpp preemption: none | **Confirmed exactly**, §2 above | `queue_tasks.defer()`, no eviction path found anywhere |
| 5 | llama.cpp out-of-context: context shift (silent) or fail | **Corrected**: context shift is **opt-in**, disabled by default; default behavior is a clean flagged stop (`STOP_TYPE_LIMIT`), not silent | §2 above — real hazard, but conditional on an explicit flag most users won't pass |
| 6 | llama.cpp prefix caching: slot-granular, matched against that slot's previous prompt, plus `--cache-ram` LRU | **Corrected**: not strictly slot-local — best-LCP search across *all* idle slots, plus cross-slot RAM cache **on by default** (8 GiB, not opt-in) | §2 above |
| 7 | vLLM KV cache: paged blocks (16 tok), on-demand, refcounted, CoW | **Confirmed** (16-tok block size confirmed; refcounting/CoW consistent with BlockPool design, not independently re-derived line-by-line in this pass) | `vllm/config/cache.py:47` |
| 8 | vLLM concurrency: soft `max_num_seqs`, real limit is free KV blocks | **Confirmed** (consistent with `allocate_slots()` returning `None` triggering preemption rather than admission failing outright) | scheduler.py:578-587 |
| 9 | vLLM degrades gracefully via preemption | **Confirmed**, §4 above | Full recompute, no data loss, LIFO/priority victim selection |
| 10 | vLLM out-of-context: preempt + recompute, never silently drops tokens | **Confirmed**, §4 above | `num_computed_tokens=0`, full token sequence retained in `Request` |
| 11 | vLLM prefill chunking: explicit | **Confirmed**, on by default | `vllm/config/scheduler.py:74` |
| 12 | vLLM prefix caching: block-granular, global, content-hashed, on by default | **Confirmed exactly**, §4 above | `kv_cache_utils.py:576-602`, `block_pool.py:184`, `cache.py:93` |
| 13 | llama-bench times in-process `llama_decode()`, no tokenizer/sampler/HTTP | **Confirmed exactly**, §1 above | `test_gen` uses `rand()%n_vocab`, not real sampling |
| 14 | llama-bench `get_ts()`/`avg()`/`stdev()` formulas as assumed | **Confirmed, with corrections**: stdev is sample (N−1) and explicitly 0 at N≤1; `t/s` avg/stdev computed over per-rep values, not avg(tokens)/avg(time) | §1 above |
| 15 | All 5 llama-bench printers as described | **Confirmed, with corrections**: field-name mismatch (`stddev_ns` not `stdev_ns`), markdown column-visibility is global-param-vs-default not per-row-diff, test-name depth suffix has a leading space | §1 above |
| 16 | vLLM `calculate_metrics()` TTFT/TPOT/ITL formulas | **Confirmed for TPOT formula**; **stdev convention differs from llama-bench** (population vs. sample) — must not conflate | §3 above |
| 17 | vLLM correctly skips empty/role-priming first chunk when timestamping TTFT | **Corrected — false.** vLLM's own reference client does NOT skip empty-content chunks on `/v1/completions`, and mis-times TTFT on role-priming deltas on `/v1/chat/completions`. The task spec's stricter behavior is a deliberate, documented improvement over vLLM's own tool, not something we're replicating | §3 above — most important correction in this document |
| 18 | vLLM Gamma-distributed inter-arrival matching spec's assumed parameterization | **Confirmed, with an easy-to-miss addition**: raw Gamma cumsum is rescaled post-hoc to hit the exact target duration | §3 above |
| 19 | `vllm:num_preemptions_total` Prometheus metric name | **Confirmed** (Python name is `vllm:num_preemptions`, wire name after `prometheus_client`'s auto `_total` suffix is `vllm:num_preemptions_total` — same metric, not a typo) | §4 above |
| 20 | llama-batched-bench / llama-parallel are the correct `c>1` comparison points, not llama-bench | **Confirmed for batched-bench** (Gate D); **partially corrected for llama-parallel** — it validates closed-loop `-c` behavior but has no arrival-rate concept, so it cannot validate open-loop mode; only `vllm bench serve --request-rate` can (Gate E) | §2 above |

---

## 6. Consolidated corrections to apply going forward

1. **Path corrections**: `tools/batched-bench/` (not `tools/llama-batched-bench/`);
   `examples/parallel/` (not `tools/parallel/`); vLLM's live serve-bench code is
   `vllm/benchmarks/serve.py` + `vllm/benchmarks/lib/endpoint_request_func.py`, not
   `benchmarks/backend_request_func.py` (dead/deprecated).
2. **Context shift default flipped**: it's off by default in this llama.cpp commit.
   `llmbench` should recommend *not passing* `--context-shift`, and treat the capacity
   preflight (n_prompt+n_depth+n_gen vs. `default_generation_settings.n_ctx`) as required
   regardless of the flag, since even the safe default silently shortens `n_gen_actual`.
3. **Prefix-cache mental model**: llama.cpp matching spans all idle slots + an 8 GiB
   cross-slot RAM cache (on by default), not one slot in isolation. Mitigation
   recommendation (`--cache-ram 0`, `cache_prompt:false`, unique random prefix) is
   unchanged, just the justification/documentation should be accurate.
4. **`/props` capacity read is simpler than assumed**: `default_generation_settings.n_ctx`
   is already the per-slot budget; no division by `total_slots` needed.
5. **`timings` object has 3 more fields than assumed**: `cache_n`,
   `prompt_per_token_ms`, `predicted_per_token_ms` — use `cache_n` directly for
   n_depth-cache verification instead of inferring from `prompt_n`.
6. **stdev convention must not be conflated**: llama-bench parity column uses sample
   stdev (N−1, explicit 0 at N≤1); everywhere else in our detailed report we should state
   which convention is in use (recommend population, matching vLLM/numpy, for the
   non-parity columns, since it's the more common statistical-reporting default — call
   this out explicitly in `docs/reference-notes.md`-derived design notes and in
   `metrics.py` docstrings/field names, e.g. `stddev_ns` vs. a clearly-population-labeled
   field if we ever add one).
7. **TTFT empty-chunk skip is a deliberate divergence from vLLM's own tool**, not
   something we're replicating — document this prominently so nobody "fixes" our stricter
   behavior to match vLLM's looser one, and expect a small, explainable non-zero delta
   vs. `vllm bench serve` TTFT specifically on `--endpoint chat` (Gate C/E should budget
   for this on the chat endpoint; the completions endpoint, our default, is unaffected in
   practice).
8. **Gamma arrival generator needs the post-hoc rescale step** for Gate E to land within
   tolerance — sample, cumsum, then rescale the whole array by
   `target_total_delay_s / cumsum[-1]`.
9. **`--max-concurrency` is a third hybrid mode** in vLLM (open-loop dispatch + closed-loop
   execution cap) not covered by the spec's closed/open binary. Treat as an
   explicitly-out-of-scope stretch goal for Milestone 6 rather than silently adding a
   third `mode` value now.
10. **Markdown test-name depth suffix** is `" @ d{n}"` (leading space before `@`), not
    `@d{n}` — match the spec's illustrative example table format only loosely; match
    llama-bench's actual string exactly.
11. **Field name casing**: llama-bench's own CSV/JSON/SQL output uses `stddev_ns`/
    `stddev_ts` (double-d), despite the C++ method names being `stdev_ns()`/`stdev_ts()`.
    Match the double-d spelling in our own printers for parity-mode output.

---

## 7. Empirical verification against a live server (weak/CPU-only hardware)

Source-reading alone can miss build-time or version drift, so before writing any
`llmbench` code we built the pinned llama.cpp commit CPU-only (`cmake -B build
-DGGML_NATIVE=ON -DLLAMA_CURL=ON -DLLAMA_BUILD_UI=OFF` — the UI target was disabled
because it tries to fetch a prebuilt web bundle from an HF bucket and hung for 15+
minutes on this connection; irrelevant to API testing) on the target machine (AMD Ryzen 7
5700U, 16 threads, 22 GiB RAM, **no discrete GPU — CPU backend only**) and ran it against
`Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M, 468 MiB) — chosen deliberately small given the
hardware. Model saved at `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (gitignored — see
below).

**Native `llama-bench` baseline** (`-p 128 -n 32 -r 3 -t 8`, CPU backend):

```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CPU        |       8 |           pp128 |        209.07 ± 1.60 |
| qwen2 1B Q4_K - Medium         | 462.96 MiB |   630.17 M | CPU        |       8 |            tg32 |         62.14 ± 1.58 |
```
Confirms the markdown printer's conditional-column logic exactly as documented in §1:
`backend` and `threads` (display name for `n_threads`) both appear because backend is CPU
(force-show rule), non-CUDA `ngl` column is absent, table formatting/alignment matches.

**`llama-server`** launched with `-c 4096 -np 2 -t 8 --metrics` (all other flags at
compiled defaults, i.e. `--context-shift` NOT passed, `--cache-ram` at its default 8192
MiB). All of the following were confirmed byte-for-byte against the source-derived
claims in §2:

- `GET /health` → `{"status":"ok"}`, `Server: llama.cpp` response header.
- `GET /props` → `default_generation_settings.n_ctx = 2048` (= 4096 ÷ 2 slots, already
  the per-slot budget, no division needed on our side) and `total_slots: 2`.
- `GET /slots` → array of `{id, n_ctx, speculative, is_processing}`, one object per slot,
  both idle (`is_processing: false`).
- `POST /completion` with a **raw token-ID array prompt** (`"prompt": [791, 4062, ...]`,
  bypassing the tokenizer entirely), `ignore_eos: true`, `cache_prompt: false`,
  `n_predict: 16` → `tokens_predicted: 16` exactly, `stop_type: "limit"`, `tokens_evaluated:
  9` matching the 9-token input array. Confirms §8's Path-1 token-ID-array approach works
  unmodified over HTTP against a live server.
- `timings` object on that response: `{cache_n: 0, prompt_n: 9, prompt_ms: 59.178,
  prompt_per_token_ms: 6.575, prompt_per_second: 152.08, predicted_n: 16, predicted_ms:
  252.656, predicted_per_token_ms: 15.791, predicted_per_second: 63.33}` — every field
  name matches §2's source-derived list exactly, including the three fields (`cache_n`,
  `prompt_per_token_ms`, `predicted_per_token_ms`) the task spec's guess had missed.
  `cache_n: 0` confirms `cache_prompt:false` actually disabled cache reuse.
- `GET /metrics` → all eleven `llamacpp:*` metric names from §2 present and populated
  (`prompt_tokens_total: 9`, `tokens_predicted_total: 16`, `requests_deferred: 0`, etc.),
  **no preemption counter**, exactly as predicted.
- `POST /tokenize` then `POST /detokenize` round-trips exactly: `"The quick brown fox
  jumps over the lazy dog"` → `[785,3974,13876,38835,34208,916,279,15678,5562]` → back to
  the identical string.
- **Oversized prompt at submission** (2200 tokens against a 2048-token per-slot budget,
  `cache_prompt:false`) → HTTP 400,
  `{"error":{"code":400,"message":"request (2200 tokens) exceeds the available context
  size (2048 tokens), try increasing it","type":"exceed_context_size_error",
  "n_prompt_tokens":2200,"n_ctx":2048}}` — confirms the clean-rejection-at-submission
  path from §2 exactly, including the field names our capacity-preflight error messages
  should probably echo (`n_prompt_tokens`, `n_ctx`).
- **New finding not covered by the source-reading pass**: streaming
  `POST /v1/completions` with `"stream": true` — the **first** SSE chunk already carries
  non-empty `choices[0].text` (no empty role/priming chunk on this endpoint, unlike the
  vLLM chat-completions case documented in §3), and the **final** chunk (the one with
  `finish_reason:"length"` and empty text) carries the `usage` object *and* the full
  `timings` object inline, in the same SSE stream. This means `llmbench`'s Path B
  (server-reported) measurement on llama.cpp can be read directly off the terminal chunk
  of a single streamed request — no separate non-streaming round-trip needed to get both
  Path A (client timestamps) and Path B (server timings) from one request.

**Non-finding worth recording**: the first attempt to start `llama-server` on port 8090
failed with `couldn't bind HTTP server socket` because an unrelated pre-existing service
(`server: uvicorn`, returning a `{"status":"ok","weaviate_ready":true}` payload — not a
llama.cpp response, no such field exists in llama.cpp's source) was already listening on
that port in this sandbox. Not a llama.cpp behavior and not a security concern — just a
port collision with unrelated sandbox infrastructure, resolved by picking a free port
(8877). Recorded here so it isn't mistaken for a llama.cpp quirk later.

**vLLM was not empirically tested in this Milestone-1 pass** (superseded below — see
docs/validation.md for the full live results). Unlike llama.cpp's CPU backend (a
first-class, well-supported build target), vLLM's CPU backend is a secondary target with
its own build path (oneDNN/AVX2 requirements, longer build times, documented rough edges)
and this machine has no GPU at all, so it was deferred past Milestone 1.

Binaries built: `reference/llama.cpp/build/bin/{llama-server,llama-bench,llama-batched-bench,llama-parallel}`
(gitignored via the existing `build/` rule in `.gitignore`). Model file at
`models/qwen2.5-0.5b-instruct-q4_k_m.gguf` — gitignored (`models/`, `*.gguf`), since model
weights should never be committed.

### Addendum: vLLM CPU empirical verification (post-course-correction pass)

vLLM `0.26.0+cpu` (prebuilt wheel, not the pinned source commit — source build requires
gcc>=12.3, only 11.4.0 available; a deliberate, documented scope/time tradeoff) was brought
up on this same machine serving `Qwen/Qwen2.5-0.5B-Instruct` (HF safetensors, bf16,
`--enforce-eager --max-model-len 2048`, `LD_PRELOAD` tcmalloc+iomp5 per vLLM's CPU docs).
Model load + engine warmup/compilation took several minutes even with `--enforce-eager`
(no CUDA graphs to skip, but still an inductor/oneDNN warmup pass on first use) — confirms
the course-correction's point that vLLM's warm-up cost is structurally different from
llama.cpp's and motivates convergence-based warmup (course_correct.txt §2.9) over a fixed
count.

Confirmed empirically, all matching the source-derived claims above:
- `GET /health` -> 200. `GET /v1/models` -> `data[0].max_model_len` present (`2048`) — the
  one parity-relevant field vLLM *does* expose over its OpenAI-compatible HTTP surface;
  everything else in `Capacity` (kv dtype, attention backend, max_num_seqs, prefix-cache
  state, chunked-prefill state) is genuinely unprobeable over plain HTTP, confirming the
  gap flagged in `llmbench/backends/vllm.py`'s module docstring — not a Milestone-1
  oversight, a real limitation of vLLM's public API.
- `GET /version` -> `{"version": "0.26.0"}` (undocumented in the original source-reading
  pass; confirmed live, a stable/long-standing part of vLLM's OpenAI server).
- `GET /metrics` -> real `vllm:*` series present, including `_sum`/`_count` pairs for
  `vllm:request_prefill_time_seconds` / `_decode_time_seconds`, usable for a delta-scraped
  Path B (implemented in `runner.py`'s `_attach_vllm_metrics_delta`) — necessarily an
  instance-level aggregate, not a true per-request raw record, since Prometheus histograms
  are server-side pre-aggregated by construction (a hard limit of vLLM's HTTP surface, not
  a shortcut we took).
- Streaming `/v1/completions` never carries an inline timings-equivalent object (confirmed
  live, matching the source-reading finding above) — Path B is `/metrics`-only for vLLM,
  never inline.
- `--backend auto` correctly distinguished the two live servers (llama.cpp via `/props`
  shape, vLLM via `/metrics` containing `vllm:` series).

Live cross-backend results (llmbench itself, `llmbench parity`, `llmbench quality`, and
`llmbench calibrate` all run against real servers, not just the fake SSE server) are in
[docs/validation.md](validation.md) — includes two real bugs this live testing caught and
fixed (an empty-prompt 400 on tg-only instances; a `parity.json` write failing on a
non-existent `--out-dir`) and one heuristic (`cache_suspected`) found to false-positive
under concurrent dispatch and scoped down accordingly.
