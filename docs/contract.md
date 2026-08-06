# The `llmbench` measurement contract

Every result this tool emits satisfies this contract, identically on every backend, or is refused.
This document is the north star for `llmbench`. It supersedes any acceptance criterion elsewhere
that asks the tool to match `llama-bench`'s or `vllm bench serve`'s numbers directly — see
"What this is not" below.

## Why this document exists

`llama-bench` and `vllm bench serve` do not measure the same thing, and neither is a valid
cross-backend target:

- `llama-bench` links `libllama` and times `llama_decode()` in-process. No server, no scheduler,
  no admission, no HTTP, no sampling, no tokenization. It measures a decode-shape ceiling for one
  sequence.
- `vllm bench serve` drives an open-loop arrival process against an HTTP server and measures
  client-observed TTFT/TPOT/ITL through the full continuous-batching scheduler.
- vLLM has no equivalent of `llama-bench` — it has exactly one execution path, and it always goes
  through the scheduler. There is no "just the decode loop" to time.

So `llmbench` defines its own measurement boundary, applies it identically to both backends, and
uses the native tools only as calibration anchors that bound and explain its numbers.

## The timed region

**From the last byte of the request leaving the client socket, to the last byte of the final
content chunk arriving.**

Nothing about either server's internals defines this boundary. The boundary is the wire, because
the wire is the only place that is definitionally identical on both backends.

**Included, by construction:**
admission/queue, tokenization, prefill, decode, sampling, detokenization, serialization, transport.

**Excluded, by construction:**
connection setup (pre-established, warm), model load, compile/graph capture, DNS.

## Initial condition

Before the timer starts, on both backends:

1. Model fully resident; all lazy init complete (convergence warmup, §2.9 of the course
   correction — see `runner.py`'s convergence-warmup loop).
2. KV cache state for the measured sequence is **defined** — either empty (default) or a verified
   prefix of exactly `d` tokens (depth tests). Never "whatever was left over."
3. No other in-flight work, unless concurrency is the variable under test.
4. Clocks settled; no thermal transient.

## Sub-measurements

TTFT, ITL, TPOT, and E2E are derived from client timestamps by **one implementation**, in
`metrics.py`, from the same raw record schema, for both backends. There is no per-backend metric
code. A `metrics.py` that branches on `backend` has broken the contract — this is enforced by a
unit test that asserts no backend identifier is referenced in that module.

## `src=server` numbers are diagnostics, not results

llama.cpp's `timings` object and vLLM's `/metrics` histograms are measured at different points in
different pipelines. They are not comparable to each other. They are only comparable to
*themselves*, across runs of the same backend — useful for regression tracking and for explaining
a gap, never for a cross-backend comparison table.

## Parity is a precondition, not a footnote

A number produced under this contract is only a valid cross-backend comparison if every parity
axis (weights/numerics, KV cache dtype, context capacity, KV memory budget, batch/admission
shaping, attention backend, prefix-cache state, sampling, warm state) is verified `matched`,
`unverifiable` axes downgrade confidence, and `mismatched` axes force refusal. See `parity.py` and
`llmbench parity --a <url> --b <url>`.

The tool may print exactly one headline cross-backend claim, and only if every parity axis is
`matched`:

> Under [contract], with [parity axes], backend A achieved X tok/s and backend B achieved Y tok/s
> at [quality divergence].

Otherwise it prints:

> Not a valid comparison. Unmatched axes: [...]. Per-backend results below are individually valid
> and jointly meaningless.

## What this is not

- Not a `llama-bench` clone. Surface-level output parity (llama-bench-shaped markdown) is a *view*
  (`-o md-llamabench`), not a goal and not a gate.
- Not scored against "match `llama-bench` within 5%". That comparison is meaningless for vLLM,
  which has no equivalent code path, and is retained only as `llmbench calibrate` — a diagnostic
  report of the HTTP+scheduler tax, not a pass/fail gate.
- Not a tool that produces a single-number "speedup" without a quality-divergence check
  (`llmbench quality`) when weights differ (`--parity-mode native`).

## Calibration anchors, not targets

- `llama-bench` (native, in-process) — bounds the decode-shape ceiling llama.cpp's HTTP path pays
  tax against. Used by `llmbench calibrate`.
- `vllm bench serve` — measured at the same point in the pipeline as `llmbench`'s client path
  (open-loop, HTTP, client-observed). This is the primary external anchor (gate G5): `llmbench`
  client numbers should agree with it within a few percent at matched rate/burstiness/dataset.
