"""Orchestration: capacity preflight, convergence warmup, reps, closed/open-loop dispatch,
raw record emission. No aggregation here (non-negotiable #2) -- that's metrics.py, run later
from the sink's raw records.
"""
from __future__ import annotations

import asyncio
import datetime
import random
import sys
import time
import uuid
from dataclasses import dataclass

from . import metrics
from .arrivals import generate_arrival_delays
from .backends.base import Backend
from .backends.vllm import METRIC_PREEMPTIONS_TOTAL
from .config import CmdParams, Instance
from .prompts import generate_depth_prefix_tokens, generate_prompt_tokens
from .records import RawRecord


class CapacityError(RuntimeError):
    """Raised when an instance does not fit the server's advertised per-request context, or
    exceeds slot capacity, and --force was not passed. Distinguished from a benchmark bug:
    this is 'did not fit', a legitimate outcome (spec §7)."""


DEFAULT_VOCAB_SIZE = 32000  # fallback only; real n_prompt is exact regardless (token IDs are
                             # taken mod vocab_size), used only when the backend doesn't expose one


async def capacity_preflight(backend: Backend, instance: Instance, *, force: bool) -> dict:
    cap = await backend.capacity()
    budget = instance.n_prompt + instance.n_depth + instance.n_gen
    notes = {}
    if cap.per_request_ctx is not None:
        notes["per_request_ctx"] = cap.per_request_ctx
        if budget > cap.per_request_ctx and not force:
            raise CapacityError(
                f"instance requires {budget} tokens of context "
                f"(n_prompt={instance.n_prompt}+n_depth={instance.n_depth}+n_gen={instance.n_gen}) "
                f"but the server's per-request budget is {cap.per_request_ctx}. Pass --force to run anyway "
                f"(expect truncation/context-shift risk, not a clean result)."
            )
    if cap.max_concurrent is not None:
        notes["max_concurrent"] = cap.max_concurrent
        if instance.concurrency > cap.max_concurrent and not force:
            raise CapacityError(
                f"-c {instance.concurrency} exceeds the server's slot count "
                f"({cap.max_concurrent}); this would measure your own queue, not the engine. "
                f"Pass --force to run anyway."
            )
    return notes


async def _send_one(
    backend: Backend,
    *,
    run_id: str,
    instance_id: str,
    rep_idx: int,
    instance: Instance,
    model: str,
    token_ids: list[int],
    request_max_tokens: int,
    request_idx: int,
    preemptions_before: float | None,
) -> RawRecord:
    # n_gen_target on the record is the *test's* target (may be 0 for a pp-only test);
    # request_max_tokens is what's actually sent over HTTP (>=1 -- see next_prompt_tokens'
    # analogous n_prompt fix: an empty/zero-length request is a 400 on both backends, but the
    # test's label must not be corrupted by that HTTP-layer minimum).
    rec = RawRecord(
        run_id=run_id, instance_id=instance_id, rep_idx=rep_idx,
        ts_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        backend=backend.name, model=model,
        n_prompt_target=instance.n_prompt, n_gen_target=instance.n_gen, n_depth=instance.n_depth,
        concurrency=instance.concurrency, shared_prefix_n=instance.shared_prefix,
    )
    n_content_chunks = 0
    n_gen_actual = 0
    last_ts_ns: int | None = None
    text_parts: list[str] = []
    try:
        t_send = time.perf_counter_ns()
        rec.t_send_ns = t_send
        async for chunk in backend.complete_stream(
            token_ids=token_ids, max_tokens=request_max_tokens, ignore_eos=True, model=model,
            cache_prompt=(instance.shared_prefix > 0 or instance.n_depth > 0),
        ):
            now = time.perf_counter_ns()
            if chunk.text:  # spec §3/§11.6: first *non-empty content* chunk, not first byte
                if rec.t_first_token_ns is None:
                    rec.t_first_token_ns = now
                elif last_ts_ns is not None:
                    rec.itl_ns.append(now - last_ts_ns)
                last_ts_ns = now
                n_content_chunks += 1
                n_gen_actual += 1
                text_parts.append(chunk.text)
            if chunk.usage:
                rec.n_prompt_actual = chunk.usage.get("prompt_tokens", rec.n_prompt_actual)
                rec.n_gen_actual = chunk.usage.get("completion_tokens", rec.n_gen_actual)
                rec.cached_tokens = chunk.usage.get("cached_tokens", rec.cached_tokens)
            if chunk.server_timings:
                st = chunk.server_timings
                rec.server_cache_n = st.get("cache_n")
                rec.server_prompt_n = st.get("prompt_n")
                rec.server_prompt_ms = st.get("prompt_ms")
                rec.server_predicted_n = st.get("predicted_n")
                rec.server_predicted_ms = st.get("predicted_ms")
        rec.t_end_ns = time.perf_counter_ns()
        rec.t_last_token_ns = last_ts_ns
        if rec.n_gen_actual is None:
            rec.n_gen_actual = n_gen_actual
        if rec.n_prompt_actual is None:
            rec.n_prompt_actual = len(token_ids)
        rec.http_status = 200
    except Exception as e:  # noqa: BLE001 -- a failed request is a valid, recorded outcome
        rec.t_end_ns = time.perf_counter_ns()
        rec.error = f"{type(e).__name__}: {e}"
        rec.http_status = getattr(e, "response", None) and getattr(e.response, "status_code", None)

    if rec.error is None:
        if rec.n_prompt_actual is not None and rec.n_prompt_actual != len(token_ids):
            rec.flags.append("prompt_token_mismatch")
        if rec.n_gen_actual is not None and rec.n_gen_actual < request_max_tokens:
            rec.flags.append("context_shift_risk")
        if instance.n_depth > 0:
            if rec.server_cache_n is None or rec.server_cache_n < instance.n_depth:
                rec.flags.append("depth_unverified")

    if backend.supports_vllm_metrics and preemptions_before is not None:
        try:
            after = await backend.metrics_snapshot()
            after_val = after.get(METRIC_PREEMPTIONS_TOTAL)
            if after_val is not None:
                rec.preemptions_delta = after_val - preemptions_before
        except Exception:  # noqa: BLE001 -- metrics scrape failure must not fail the request
            pass

    return rec


@dataclass
class WarmupResult:
    iterations: int
    converged: bool


async def warmup_convergence(
    send_fn, *, no_warmup: bool, warmup_fixed: int | None, threshold: float = 0.02, k: int = 5, hard_cap: int = 50,
    on_iteration=None,
) -> WarmupResult:
    """course_correct.txt §2.9: repeat, discard, until rolling stddev over the last k runs
    < threshold*mean, or a hard cap. Iteration count is itself a reportable finding.

    on_iteration(i, hard_cap, rec), if given, fires after every single request -- this loop
    has no other output, and on slow backends (large prompts, CPU-only) each iteration can
    take minutes; without a progress signal this looks indistinguishable from a hang.
    """
    if no_warmup:
        return WarmupResult(iterations=0, converged=True)
    if warmup_fixed is not None:
        for i in range(warmup_fixed):
            rec = await send_fn()
            if on_iteration:
                on_iteration(i + 1, warmup_fixed, rec)
        return WarmupResult(iterations=warmup_fixed, converged=True)

    history: list[float] = []
    for i in range(hard_cap):
        rec = await send_fn()
        if on_iteration:
            on_iteration(i + 1, hard_cap, rec)
        n_tokens = (rec.n_prompt_actual or 0) + (rec.n_gen_actual or 0)
        t_ns = (rec.t_end_ns or 0) - (rec.t_send_ns or 0)
        if rec.error or t_ns <= 0 or n_tokens <= 0:
            continue
        history.append(1e9 * n_tokens / t_ns)
        if len(history) >= k:
            window = history[-k:]
            mean = sum(window) / k
            if mean > 0 and metrics.sample_stddev(window) / mean < threshold:
                return WarmupResult(iterations=i + 1, converged=True)
    return WarmupResult(iterations=hard_cap, converged=False)


async def run_instance(
    backend: Backend,
    instance: Instance,
    params: CmdParams,
    *,
    run_id: str,
    sink,
    run_salt: int,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    bos_token_id: int | None = None,
    workload_writer=None,
) -> list[RawRecord]:
    await capacity_preflight(backend, instance, force=params.force)
    instance_id = str(uuid.uuid4())
    rng = random.Random(run_salt)
    request_counter = [0]

    def next_prompt_tokens() -> list[int]:
        request_counter[0] += 1
        idx = request_counter[0]
        # A tg-only test (n_prompt target 0) still needs >=1 real prompt token to seed
        # generation -- an empty prompt array is a 400 on both backends. n_prompt_target
        # stays 0 (preserves the tg{n} test label); n_prompt_actual reflects what was really
        # sent (found live against llama-server: empty "prompt": [] -> HTTP 400).
        effective_n_prompt = instance.n_prompt if instance.n_prompt > 0 else 1
        return generate_prompt_tokens(
            n_prompt=effective_n_prompt, shared_prefix_n=instance.shared_prefix,
            run_salt=run_salt, request_idx=idx, vocab_size=vocab_size, rng=rng,
            bos_token_id=bos_token_id,
        ), idx

    request_max_tokens = instance.n_gen if instance.n_gen > 0 else 1

    async def send_next(rep_idx: int) -> RawRecord:
        token_ids, idx = next_prompt_tokens()
        if instance.n_depth > 0:
            prefix = generate_depth_prefix_tokens(
                n_depth=instance.n_depth, run_salt=run_salt, request_idx=idx,
                vocab_size=vocab_size, rng=rng,
            )
            async for _ in backend.complete_stream(
                token_ids=prefix, max_tokens=1, ignore_eos=True, model=instance.model,
                cache_prompt=True,
            ):
                pass
            token_ids = prefix + token_ids
        if workload_writer is not None:
            from .prompts import WorkloadItem

            workload_writer.write(WorkloadItem(
                request_idx=idx, instance_id=instance_id, rep_idx=rep_idx,
                parity_mode=params.parity_mode or "", token_ids=token_ids,
            ))
        preemptions_before = None
        if backend.supports_vllm_metrics:
            try:
                snap = await backend.metrics_snapshot()
                preemptions_before = snap.get(METRIC_PREEMPTIONS_TOTAL)
            except Exception:  # noqa: BLE001
                pass
        return await _send_one(
            backend, run_id=run_id, instance_id=instance_id, rep_idx=rep_idx, instance=instance,
            model=instance.model, token_ids=token_ids, request_max_tokens=request_max_tokens,
            request_idx=idx, preemptions_before=preemptions_before,
        )

    show_progress = params.progress or params.verbose
    test_name = instance.test_name()

    def _warmup_progress(i: int, cap: int, rec: RawRecord) -> None:
        if not show_progress:
            return
        status = "error" if rec.error else "ok"
        elapsed_s = (rec.t_end_ns - rec.t_send_ns) / 1e9 if rec.t_end_ns and rec.t_send_ns else 0.0
        print(f"[{test_name}] warmup {i}/{cap} ({status}, {elapsed_s:.1f}s)", file=sys.stderr, flush=True)

    warmup = await warmup_convergence(
        lambda: send_next(-1), no_warmup=params.no_warmup, warmup_fixed=params.warmup_fixed,
        on_iteration=_warmup_progress,
    )

    # Taken after the warm-up, not before: vLLM's server-side numbers are a /metrics delta over
    # the whole window, so a baseline from before the warm-up folded the (cold, slower) warm-up
    # requests into every measured rep -- server t/s came out low and overhead_ms negative.
    vllm_metrics_before = None
    if backend.supports_vllm_metrics and params.measure in ("server", "both"):
        try:
            vllm_metrics_before = await backend.metrics_snapshot()
        except Exception:  # noqa: BLE001 -- Path B is best-effort, never blocks the run
            pass

    records: list[RawRecord] = []
    if instance.concurrency <= 1 and not params.request_rate:
        for rep in range(params.reps):
            rec = await send_next(rep)
            if show_progress:
                status = "error" if rec.error else "ok"
                elapsed_s = (rec.t_end_ns - rec.t_send_ns) / 1e9 if rec.t_end_ns and rec.t_send_ns else 0.0
                print(f"[{test_name}] rep {rep + 1}/{params.reps} ({status}, {elapsed_s:.1f}s)", file=sys.stderr, flush=True)
            records.append(rec)
            if params.delay > 0:
                await asyncio.sleep(params.delay)
    else:
        completed = [0]

        async def _tracked(rep: int, coro) -> RawRecord:
            rec = await coro
            completed[0] += 1
            if show_progress:
                status = "error" if rec.error else "ok"
                elapsed_s = (rec.t_end_ns - rec.t_send_ns) / 1e9 if rec.t_end_ns and rec.t_send_ns else 0.0
                print(f"[{test_name}] {completed[0]}/{params.reps} completed (rep {rep} {status}, {elapsed_s:.1f}s)",
                      file=sys.stderr, flush=True)
            return rec

        if params.request_rate:
            # Open loop: dispatch on a Gamma-paced schedule, execute concurrently, gather at end.
            delays = generate_arrival_delays(params.reps, params.request_rate, params.burstiness, rng)
            deadline = time.monotonic()
            tasks = []
            for rep, d in enumerate(delays):
                deadline += d
                now = time.monotonic()
                if deadline > now:
                    await asyncio.sleep(deadline - now)
                tasks.append(asyncio.create_task(_tracked(rep, send_next(rep))))
            results = await asyncio.gather(*tasks)
            for rec in results:
                rec.load_mode = "open"
                rec.request_rate = params.request_rate
                rec.burstiness = params.burstiness
                records.append(rec)
        else:
            # Closed loop, concurrency > 1: keep exactly `concurrency` requests in flight
            # until `reps` total requests have been sent (spec §4.6).
            sem = asyncio.Semaphore(instance.concurrency)

            async def bounded(rep: int) -> RawRecord:
                async with sem:
                    return await send_next(rep)

            tasks = [asyncio.create_task(_tracked(rep, bounded(rep))) for rep in range(params.reps)]
            results = await asyncio.gather(*tasks)
            for rec in results:
                rec.slots_busy_at_send = instance.concurrency
                records.append(rec)

    for rec in records:
        rec.warmup_iterations = warmup.iterations

    if vllm_metrics_before is not None and records:
        # vLLM has no per-response timings (docs/reference-notes.md §3); Path B here is a
        # single before/after /metrics delta over the whole instance, not true per-request
        # data -- Prometheus histograms are server-side pre-aggregated by construction, so
        # "one raw record per request" (non-negotiable #2) is not obtainable for vLLM's
        # server path at all. Attached to the *last* record only, so aggregate_server() sees
        # exactly one server-side sample for the instance rather than a fabricated N-way
        # duplicate -- see metrics.py's _server_ts_samples.
        try:
            after = await backend.metrics_snapshot()
            _attach_vllm_metrics_delta(records[-1], vllm_metrics_before, after)
        except Exception:  # noqa: BLE001 -- Path B is best-effort, never blocks the run
            pass

    for rec in records:
        sink.write(rec)

    return records


def _attach_vllm_metrics_delta(rec: RawRecord, before: dict, after: dict) -> None:
    def delta(name: str) -> float | None:
        a, b = before.get(name), after.get(name)
        if a is None or b is None:
            return None
        return b - a

    prefill_sum, prefill_cnt = delta("vllm:request_prefill_time_seconds_sum"), delta("vllm:request_prefill_time_seconds_count")
    decode_sum, decode_cnt = delta("vllm:request_decode_time_seconds_sum"), delta("vllm:request_decode_time_seconds_count")
    prompt_tok, gen_tok = delta("vllm:prompt_tokens_total"), delta("vllm:generation_tokens_total")
    cached_tok = delta("vllm:prompt_tokens_cached_total")

    if prefill_cnt or decode_cnt:
        # These are means over every request in the window, not this record's own timings.
        rec.flags.append(metrics.SERVER_TIMINGS_TRIAL_MEAN)
    if prefill_cnt and prefill_cnt > 0 and prefill_sum is not None:
        rec.server_prompt_ms = (prefill_sum / prefill_cnt) * 1000.0
    if decode_cnt and decode_cnt > 0 and decode_sum is not None:
        rec.server_predicted_ms = (decode_sum / decode_cnt) * 1000.0
    if prompt_tok is not None and prefill_cnt:
        rec.server_prompt_n = round(prompt_tok / prefill_cnt)
    if gen_tok is not None and decode_cnt:
        rec.server_predicted_n = round(gen_tok / decode_cnt)
    if cached_tok is not None and prefill_cnt:
        # Per request, like server_prompt_n above -- NOT the raw delta. `vllm:prompt_tokens_
        # cached_total` is a counter over the whole trial, so the undivided value lands in a
        # per-request field as a trial-wide sum: measured here, a 1024-token prompt with a
        # 896-token shared prefix reported 34048 cached tokens on one record (896 x 38
        # requests) instead of 896. Every other quantity in this function is already divided
        # by the request count; this one was not.
        rec.cached_tokens = round(cached_tok / prefill_cnt)
