"""The integration test spec §14 calls out as the most valuable in the suite: a fake
OAI-compatible SSE server with injectable delays, asserting a known TTFT/ITL are recovered
within tolerance, plus the four required stream-shape simulations.
"""
import math

import pytest

from llmbench.backends.llamacpp import LlamaCppBackend
from tests.fake_server.server import FakeServerConfig, FakeSSEServer

TOL_MS = 25.0  # event-loop scheduling jitter tolerance on a shared/virtualized CI box


async def _collect(backend, token_ids, max_tokens):
    import time

    t_send = time.perf_counter_ns()
    t_first = None
    itl_ns = []
    last_ts = None
    n_gen = 0
    usage = None
    timings = None
    try:
        async for chunk in backend.complete_stream(token_ids=token_ids, max_tokens=max_tokens, ignore_eos=True, model="fake"):
            now = time.perf_counter_ns()
            if chunk.text:
                if t_first is None:
                    t_first = now
                elif last_ts is not None:
                    itl_ns.append(now - last_ts)
                last_ts = now
                n_gen += 1
            if chunk.usage:
                usage = chunk.usage
            if chunk.server_timings:
                timings = chunk.server_timings
    except Exception:  # noqa: BLE001 -- a dropped connection ends the stream, same as runner._send_one
        pass
    t_end = time.perf_counter_ns()
    return {
        "ttft_ms": (t_first - t_send) / 1e6 if t_first else None,
        "itl_ms": [x / 1e6 for x in itl_ns],
        "n_gen": n_gen,
        "usage": usage,
        "timings": timings,
        "e2e_ms": (t_end - t_send) / 1e6,
    }


@pytest.mark.asyncio
async def test_recovers_known_ttft_and_itl():
    config = FakeServerConfig(ttft_ms=100.0, itl_ms=10.0, n_tokens=10)
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1, 2, 3], max_tokens=10)
        finally:
            await backend.close()

    assert math.isclose(result["ttft_ms"], 100.0, abs_tol=TOL_MS)
    mean_itl = sum(result["itl_ms"]) / len(result["itl_ms"])
    assert math.isclose(mean_itl, 10.0, abs_tol=TOL_MS)
    assert result["n_gen"] == 10
    assert result["usage"]["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_empty_first_chunk_is_not_mistaken_for_ttft():
    """spec §3/§11.6: timestamp the first *non-empty content* chunk, not the first byte."""
    config = FakeServerConfig(ttft_ms=100.0, itl_ms=5.0, n_tokens=3, empty_first_chunk=True)
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1, 2, 3], max_tokens=3)
        finally:
            await backend.close()
    # If the empty chunk had been mis-timestamped as TTFT, ttft_ms would be ~0, not ~100.
    assert math.isclose(result["ttft_ms"], 100.0, abs_tol=TOL_MS)


@pytest.mark.asyncio
async def test_stalled_mid_stream_is_visible_as_a_large_itl_gap():
    config = FakeServerConfig(ttft_ms=20.0, itl_ms=5.0, n_tokens=6, stall_at_token=2, stall_ms=150.0)
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1, 2, 3], max_tokens=6)
        finally:
            await backend.close()
    assert max(result["itl_ms"]) > 100.0
    assert result["n_gen"] == 6


@pytest.mark.asyncio
async def test_truncated_generation_yields_short_output_not_a_crash():
    config = FakeServerConfig(ttft_ms=10.0, itl_ms=5.0, n_tokens=10, truncate_at_token=3)
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1, 2, 3], max_tokens=10)
        finally:
            await backend.close()
    assert result["n_gen"] == 3
    assert result["usage"] is None  # connection dropped before the final usage-bearing chunk


@pytest.mark.asyncio
async def test_cached_prefix_response_reports_cached_tokens_and_fast_ttft():
    config = FakeServerConfig(ttft_ms=100.0, itl_ms=5.0, n_tokens=4, cached_tokens=12, n_prompt=16)
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1] * 16, max_tokens=4)
        finally:
            await backend.close()
    assert result["usage"]["cached_tokens"] == 12
    assert result["timings"]["cache_n"] == 12
    assert result["ttft_ms"] < 50.0  # far below the configured 100ms non-cached TTFT


@pytest.mark.asyncio
async def test_llamacpp_server_timings_arrive_inline_on_final_chunk():
    config = FakeServerConfig(ttft_ms=10.0, itl_ms=5.0, n_tokens=5, mode="llamacpp")
    async with FakeSSEServer(config) as server:
        backend = LlamaCppBackend(server.base_url)
        try:
            result = await _collect(backend, [1, 2, 3], max_tokens=5)
        finally:
            await backend.close()
    assert result["timings"] is not None
    assert result["timings"]["predicted_n"] == 5
