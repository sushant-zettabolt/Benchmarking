"""HTTP-level behaviour of the backends that the Turin smoke run showed matters."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llmbench import metrics
from llmbench.backends.llamacpp import LlamaCppBackend
from llmbench.backends.vllm import VllmBackend


@pytest.mark.parametrize("cls", [LlamaCppBackend, VllmBackend])
def test_backends_do_not_reuse_connections(cls):
    """llama-server closes a kept-alive connection after a few requests; reusing one it has just
    closed failed ~1 measured request in 8 with 'Server disconnected without sending a
    response', before the request reached the server. Every request gets a fresh connection."""
    backend = cls("http://127.0.0.1:1")
    pool = backend._client._transport._pool
    assert pool._max_keepalive_connections == 0
    asyncio.run(backend._client.aclose())


def test_vllm_keeps_special_tokens_in_the_streamed_text():
    """Otherwise a generated special token streams as empty text, and TTFT -- the first
    non-empty chunk -- is never recorded for a pp test whose only token is one."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = ('data: {"choices":[{"text":"<|im_end|>","finish_reason":"length"}]}\n\n'
                "data: [DONE]\n\n")
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    backend = VllmBackend("http://vllm.test")
    backend._client = httpx.AsyncClient(base_url="http://vllm.test",
                                        transport=httpx.MockTransport(handler))

    async def go():
        return [c async for c in backend.complete_stream(token_ids=[1, 2], max_tokens=1)]

    chunks = asyncio.run(go())
    assert sent[0]["skip_special_tokens"] is False
    assert any(c.text for c in chunks)
    asyncio.run(backend._client.aclose())


def test_a_single_generated_token_has_no_server_decode_rate():
    """llama-server reports predicted_ms = 0.001 for one generated token, which divided out to
    1,000,000 t/s in the report. A decode rate needs at least two tokens."""
    one = {"server_prompt_n": 16, "server_prompt_ms": 100.0,
           "server_predicted_n": 1, "server_predicted_ms": 0.001, "n_gen_actual": 1,
           "n_prompt_actual": 16, "t_send_ns": 0, "t_first_token_ns": 100_000_000,
           "t_end_ns": 101_000_000, "itl_ns": []}
    many = {**one, "server_predicted_n": 16, "server_predicted_ms": 1000.0, "n_gen_actual": 16}

    assert metrics.aggregate_server([one], "pp16").decode_tps.n == 0
    assert metrics.aggregate_server([many], "tg16").decode_tps.mean == 16.0
    assert metrics.per_request_values([one])[0]["server_decode_tps"] is None
