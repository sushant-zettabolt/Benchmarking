"""Fake OAI-compatible SSE server with injectable delays (spec §14: "the most valuable test
in the suite"). Hand-rolled asyncio TCP server (not uvicorn) so response timing is exact --
we are asserting on wall-clock TTFT/ITL, so anything that adds its own buffering would
contaminate the measurement we're testing.

Simulates: a normal stream, an empty-content first chunk (role-priming style), a stalled
mid-stream gap, a truncated generation (connection dropped early), and a cached-prefix
response (usage.cached_tokens > 0, near-zero effective prefill time).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass


@dataclass
class FakeServerConfig:
    ttft_ms: float = 100.0
    itl_ms: float = 10.0
    n_tokens: int = 10
    n_prompt: int = 16
    empty_first_chunk: bool = False
    stall_at_token: int | None = None
    stall_ms: float = 0.0
    truncate_at_token: int | None = None
    cached_tokens: int = 0
    mode: str = "llamacpp"  # "llamacpp" (inline timings) | "vllm" (no inline timings)


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def _write_headers(writer: asyncio.StreamWriter, status: int, content_type: str, extra: str = "") -> None:
    reason = {200: "OK", 404: "Not Found", 400: "Bad Request"}.get(status, "OK")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n"
        f"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n{extra}\r\n".encode()
    )
    await writer.drain()


async def _write_json(writer: asyncio.StreamWriter, status: int, obj) -> None:
    await _write_headers(writer, status, "application/json")
    body = json.dumps(obj).encode()
    writer.write(_chunk(body))
    writer.write(b"0\r\n\r\n")
    await writer.drain()


async def _stream_completion(writer: asyncio.StreamWriter, payload: dict, config: FakeServerConfig) -> None:
    await _write_headers(writer, 200, "text/event-stream")
    n_tokens = config.n_tokens
    max_tokens = payload.get("max_tokens")
    if max_tokens:
        n_tokens = min(n_tokens, max_tokens)

    if config.empty_first_chunk:
        writer.write(_chunk(_sse({"choices": [{"text": "", "index": 0, "finish_reason": None}]})))
        await writer.drain()

    effective_ttft_ms = 1.0 if config.cached_tokens else config.ttft_ms
    await asyncio.sleep(effective_ttft_ms / 1000.0)

    for i in range(n_tokens):
        if config.truncate_at_token is not None and i >= config.truncate_at_token:
            writer.close()
            return
        writer.write(_chunk(_sse({"choices": [{"text": f"tok{i} ", "index": 0, "finish_reason": None}]})))
        await writer.drain()
        if config.stall_at_token is not None and i == config.stall_at_token:
            await asyncio.sleep(config.stall_ms / 1000.0)
        elif i < n_tokens - 1:
            await asyncio.sleep(config.itl_ms / 1000.0)

    final: dict = {"choices": [{"text": "", "index": 0, "finish_reason": "length"}]}
    usage = {"prompt_tokens": config.n_prompt, "completion_tokens": n_tokens, "total_tokens": config.n_prompt + n_tokens}
    if config.cached_tokens:
        usage["cached_tokens"] = config.cached_tokens
    final["usage"] = usage
    if config.mode == "llamacpp":
        final["timings"] = {
            "cache_n": config.cached_tokens,
            "prompt_n": config.n_prompt - config.cached_tokens,
            "prompt_ms": max(config.ttft_ms - (1.0 if config.cached_tokens else 0.0), 0.1),
            "predicted_n": n_tokens,
            "predicted_ms": config.itl_ms * max(n_tokens - 1, 0),
        }
    writer.write(_chunk(_sse(final)))
    writer.write(_chunk(b"data: [DONE]\n\n"))
    writer.write(b"0\r\n\r\n")
    await writer.drain()


class FakeSSEServer:
    def __init__(self, config: FakeServerConfig | None = None):
        self.config = config or FakeServerConfig()
        self._server: asyncio.AbstractServer | None = None
        self.port: int = 0

    async def __aenter__(self) -> "FakeSSEServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            method, path, _ = request_line.decode().split()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                if b":" in line:
                    k, v = line.decode().split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body = b""
            if "content-length" in headers:
                body = await reader.readexactly(int(headers["content-length"]))

            if path in ("/health", "/v1/health"):
                await _write_json(writer, 200, {"status": "ok"})
            elif path == "/props":
                await _write_json(writer, 200, {
                    "default_generation_settings": {"n_ctx": 4096}, "total_slots": 1,
                })
            elif path == "/v1/models":
                await _write_json(writer, 200, {"data": [{"id": "fake-model", "max_model_len": 4096}]})
            elif path == "/metrics":
                await _write_headers(writer, 200, "text/plain")
                text = "vllm:num_preemptions_total 0\n" if self.config.mode == "vllm" else "llamacpp:requests_processing 0\n"
                writer.write(_chunk(text.encode()))
                writer.write(b"0\r\n\r\n")
                await writer.drain()
            elif path in ("/v1/completions", "/completion") and method == "POST":
                payload = json.loads(body) if body else {}
                await _stream_completion(writer, payload, self.config)
            else:
                await _write_json(writer, 404, {"error": "not found"})
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
