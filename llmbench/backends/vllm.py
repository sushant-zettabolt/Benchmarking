"""vLLM backend.

Verified against docs/reference-notes.md §3/§4: no per-response timings object exists (Path
B is always a /metrics delta scrape, never inline on the stream, unlike llama.cpp).
Prometheus metric names/wire names for TTFT/ITL/preemptions/cached-tokens are in §4's table.

Honesty gap, stated rather than papered over: vLLM's OpenAI-compatible HTTP surface has no
/props equivalent. Most parity axes (kv cache dtype, attention backend, max_num_seqs,
prefix-cache enabled, chunked-prefill enabled) are **not exposed over HTTP at all** in the
pinned commit's public API. capacity() returns None for those in --server-mode attach; only
--server-mode manage (which owns the launch command) can fill them in, from the recorded
launch flags rather than a probe -- and env.py must mark those as `requested`, not
`observed`, per course_correct.txt §2.10.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from .base import Backend, Capacity, ServerInfo, StreamChunk
from .llamacpp import parse_prometheus_text
from .openai_common import DONE, SSEParser, build_completions_payload

# Wire-format Prometheus names, from docs/reference-notes.md §4 (Python name auto-suffixed
# _total for Counters by prometheus_client on scrape).
METRIC_TTFT_HIST = "vllm:time_to_first_token_seconds"
METRIC_ITL_HIST = "vllm:inter_token_latency_seconds"
METRIC_E2E_HIST = "vllm:e2e_request_latency_seconds"
METRIC_PREEMPTIONS_TOTAL = "vllm:num_preemptions_total"
METRIC_PROMPT_TOKENS_CACHED_TOTAL = "vllm:prompt_tokens_cached_total"
METRIC_PROMPT_TOKENS_TOTAL = "vllm:prompt_tokens_total"
METRIC_GENERATION_TOKENS_TOTAL = "vllm:generation_tokens_total"


def _normalise_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lift vLLM's nested prefix-cache counter to the flat key the recorder reads.

    vLLM reports prefix-cache hits as `usage.prompt_tokens_details.cached_tokens`, following
    the OpenAI schema; llama.cpp reports the equivalent as a flat `timings.cache_n`. The
    recorder reads one flat `usage["cached_tokens"]` for both, so without this the field was
    silently None on every vLLM row -- and `cached_tokens` is the only direct evidence that a
    prefix actually hit rather than being re-prefilled. A prefix-caching experiment on vLLM
    was therefore unfalsifiable: hits and misses looked identical.

    Normalising here rather than in metrics.py is deliberate: metrics.py must contain no
    backend-specific branch (enforced by tests/test_metrics_no_backend_branch.py), so
    schema differences get flattened at the backend boundary.
    """
    if not usage:
        return usage
    if usage.get("cached_tokens") is None:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached is not None:
            usage = {**usage, "cached_tokens": cached}
    return usage


class VllmBackend(Backend):
    name = "vllm"
    supports_vllm_metrics = True

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 300.0):
        super().__init__(base_url, api_key, timeout_s)
        timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0)
        # No keep-alive pooling. llama-server (cpp-httplib) closes a kept-alive connection after a
        # few requests, and reusing one it has just closed fails the next request instantly with
        # "Server disconnected without sending a response" -- before it ever reaches the server
        # (seen in the Turin smoke run, ~1 request in 8). A fresh localhost connection per request
        # costs ~0.1 ms, the same for every backend, and makes that race impossible rather than
        # retried: a retry would be timed as one slow request.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout,
                                         limits=httpx.Limits(max_keepalive_connections=0))
        self._token_id_prompts_supported: bool | None = None
        self._tokenizer_model: str | None = None

    def _headers(self) -> dict[str, str]:
        h = {"Accept-Encoding": "identity"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def detect(self) -> bool:
        try:
            r = await self._client.get("/metrics", headers=self._headers())
            if r.status_code != 200:
                return False
            return "vllm:" in r.text or r.text.startswith("# HELP vllm")
        except httpx.HTTPError:
            return False

    async def info(self) -> ServerInfo:
        version = "unknown"
        try:
            r = await self._client.get("/version", headers=self._headers())
            if r.status_code == 200:
                version = r.json().get("version", "unknown")
        except (httpx.HTTPError, ValueError):
            pass
        model = ""
        try:
            r = await self._client.get("/v1/models", headers=self._headers())
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    model = data[0].get("id", "")
        except (httpx.HTTPError, ValueError):
            pass
        return ServerInfo(
            backend=self.name,
            version=version,
            build="unknown",
            model=model,
            model_size_bytes=None,  # vLLM does not expose this over HTTP; print N/A, never guess (spec §10)
            model_n_params=None,
            raw_props={},
        )

    async def capacity(self) -> Capacity:
        per_request_ctx = None
        try:
            r = await self._client.get("/v1/models", headers=self._headers())
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data and "max_model_len" in data[0]:
                    per_request_ctx = data[0]["max_model_len"]
        except (httpx.HTTPError, ValueError):
            pass
        return Capacity(
            per_request_ctx=per_request_ctx,
            max_concurrent=None,   # max_num_seqs not exposed over HTTP; see module docstring
            kv_bytes=None,
            kv_dtype=None,
            attn_backend=None,
            prefix_cache_enabled=None,
            batch_token_budget=None,
            chunked_prefill=None,
            raw={},
        )

    async def tokenize(self, text: str) -> list[int]:
        r = await self._client.post("/tokenize", json={"model": self._tokenizer_model or "", "prompt": text}, headers=self._headers())
        r.raise_for_status()
        return r.json()["tokens"]

    async def detokenize(self, token_ids: list[int]) -> str:
        r = await self._client.post(
            "/detokenize", json={"model": self._tokenizer_model or "", "tokens": token_ids}, headers=self._headers()
        )
        r.raise_for_status()
        return r.json()["prompt"]

    async def supports_token_id_prompts(self) -> bool:
        if self._token_id_prompts_supported is not None:
            return self._token_id_prompts_supported
        try:
            r = await self._client.post(
                "/v1/completions",
                json={"model": self._tokenizer_model or "", "prompt": [1], "max_tokens": 1, "stream": False},
                headers=self._headers(),
            )
            self._token_id_prompts_supported = r.status_code == 200
        except httpx.HTTPError:
            self._token_id_prompts_supported = False
        return self._token_id_prompts_supported

    async def complete_stream(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        max_tokens: int,
        ignore_eos: bool = True,
        model: str = "",
        cache_prompt: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self._tokenizer_model = model
        prompt: Any = token_ids if token_ids is not None else text
        # skip_special_tokens=False: vLLM otherwise drops special tokens from the streamed text,
        # so a generated special token arrives as an empty chunk and TTFT (first *non-empty*
        # chunk) goes unmeasured -- a pp test's only token, sampled from a random prompt, is
        # sometimes one (seen with Qwen3.6 W8A8). Detokenization only; compute is unchanged.
        payload = build_completions_payload(
            model=model, prompt=prompt, max_tokens=max_tokens, ignore_eos=ignore_eos,
            extra={"skip_special_tokens": False, **(extra or {})},
        )
        parser = SSEParser()
        async with self._client.stream(
            "POST", "/v1/completions", json=payload, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_bytes():
                for ev in parser.feed(raw):
                    chunk = self._parse_event(ev)
                    if chunk is not None:
                        yield chunk
            for ev in parser.flush():
                chunk = self._parse_event(ev)
                if chunk is not None:
                    yield chunk

    def _parse_event(self, ev) -> StreamChunk | None:
        if ev == DONE:
            return None
        choices = ev.get("choices") or []
        text = choices[0].get("text") if choices else None
        finish_reason = choices[0].get("finish_reason") if choices else None
        usage = _normalise_usage(ev.get("usage"))
        # vLLM never carries a timings-equivalent object inline (docs/reference-notes.md §3).
        return StreamChunk(text=text, finish_reason=finish_reason, usage=usage, server_timings=None, raw=ev)

    async def metrics_snapshot(self) -> dict[str, Any]:
        r = await self._client.get("/metrics", headers=self._headers())
        r.raise_for_status()
        return parse_prometheus_text(r.text)

    async def close(self) -> None:
        await self._client.aclose()
