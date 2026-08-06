"""llama.cpp server backend.

Endpoint/field names verified against docs/reference-notes.md §2 and the live-server
empirical pass in §7: /props -> default_generation_settings.n_ctx (already per-slot, no
division needed) and total_slots; timings object fields (cache_n, prompt_n, prompt_ms,
prompt_per_second, predicted_n, predicted_ms, predicted_per_second, prompt_per_token_ms,
predicted_per_token_ms); streamed /v1/completions carries the full timings object inline on
the terminal SSE chunk, so server (Path B) and client (Path A) numbers come from one request.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from .base import Backend, Capacity, ServerInfo, StreamChunk
from .openai_common import DONE, SSEParser, build_completions_payload


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 300.0):
        super().__init__(base_url, api_key, timeout_s)
        # read timeout is the one that matters for large-prompt/CPU-backend prefill times;
        # connect stays short so a genuinely down server fails fast instead of hanging.
        timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._token_id_prompts_supported: bool | None = None

    def _headers(self) -> dict[str, str]:
        h = {"Accept-Encoding": "identity"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def detect(self) -> bool:
        try:
            r = await self._client.get("/props", headers=self._headers())
            if r.status_code != 200:
                return False
            data = r.json()
            return "default_generation_settings" in data or "total_slots" in data
        except (httpx.HTTPError, ValueError):
            return False

    async def info(self) -> ServerInfo:
        r = await self._client.get("/props", headers=self._headers())
        r.raise_for_status()
        props = r.json()
        model_path = props.get("default_generation_settings", {}).get("model") or props.get("model", "")
        # model_size/model_n_params field names are NOT confirmed present in /props in the
        # pinned commit (docs/reference-notes.md §2 does not document them) -- best-effort
        # read, defaults to None (printed N/A, never a guess) if absent.
        return ServerInfo(
            backend=self.name,
            version="unknown",  # llama.cpp server does not expose a version endpoint over HTTP
            build="unknown",
            model=model_path,
            model_size_bytes=props.get("model_size"),
            model_n_params=props.get("model_params"),
            raw_props=props,
        )

    async def capacity(self) -> Capacity:
        r = await self._client.get("/props", headers=self._headers())
        r.raise_for_status()
        props = r.json()
        dgs = props.get("default_generation_settings", {})
        return Capacity(
            per_request_ctx=dgs.get("n_ctx"),
            max_concurrent=props.get("total_slots"),
            kv_bytes=None,          # not exposed over HTTP; would need GGUF header introspection
            kv_dtype=None,          # ctk/ctv not exposed via /props
            attn_backend=None,      # flash-attn selection not exposed via /props
            prefix_cache_enabled=None,  # cache_prompt is per-request; server-level cache-ram state not queryable
            batch_token_budget=None,    # n_batch/n_ubatch not exposed via /props
            chunked_prefill=True,       # implicit: slot prompts always fed in n_batch pieces, not togglable
            raw=props,
        )

    async def tokenize(self, text: str) -> list[int]:
        r = await self._client.post("/tokenize", json={"content": text}, headers=self._headers())
        r.raise_for_status()
        return r.json()["tokens"]

    async def detokenize(self, token_ids: list[int]) -> str:
        r = await self._client.post("/detokenize", json={"tokens": token_ids}, headers=self._headers())
        r.raise_for_status()
        return r.json()["content"]

    async def supports_token_id_prompts(self) -> bool:
        if self._token_id_prompts_supported is not None:
            return self._token_id_prompts_supported
        try:
            r = await self._client.post(
                "/completion",
                json={"prompt": [1], "n_predict": 1, "cache_prompt": False},
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
        prompt: Any = token_ids if token_ids is not None else text
        extra = dict(extra or {})
        extra["cache_prompt"] = cache_prompt
        payload = build_completions_payload(
            model=model, prompt=prompt, max_tokens=max_tokens, ignore_eos=ignore_eos, extra=extra,
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
        usage = ev.get("usage")
        timings = ev.get("timings")
        return StreamChunk(text=text, finish_reason=finish_reason, usage=usage, server_timings=timings, raw=ev)

    async def metrics_snapshot(self) -> dict[str, Any]:
        try:
            r = await self._client.get("/metrics", headers=self._headers())
        except httpx.HTTPError:
            return {}
        if r.status_code != 200:
            return {}
        return parse_prometheus_text(r.text)

    async def close(self) -> None:
        await self._client.aclose()


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Bare metric name -> value, for label-free series (counters/gauges like
    vllm:num_preemptions_total, llamacpp:requests_processing). For a name that appears with
    multiple label sets (e.g. histogram _bucket lines), only the first-seen value is kept
    under the bare name -- callers needing per-label data (not needed anywhere in llmbench
    today; we only ever read _sum/_count/unlabeled counters) should parse `text` directly.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        raw_name, value = parts
        name = raw_name.split("{", 1)[0]
        if name in out:
            continue  # keep first occurrence; don't let a later same-name/different-label line clobber it
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out
