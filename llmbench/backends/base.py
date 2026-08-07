"""Backend protocol. capacity() is where flag-normalisation lives (course_correct.txt §5) —
each backend computes its own normalised view from its own probes; parity.py and the runner
never reach into backend-specific fields directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ServerInfo:
    backend: str = ""
    version: str = "unknown"
    build: str = "unknown"
    model: str = ""
    model_size_bytes: int | None = None
    model_n_params: int | None = None
    raw_props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Capacity:
    """Normalised parity-axis view. None means unverifiable, not zero/false."""

    per_request_ctx: int | None = None
    max_concurrent: int | None = None
    kv_bytes: int | None = None
    kv_dtype: str | None = None
    attn_backend: str | None = None
    prefix_cache_enabled: bool | None = None
    batch_token_budget: int | None = None
    chunked_prefill: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    text: str | None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    server_timings: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class Backend(ABC):
    name: str = "base"

    # Path B for vLLM is a /metrics delta scrape rather than per-response timings. The runner
    # keys that behaviour off this flag rather than off isinstance(), so a wrapper that
    # delegates to several vLLM backends (suite.lb.fanout.FanoutBackend, used for
    # client-side multi-instance load balancing) still takes the same path.
    supports_vllm_metrics: bool = False

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    @abstractmethod
    async def detect(self) -> bool:
        """Probe-based self-identification. Used by --backend auto."""

    @abstractmethod
    async def info(self) -> ServerInfo:
        ...

    @abstractmethod
    async def capacity(self) -> Capacity:
        ...

    @abstractmethod
    async def tokenize(self, text: str) -> list[int]:
        ...

    @abstractmethod
    async def detokenize(self, token_ids: list[int]) -> str:
        ...

    @abstractmethod
    async def supports_token_id_prompts(self) -> bool:
        ...

    @abstractmethod
    def complete_stream(
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
        ...

    @abstractmethod
    async def metrics_snapshot(self) -> dict[str, Any]:
        """vLLM: parsed /metrics scrape. llama.cpp: parsed /metrics if --metrics was passed,
        else {}."""

    async def close(self) -> None:
        pass
