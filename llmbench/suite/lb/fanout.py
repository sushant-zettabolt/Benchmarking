"""Client-side load balancing: a Backend that spreads requests over a fleet of instances.

This is the default multi-instance path, and it is the more accurate one for measurement:
there is no proxy process, so nothing is added to the wire-to-wire window the client is
timing, and no cores are spent forwarding bytes. `lb.kind: nginx` exists for when you
specifically want the production topology -- including the proxy's cost -- inside the number.

`FanoutBackend` implements the full `Backend` interface by delegating, so `runner.run_instance`
drives a fleet with no changes: it still sees one backend object.

Balancing is per *request*, chosen at dispatch. `least-outstanding` tracks in-flight requests
per member and picks the emptiest, which is what nginx's `least_conn` approximates and is the
right policy when request costs vary by orders of magnitude (pp1024 vs tg64). `round-robin`
is available for a strictly deterministic distribution.
"""
from __future__ import annotations

import itertools
from typing import Any, AsyncIterator

from ...backends.base import Backend, Capacity, ServerInfo, StreamChunk


class FanoutBackend(Backend):
    """Fronts N homogeneous backends as one. Not for mixing backend types."""

    def __init__(self, members: list[Backend], *, strategy: str = "least-outstanding"):
        if not members:
            raise ValueError("FanoutBackend requires at least one member backend")
        names = {m.name for m in members}
        if len(names) > 1:
            raise ValueError(f"FanoutBackend members must be the same backend type, got {sorted(names)}")
        super().__init__(members[0].base_url, members[0].api_key, members[0].timeout_s)
        self.members = members
        self.strategy = strategy
        self.name = members[0].name
        self.supports_vllm_metrics = any(m.supports_vllm_metrics for m in members)
        self._inflight = [0] * len(members)
        self._rr = itertools.cycle(range(len(members)))
        self.dispatch_counts = [0] * len(members)

    # -- selection --

    def _pick(self) -> int:
        if len(self.members) == 1:
            return 0
        if self.strategy == "round-robin":
            return next(self._rr)
        # Least-outstanding, with ties broken by fewest dispatches so far.
        #
        # The tie-break matters more than it looks. At concurrency 1 every member always has
        # zero outstanding requests at dispatch time, so a naive lowest-index tie-break sends
        # 100% of traffic to instance 0 and leaves the rest of the fleet idle -- turning a
        # "4-instance" measurement into a 1-instance one on a quarter of the cores, with no
        # outward sign anything was wrong. Preferring the least-dispatched member makes ties
        # round-robin, so the load actually spreads. Index is the final tie-break, so the
        # choice stays deterministic and runs remain reproducible.
        return min(
            range(len(self.members)),
            key=lambda i: (self._inflight[i], self.dispatch_counts[i], i),
        )

    # -- delegation: probes go to member 0, which is representative of a homogeneous fleet --

    async def detect(self) -> bool:
        return await self.members[0].detect()

    async def info(self) -> ServerInfo:
        return await self.members[0].info()

    async def capacity(self) -> Capacity:
        """Report the fleet's capacity, not one instance's.

        `max_concurrent` is summed: N instances of `-np 8` really do admit 8N concurrent
        requests, and the runner's capacity preflight would otherwise reject a perfectly
        valid concurrency. Per-request context is NOT summed -- context is a property of a
        single request and does not aggregate.
        """
        caps = [await m.capacity() for m in self.members]
        head = caps[0]
        max_concurrent = None
        if all(c.max_concurrent is not None for c in caps):
            max_concurrent = sum(c.max_concurrent for c in caps)  # type: ignore[misc]
        kv_bytes = None
        if all(c.kv_bytes is not None for c in caps):
            kv_bytes = sum(c.kv_bytes for c in caps)  # type: ignore[misc]
        return Capacity(
            per_request_ctx=head.per_request_ctx,
            max_concurrent=max_concurrent,
            kv_bytes=kv_bytes,
            kv_dtype=head.kv_dtype,
            attn_backend=head.attn_backend,
            prefix_cache_enabled=head.prefix_cache_enabled,
            batch_token_budget=head.batch_token_budget,
            chunked_prefill=head.chunked_prefill,
            raw={"n_instances": len(self.members), "member_0": head.raw},
        )

    async def tokenize(self, text: str) -> list[int]:
        return await self.members[0].tokenize(text)

    async def detokenize(self, token_ids: list[int]) -> str:
        return await self.members[0].detokenize(token_ids)

    async def supports_token_id_prompts(self) -> bool:
        return await self.members[0].supports_token_id_prompts()

    async def complete_stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        idx = self._pick()
        self._inflight[idx] += 1
        self.dispatch_counts[idx] += 1
        try:
            async for chunk in self.members[idx].complete_stream(**kwargs):
                yield chunk
        finally:
            self._inflight[idx] -= 1

    async def metrics_snapshot(self) -> dict[str, Any]:
        """Sum counters across the fleet.

        Every vLLM metric the runner reads (`preemptions_total`, `prompt_tokens_total`,
        `*_time_seconds_sum`/`_count`) is a monotonic counter, so a fleet-wide sum is the
        correct aggregate and the before/after delta stays meaningful. A member that fails to
        scrape is skipped rather than zeroed -- zeroing would fabricate a negative delta.
        """
        total: dict[str, float] = {}
        for m in self.members:
            try:
                snap = await m.metrics_snapshot()
            except Exception:  # noqa: BLE001 -- Path B is best-effort
                continue
            for k, v in snap.items():
                if isinstance(v, (int, float)):
                    total[k] = total.get(k, 0.0) + v
        return total

    async def close(self) -> None:
        for m in self.members:
            await m.close()

    def distribution(self) -> dict[str, Any]:
        """How requests actually landed. Reported so an uneven split is visible rather than
        silently skewing a multi-instance result."""
        total = sum(self.dispatch_counts)
        return {
            "strategy": self.strategy,
            "n_instances": len(self.members),
            "dispatch_counts": list(self.dispatch_counts),
            "dispatch_share": [round(c / total, 4) for c in self.dispatch_counts] if total else [],
            "urls": [m.base_url for m in self.members],
        }


__all__ = ["FanoutBackend"]
