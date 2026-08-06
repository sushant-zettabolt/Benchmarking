"""Prompt construction (spec §8) and workload persistence/replay (course_correct.txt §4).

Path 1 (preferred): raw random token-ID arrays, exact n_prompt control, zero tokenizer
ambiguity -- probed via Backend.supports_token_id_prompts() at startup.
Path 2 (fallback): random text -> tokenize -> trim/pad -> detokenize, with a round-trip
stability assertion (detokenize->retokenize is not guaranteed idempotent).

Uniqueness: a per-run salt plus a per-request counter are folded into the first tokens
after any shared prefix, so prompts are unique across the *entire run* -- required for
vLLM's global content-hashed prefix cache, not just consecutive-request uniqueness
(docs/reference-notes.md §4, spec §4.2).

--shared-prefix draws its shared span from a separate RNG seeded *only* by the run salt, so
every request in the run reproduces byte-identical shared tokens regardless of request_idx.
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Tokenizer(Protocol):
    async def tokenize(self, text: str) -> list[int]: ...
    async def detokenize(self, token_ids: list[int]) -> str: ...


def new_run_salt() -> int:
    return time.time_ns() & 0x7FFFFFFF


def _salt_token(value: int, vocab_size: int) -> int:
    return value % max(vocab_size, 2)


def generate_prompt_tokens(
    *,
    n_prompt: int,
    shared_prefix_n: int,
    run_salt: int,
    request_idx: int,
    vocab_size: int,
    rng: random.Random,
    bos_token_id: int | None = None,
) -> list[int]:
    if n_prompt <= 0:
        return []
    tokens: list[int] = []
    remaining = n_prompt
    if bos_token_id is not None:
        tokens.append(bos_token_id)
        remaining -= 1
    if shared_prefix_n > 0 and remaining > 0:
        prefix_rng = random.Random(run_salt)  # salt-only seed -> identical across every request
        shared_n = min(shared_prefix_n, remaining)
        tokens.extend(prefix_rng.randrange(vocab_size) for _ in range(shared_n))
        remaining -= shared_n
    if remaining > 0:
        marker = [_salt_token(run_salt, vocab_size), _salt_token(request_idx, vocab_size)]
        marker_n = min(2, remaining)
        tokens.extend(marker[:marker_n])
        remaining -= marker_n
    if remaining > 0:
        tokens.extend(rng.randrange(vocab_size) for _ in range(remaining))
    return tokens[:n_prompt]


def generate_depth_prefix_tokens(
    *, n_depth: int, run_salt: int, request_idx: int, vocab_size: int, rng: random.Random,
) -> list[int]:
    """A d-token priming prefix, unique per request (spec §8 n_depth emulation)."""
    if n_depth <= 0:
        return []
    marker = [_salt_token(run_salt ^ 0x5A5A, vocab_size), _salt_token(request_idx, vocab_size)]
    marker_n = min(2, n_depth)
    tokens = marker[:marker_n]
    tokens.extend(rng.randrange(vocab_size) for _ in range(n_depth - marker_n))
    return tokens[:n_depth]


async def fallback_text_prompt(tok: Tokenizer, n_prompt: int, rng: random.Random) -> tuple[list[int], str]:
    """Path 2: random text -> /tokenize -> trim/pad to exactly n_prompt -> /detokenize ->
    re-tokenize to confirm the round trip is stable. Raises AssertionError on drift."""
    words = " ".join(f"w{rng.randrange(100000)}" for _ in range(n_prompt))
    tokens = await tok.tokenize(words)
    if len(tokens) > n_prompt:
        tokens = tokens[:n_prompt]
    elif len(tokens) < n_prompt:
        pad = await tok.tokenize(" pad" * (n_prompt - len(tokens)))
        tokens = (tokens + pad)[:n_prompt]
    text = await tok.detokenize(tokens)
    retokenized = await tok.tokenize(text)
    if retokenized != tokens:
        raise AssertionError(
            f"detokenize->retokenize drift: {len(tokens)} tokens in, {len(retokenized)} out"
        )
    return tokens, text


# --- Workload persistence/replay (course_correct.txt §4, gate G8) ---


@dataclass
class WorkloadItem:
    request_idx: int
    instance_id: str
    rep_idx: int
    parity_mode: str  # "weights" | "native" | "" (no cross-backend parity requested)
    token_ids: list[int]  # weights mode: the exact IDs sent. native mode: reference-tokenizer IDs.
    text: str | None = None  # native mode only: decoded once via the reference tokenizer
    shared_prefix_n: int = 0
    n_depth: int = 0
    depth_prefix_token_ids: list[int] = field(default_factory=list)


class WorkloadWriter:
    def __init__(self, path: str | Path, seed: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._f = open(self.path, "w")
        self._f.write(json.dumps({"schema": 1, "seed": seed}) + "\n")

    def write(self, item: WorkloadItem) -> None:
        self._f.write(json.dumps(dataclasses.asdict(item)) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_workload_jsonl(path: str | Path) -> tuple[int, list[WorkloadItem]]:
    items = []
    seed = 0
    with open(path) as f:
        header = json.loads(f.readline())
        seed = header.get("seed", 0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(WorkloadItem(**json.loads(line)))
    return seed, items
