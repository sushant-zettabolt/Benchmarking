"""CLI param model, YAML loading, and llama-bench-compatible Cartesian expansion.

Range syntax and Cartesian nesting are replicated from llama-bench.cpp as documented in
docs/reference-notes.md §1 (parse_int_range at llama-bench.cpp:280-320,
get_cmd_params_instances at llama-bench.cpp:1294-1435).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

RANGE_RE = re.compile(r"^(-?\d+)(?:-(\d+)(?:([+*])(\d+))?)?$")


class RangeSyntaxError(ValueError):
    pass


def parse_int_range(spec: str, allow_negative: bool = False) -> list[int]:
    """Replicates llama-bench's parse_int_range(): first[-last[(+|*)step]], comma-separated.

    Guards against non-increasing sequences (+0, *1, *0) exactly as the C++ does.
    `*` steps may overshoot `last` and stop without landing on it exactly.
    """
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = RANGE_RE.match(token)
        if not m:
            raise RangeSyntaxError(f"invalid range token: {token!r}")
        first_s, last_s, op, step_s = m.groups()
        first = int(first_s)
        if not allow_negative and first < 0:
            raise RangeSyntaxError(f"negative value not allowed here: {token!r}")
        if last_s is None:
            out.append(first)
            continue
        last = int(last_s)
        op = op or "+"
        step = int(step_s) if step_s is not None else 1
        if op == "+" and step <= 0:
            raise RangeSyntaxError(f"non-increasing range (+{step}): {token!r}")
        if op == "*" and step <= 1:
            raise RangeSyntaxError(f"non-increasing range (*{step}): {token!r}")
        i = first
        while i <= last:
            out.append(i)
            i = i + step if op == "+" else i * step
    return out


def parse_str_list(spec: str) -> list[str]:
    """Comma-separated enum lookups only, no ranges (-sm, -fa, -ctk, -ctv)."""
    return [s.strip() for s in spec.split(",") if s.strip()]


def parse_bool_list(spec: str) -> list[bool]:
    out = []
    for s in parse_str_list(spec):
        low = s.lower()
        if low in ("1", "true", "yes", "on"):
            out.append(True)
        elif low in ("0", "false", "no", "off"):
            out.append(False)
        else:
            raise RangeSyntaxError(f"invalid boolean value: {s!r}")
    return out


def parse_pg_list(spec: str) -> list[tuple[int, int]]:
    """-pg <pp,tg>[;<pp,tg>...] literal pairs, not a range."""
    out = []
    for pair in spec.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        pp_s, tg_s = pair.split(",")
        out.append((int(pp_s), int(tg_s)))
    return out


# --- Group 1: request-shaping (multi-valued, Cartesian axes we control per request) ---

GROUP1_DEFAULTS: dict[str, Any] = {
    "model": [],
    "n_prompt": [512],
    "n_gen": [128],
    "pg": [],
    "n_depth": [0],
    "concurrency": [1],
    "shared_prefix": [0],
}

# --- Group 2: server-side (fixed at launch; >1 value is an error in --server-mode attach) ---

GROUP2_DEFAULTS: dict[str, Any] = {
    "ngl": [-1],
    "flash_attn": ["auto"],
    "ctk": ["f16"],
    "ctv": ["f16"],
    "batch": [2048],
    "ubatch": [512],
    "threads": [-1],
    "n_parallel": [-1],       # -np / max_num_seqs
    "n_ctx": [0],             # -c / max_model_len (0 = server default / model-loaded)
}

# llama-bench.cpp:1297 nesting order, outermost (varies slowest) -> innermost, restricted to
# the axes we implement. n_prompt/n_gen/pg are parallel inner loops per llama-bench, not
# nested against each other (each skipped when zero/zero-pair).
NESTING_ORDER = [
    "model",
    "ngl",
    "batch",
    "ubatch",
    "ctk",
    "ctv",
    "flash_attn",
    "threads",
    "n_parallel",
    "n_ctx",
    "n_depth",
    "concurrency",
    "shared_prefix",
]


@dataclass
class CmdParams:
    """Multi-valued axes (Cartesian product source) plus run-scalar options."""

    # Group 1
    model: list[str] = field(default_factory=list)
    n_prompt: list[int] = field(default_factory=lambda: [512])
    n_gen: list[int] = field(default_factory=lambda: [128])
    pg: list[tuple[int, int]] = field(default_factory=list)
    n_depth: list[int] = field(default_factory=lambda: [0])
    concurrency: list[int] = field(default_factory=lambda: [1])
    shared_prefix: list[int] = field(default_factory=lambda: [0])

    # Group 2
    ngl: list[int] = field(default_factory=lambda: [-1])
    flash_attn: list[str] = field(default_factory=lambda: ["auto"])
    ctk: list[str] = field(default_factory=lambda: ["f16"])
    ctv: list[str] = field(default_factory=lambda: ["f16"])
    batch: list[int] = field(default_factory=lambda: [2048])
    ubatch: list[int] = field(default_factory=lambda: [512])
    threads: list[int] = field(default_factory=lambda: [-1])
    n_parallel: list[int] = field(default_factory=lambda: [-1])
    n_ctx: list[int] = field(default_factory=lambda: [0])

    # scalars
    reps: int = 5
    no_warmup: bool = False
    warmup_fixed: int | None = None
    delay: float = 0.0
    request_rate: float | None = None
    burstiness: float = 1.0
    output: str = "md"
    output_err: str | None = None
    verbose: bool = False
    progress: bool = False

    # Group 3: HTTP-specific
    url: str = "http://127.0.0.1:8080"
    api_key: str | None = None
    timeout: float = 300.0  # HTTP read timeout in seconds; large prompts on slow/CPU backends need headroom
    backend: str = "auto"
    measure: str = "both"
    server_mode: str = "attach"
    server_cmd: str | None = None
    server_model_path: str | None = None  # manage mode only: filesystem path for the launched server's own -m/--model
    endpoint: str = "completions"
    no_verify_tokens: bool = False
    force: bool = False
    out_dir: str = "out"
    tag: str | None = None
    parity_mode: str | None = None  # required for `llmbench compare`/cross-backend runs

    def group2_multi_valued(self) -> list[str]:
        return [
            name
            for name in ("ngl", "flash_attn", "ctk", "ctv", "batch", "ubatch", "threads", "n_parallel", "n_ctx")
            if len(getattr(self, name)) > 1
        ]

    def validate_attach_mode(self) -> None:
        if self.server_mode != "attach":
            return
        multi = self.group2_multi_valued()
        if multi:
            raise ValueError(
                f"--server-mode attach requires exactly one value for server-side flags; "
                f"got multiple values for: {', '.join(multi)}. Use --server-mode manage to sweep these."
            )


@dataclass
class Instance:
    """One point in the Cartesian product of Group-1 + Group-2 axes."""

    model: str
    ngl: int
    batch: int
    ubatch: int
    ctk: str
    ctv: str
    flash_attn: str
    threads: int
    n_parallel: int
    n_ctx: int
    n_depth: int
    concurrency: int
    shared_prefix: int
    n_prompt: int = 0
    n_gen: int = 0
    is_pg: bool = False

    def test_name(self) -> str:
        if self.is_pg:
            base = f"pp{self.n_prompt}+tg{self.n_gen}"
        elif self.n_gen == 0:
            base = f"pp{self.n_prompt}"
        elif self.n_prompt == 0:
            base = f"tg{self.n_gen}"
        else:
            base = f"pp{self.n_prompt}+tg{self.n_gen}"
        if self.n_depth > 0:
            base += f" @ d{self.n_depth}"
        return base

    def group2_flags(self) -> dict[str, Any]:
        return {
            "ngl": self.ngl,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "ctk": self.ctk,
            "ctv": self.ctv,
            "flash_attn": self.flash_attn,
            "threads": self.threads,
            "n_parallel": self.n_parallel,
            "n_ctx": self.n_ctx,
        }


def get_cmd_params_instances(p: CmdParams) -> list[Instance]:
    """Cartesian expansion mirroring llama-bench.cpp:1294-1435's nesting order.

    Outer axes vary slowest (minimizes server restarts in --server-mode manage);
    n_prompt/n_gen/pg are parallel inner loops, each skipped when zero-valued, matching
    llama-bench's test_prompt/test_gen/test_pg dispatch.
    """
    models = p.model or [""]
    outer_axes = [
        models,
        p.ngl,
        p.batch,
        p.ubatch,
        p.ctk,
        p.ctv,
        p.flash_attn,
        p.threads,
        p.n_parallel,
        p.n_ctx,
        p.n_depth,
        p.concurrency,
        p.shared_prefix,
    ]

    def cartesian(axes):
        if not axes:
            yield ()
            return
        head, *rest = axes
        for h in head:
            for t in cartesian(rest):
                yield (h,) + t

    instances: list[Instance] = []
    for combo in cartesian(outer_axes):
        (model, ngl, batch, ubatch, ctk, ctv, flash_attn, threads, n_parallel, n_ctx,
         n_depth, concurrency, shared_prefix) = combo
        base_kwargs = dict(
            model=model, ngl=ngl, batch=batch, ubatch=ubatch, ctk=ctk, ctv=ctv,
            flash_attn=flash_attn, threads=threads, n_parallel=n_parallel, n_ctx=n_ctx,
            n_depth=n_depth, concurrency=concurrency, shared_prefix=shared_prefix,
        )
        for n_prompt in p.n_prompt:
            if n_prompt == 0:
                continue
            instances.append(Instance(**base_kwargs, n_prompt=n_prompt, n_gen=0))
        for n_gen in p.n_gen:
            if n_gen == 0:
                continue
            instances.append(Instance(**base_kwargs, n_prompt=0, n_gen=n_gen))
        for pp, tg in p.pg:
            if pp == 0 and tg == 0:
                continue
            instances.append(Instance(**base_kwargs, n_prompt=pp, n_gen=tg, is_pg=True))
    return instances


def load_yaml_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_cli_over_yaml(yaml_dict: dict[str, Any], cli_dict: dict[str, Any]) -> dict[str, Any]:
    """CLI overrides YAML, field by field. Only keys explicitly set on the CLI override."""
    merged = dict(yaml_dict)
    merged.update({k: v for k, v in cli_dict.items() if v is not None})
    return merged
