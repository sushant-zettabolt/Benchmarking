"""Shared row flattening for every report format.

All three formats render the same table, so the column set and the value formatting live here
once. A metric a trial did not measure renders as empty/`N/A`, never as 0 -- a zero in a
latency column would read as an extraordinary result rather than as missing data.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Iterable

from ..execute import STATUS_OK, TrialResult

# (column header, source, key). source: "top" = TrialResult attr, "axis", "metric", "prov"
# (provenance; a dotted key descends into nested dicts).
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("status", "top", "status"),
    ("kind", "top", "kind"),
    ("src", "top", "src"),
    ("backend", "top", "backend"),
    ("model", "top", "model"),
    ("test", "top", "test"),
    ("instances", "axis", "instances"),
    ("cores/inst", "axis", "cores_per_instance"),
    ("threads/inst", "axis", "threads_per_instance"),
    ("np", "axis", "n_parallel"),
    ("n_ctx", "axis", "n_ctx"),
    ("batch", "axis", "batch"),
    ("ubatch", "axis", "ubatch"),
    ("lb", "axis", "lb"),
    ("tool", "axis", "tool"),
    ("batch_size", "axis", "batch_size"),
    ("conc", "axis", "concurrency"),
    ("rate", "axis", "request_rate"),
    ("t/s", "metric", "tps_mean"),
    ("t/s sd", "metric", "tps_stddev"),
    # t/s is (prompt + generated tokens) / end-to-end time. These two split it the way
    # llama-batched-bench's S_PP / S_TG do: prompt tokens / TTFT, and (n_gen - 1) / decode time.
    ("prefill t/s", "metric", "prefill_tps_mean"),
    ("decode t/s", "metric", "decode_tps_mean"),
    ("tok tput", "metric", "total_token_throughput"),
    ("req tput", "metric", "request_throughput"),
    ("ttft p50", "metric", "ttft_ms_p50"),
    ("ttft p99", "metric", "ttft_ms_p99"),
    ("itl mean", "metric", "itl_ms_mean"),
    ("itl p95", "metric", "itl_ms_p95"),
    ("tok/chunk", "metric", "tokens_per_chunk"),
    ("tpot mean", "metric", "tpot_ms_mean"),
    ("e2e p50", "metric", "e2e_ms_mean"),
    ("overhead ms", "metric", "overhead_ms_mean"),
    ("reps", "metric", "n_reps_valid"),
    # measurement conditions: the cores' mean clock during the measured requests, and the share
    # of the servers' memory on their bound NUMA node(s) (see deploy.verify_memory_placement)
    ("MHz", "prov", "clock_mhz"),
    ("mem local", "prov", "placement.memory_local_fraction"),
    ("trial", "top", "trial_id"),
    ("error", "top", "error"),
)

# Columns whose values are latency-like and therefore better when smaller. Used to colour
# the HTML heatmap in the right direction.
LOWER_IS_BETTER = {
    "ttft p50", "ttft p99", "itl mean", "itl p95", "tpot mean", "e2e p50", "overhead ms",
}


def cell(result: TrialResult, source: str, key: str) -> Any:
    if source == "top":
        return getattr(result, key, None)
    if source == "axis":
        return result.axes.get(key)
    if source == "prov":
        value: Any = result.provenance or {}
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return value
    return result.metrics.get(key)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def used_columns(results: Iterable[TrialResult]) -> list[tuple[str, str, str]]:
    """Drop columns that are empty for every row, so a llama.cpp-only run does not carry a
    column of blank vLLM fields (and vice versa)."""
    results = list(results)
    keep = []
    for col in COLUMNS:
        header, source, key = col
        if header in ("status", "backend", "test", "trial"):
            keep.append(col)
            continue
        if any(cell(r, source, key) not in (None, "", []) for r in results):
            keep.append(col)
    return keep


def rows_for(results: list[TrialResult], columns: list[tuple[str, str, str]]) -> list[list[str]]:
    return [[fmt(cell(r, s, k)) for _, s, k in columns] for r in results]


# report_reps.csv: one line per measured request. (header, key in metrics.per_request_values)
REP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("rep", "rep"),
    ("n_prompt", "n_prompt"),
    ("n_gen", "n_gen"),
    ("t/s", "tps"),
    ("prefill t/s", "prefill_tps"),
    ("decode t/s", "decode_tps"),
    ("ttft ms", "ttft_ms"),
    ("e2e ms", "e2e_ms"),
    ("tpot ms", "tpot_ms"),
    ("itl mean ms", "itl_ms_mean"),
    ("server t/s", "server_tps"),
    ("server prefill t/s", "server_prefill_tps"),
    ("server decode t/s", "server_decode_tps"),
    ("error", "error"),
)


def shell_command(cmd: dict[str, Any]) -> str:
    """One server's launch as a copy-pasteable shell line: `K=V ... numactl ... server ...`.
    The env part is what the sweep set on top of its own environment, not the whole env."""
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (cmd.get("env") or {}).items())
    argv = shlex.join(str(a) for a in cmd.get("argv") or [])
    return f"{env} {argv}".strip()


def server_commands(results: Iterable[TrialResult]) -> dict[str, tuple[str, list[dict]]]:
    """deployment id -> (backend, the commands its servers were started with), in run order,
    read back from each row's provenance so a re-rendered report still has them."""
    out: dict[str, tuple[str, list[dict]]] = {}
    for r in results:
        cmds = (r.provenance or {}).get("server_commands")
        if cmds and r.deployment_id not in out:
            out[r.deployment_id] = (r.backend, cmds)
    return out


def host_lines(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, value) pairs describing the host/pod and the server software, for md/html."""
    h = manifest.get("host_info") or {}
    if not h:
        return []
    k8s = h.get("kubernetes") or {}
    lines = [("host", h.get("hostname", ""))]
    if k8s.get("in_pod"):
        pod = k8s.get("pod") or h.get("hostname", "")
        lines.append(("kubernetes pod", f"{pod} (namespace {k8s.get('namespace') or '?'}"
                                        f"{', node ' + k8s['node'] if k8s.get('node') else ''})"))
    lines += [
        ("os / kernel", f"{h.get('os', '')} / {h.get('kernel', '')}"),
        ("cpu", h.get("cpu_model", "")),
        ("cpus online / allowed", f"{h.get('cpus_online', '')} / {h.get('cpus_allowed', '')}"),
        ("numa mems allowed", h.get("mems_allowed", "")),
        ("memory", f"{h.get('mem_total_gib')} GiB total"),
    ]
    allowed = {str(n) for n in _expand(h.get("mems_allowed", ""))}
    for n in h.get("numa_nodes") or []:
        if not allowed or str(n.get("node")) in allowed:
            lines.append((f"numa node {n.get('node')}",
                          f"cpus {n.get('cpus')}, {n.get('mem_total_gib')} GiB total, "
                          f"{n.get('mem_free_gib')} GiB free at start"))
    cg = h.get("cgroup") or {}
    lines += [
        ("cgroup cpu.max / memory.max", f"{cg.get('cpu_max') or '-'} / {cg.get('memory_max') or '-'}"),
        ("memlock limit", h.get("memlock_limit", "")),
        ("python / llmbench", f"{h.get('python', '')} / {h.get('llmbench_commit') or '?'}"),
    ]
    for path, sw in (manifest.get("software") or {}).items():
        pkgs = ", ".join(f"{k} {v}" for k, v in (sw.get("packages") or {}).items())
        lines.append((f"{sw.get('type')} {path}",
                      sw.get("version", "") + (f" [{pkgs}]" if pkgs else "")))
    return lines


def _expand(cpulist: str) -> list[int]:
    """'0-2,5' -> [0, 1, 2, 5]; tolerant of empty or malformed input."""
    out: list[int] = []
    for part in (cpulist or "").split(","):
        lo, _, hi = part.strip().partition("-")
        try:
            out.extend(range(int(lo), int(hi or lo) + 1))
        except ValueError:
            continue
    return out


@dataclass
class ReportContext:
    """Everything the renderers need, assembled once."""

    manifest: dict[str, Any]
    results: list[TrialResult]
    ranking: Any                    # objective.RankingReport
    plan: dict[str, Any]

    @property
    def ok_results(self) -> list[TrialResult]:
        return [r for r in self.results if r.status == STATUS_OK]

    @property
    def online(self) -> list[TrialResult]:
        return [r for r in self.results if r.kind == "online"]

    @property
    def offline(self) -> list[TrialResult]:
        return [r for r in self.results if r.kind == "offline"]

    @property
    def failures(self) -> list[TrialResult]:
        return [r for r in self.results if r.status not in (STATUS_OK,)]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def all_warnings(self) -> list[str]:
        seen: dict[str, None] = {}
        for w in self.plan.get("warnings", []):
            seen.setdefault(str(w), None)
        for r in self.results:
            for w in r.warnings:
                seen.setdefault(str(w), None)
        return list(seen)


__all__ = ["COLUMNS", "LOWER_IS_BETTER", "REP_COLUMNS", "ReportContext", "cell", "fmt",
           "host_lines", "rows_for", "server_commands", "shell_command", "used_columns"]
