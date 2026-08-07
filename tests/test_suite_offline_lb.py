"""Offline output parsing, command construction, and load-balancing behaviour."""
from __future__ import annotations

import asyncio
import json

import pytest

from llmbench.backends.base import Backend, Capacity, ServerInfo, StreamChunk
from llmbench.suite.lb.fanout import FanoutBackend
from llmbench.suite.offline.llamacpp import (
    build_batched_bench_argv, build_llama_bench_argv, parse_batched_bench, parse_llama_bench,
)
from llmbench.suite.offline.vllm import build_latency_argv
from llmbench.suite.plan import OfflinePlan
from llmbench.suite.spec import BackendSpec, CpuSpec
from llmbench.suite.topology import allocate
from tests.test_suite_topology import make_topology


@pytest.fixture
def cores():
    return allocate(make_topology(), "96-103", n_instances=1).instances[0]


def make_offline_plan(backend="llamacpp", tool="llama-bench", *, batch_size=1, cores=None,
                      **kw) -> OfflinePlan:
    bspec = BackendSpec(
        name=backend, model="/m/model.gguf", server_bin="/b/server",
        offline_bin="/b/llama-bench", batched_bin="/b/llama-batched-bench",
    )
    return OfflinePlan(
        id="o0000", backend=backend, backend_spec=bspec, tool=tool, batch_size=batch_size,
        n_prompt=kw.pop("n_prompt", 512), n_gen=kw.pop("n_gen", 128), reps=kw.pop("reps", 3),
        cores=cores, **kw,
    )


# --- llama-bench ---


LLAMA_BENCH_JSON = """
build: a1f96d4fc (10293)
[
  {"model_filename":"model.gguf","n_prompt":512,"n_gen":0,"n_depth":0,
   "avg_ts":606.69,"stddev_ts":0.99,"avg_ns":844000000,"n_threads":96},
  {"model_filename":"model.gguf","n_prompt":0,"n_gen":128,"n_depth":0,
   "avg_ts":21.02,"stddev_ts":0.05,"avg_ns":6089000000,"n_threads":96}
]
"""


def test_parse_llama_bench_skips_leading_log_noise():
    rows = parse_llama_bench(None, LLAMA_BENCH_JSON)
    assert [r["n_prompt"] for r in rows] == [512, 0]


def test_llama_bench_argv_pins_and_passes_reps(cores):
    plan = make_offline_plan(cores=cores, reps=7, n_prompt=512, n_gen=128)
    argv = build_llama_bench_argv(plan, CpuSpec(budget="96-103"))
    assert argv[0] == "numactl"
    assert "--physcpubind=96-103" in argv
    assert "--membind=1" in argv
    assert argv[argv.index("-r") + 1] == "7"
    assert argv[argv.index("-o") + 1] == "json"
    # ggml re-pins workers to the numactl cpuset only under --numa numactl
    assert argv[argv.index("--numa") + 1] == "numactl"
    assert argv[argv.index("-t") + 1] == "8"


# --- llama-batched-bench ---


BATCHED_JSONL = """
main: n_kv_max = 4096, n_batch = 2048
| PP | TG | B |
{"n_kv_max": 4096, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "is_pp_shared": 0, "n_gpu_layers": 0, "n_threads": 8, "n_threads_batch": 8, "pp": 512, "tg": 128, "pl": 4, "n_kv": 2560, "t_pp": 3.5, "speed_pp": 585.1, "t_tg": 24.3, "speed_tg": 21.1, "t": 27.8, "speed": 92.1}
not json at all
"""


def test_parse_batched_bench_ignores_non_json_lines():
    rows = parse_batched_bench(BATCHED_JSONL)
    assert len(rows) == 1
    assert rows[0]["pl"] == 4 and rows[0]["speed"] == 92.1


def test_batched_bench_argv_sizes_context_for_the_whole_batch(cores):
    """Every one of the `pl` sequences needs pp+tg tokens of KV, so n_ctx must cover the
    batch or the run dies partway through with a context overflow."""
    plan = make_offline_plan(tool="llama-batched-bench", batch_size=4, cores=cores,
                             n_prompt=512, n_gen=128)
    argv = build_batched_bench_argv(plan, CpuSpec(budget="96-103"))
    assert argv[argv.index("-npl") + 1] == "4"
    assert argv[argv.index("-c") + 1] == str(4 * 512 + 4 * 128)
    assert argv[argv.index("--output-format") + 1] == "jsonl"
    assert "-pps" not in argv


def test_batched_bench_shared_prompt_stores_the_prompt_once(cores):
    plan = make_offline_plan(tool="llama-batched-bench", batch_size=4, cores=cores,
                             n_prompt=512, n_gen=128, shared_prompt=True)
    argv = build_batched_bench_argv(plan, CpuSpec(budget="96-103"))
    assert "-pps" in argv
    assert argv[argv.index("-c") + 1] == str(512 + 4 * 128)


# --- vllm ---


def test_vllm_latency_argv_maps_static_batch(cores, tmp_path):
    plan = make_offline_plan(backend="vllm", tool="vllm-latency", batch_size=8, cores=cores,
                             n_prompt=256, n_gen=64, reps=5)
    plan.backend_spec.offline_bin = "/b/vllm"
    argv = build_latency_argv(plan, CpuSpec(budget="96-103"), output_json=tmp_path / "o.json")
    assert argv[:2] == ["numactl", "--physcpubind=96-103"]
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert argv[argv.index("--input-len") + 1] == "256"
    assert argv[argv.index("--output-len") + 1] == "64"
    assert argv[argv.index("--num-iters") + 1] == "5"


# --- fanout load balancing ---


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, url, *, max_concurrent=8, metrics=None):
        super().__init__(url)
        self.calls = 0
        self._max_concurrent = max_concurrent
        self._metrics = metrics or {}
        self.closed = False

    async def detect(self): return True
    async def info(self): return ServerInfo(backend="fake", model="m")
    async def capacity(self):
        return Capacity(per_request_ctx=4096, max_concurrent=self._max_concurrent, kv_bytes=100)
    async def tokenize(self, text): return [1]
    async def detokenize(self, ids): return "x"
    async def supports_token_id_prompts(self): return True
    async def metrics_snapshot(self): return dict(self._metrics)
    async def close(self): self.closed = True

    async def complete_stream(self, **kw):
        self.calls += 1
        yield StreamChunk(text="a")


def run(coro):
    return asyncio.run(coro)


def test_least_outstanding_spreads_evenly_at_concurrency_one():
    """The tie-break is what makes this work. With a naive lowest-index tie-break every
    request at concurrency 1 lands on instance 0 and the rest of the fleet idles, turning a
    4-instance measurement into a 1-instance one with no outward sign."""
    members = [FakeBackend(f"http://h:{800 + i}") for i in range(4)]
    fan = FanoutBackend(members, strategy="least-outstanding")

    async def drive():
        for _ in range(12):
            async for _chunk in fan.complete_stream(max_tokens=1):
                pass

    run(drive())
    assert fan.dispatch_counts == [3, 3, 3, 3]


def test_round_robin_is_strictly_sequential():
    members = [FakeBackend(f"http://h:{800 + i}") for i in range(3)]
    fan = FanoutBackend(members, strategy="round-robin")

    async def drive():
        for _ in range(6):
            async for _chunk in fan.complete_stream(max_tokens=1):
                pass

    run(drive())
    assert fan.dispatch_counts == [2, 2, 2]


def test_capacity_sums_slots_but_not_context():
    """N instances of -np 8 really do admit 8N concurrent requests; per-request context is a
    property of one request and must not be summed."""
    fan = FanoutBackend([FakeBackend("http://h:1", max_concurrent=8) for _ in range(4)])
    cap = run(fan.capacity())
    assert cap.max_concurrent == 32
    assert cap.per_request_ctx == 4096
    assert cap.kv_bytes == 400


def test_metrics_snapshot_sums_counters_across_the_fleet():
    members = [FakeBackend("http://h:1", metrics={"vllm:num_preemptions_total": 3.0}),
               FakeBackend("http://h:2", metrics={"vllm:num_preemptions_total": 4.0})]
    fan = FanoutBackend(members)
    assert run(fan.metrics_snapshot())["vllm:num_preemptions_total"] == 7.0


def test_fanout_propagates_vllm_metrics_capability():
    """The runner keys Path B off this flag, so a wrapper must not hide it."""
    plain = FanoutBackend([FakeBackend("http://h:1")])
    assert plain.supports_vllm_metrics is False

    vllm_like = FakeBackend("http://h:1")
    vllm_like.supports_vllm_metrics = True
    assert FanoutBackend([vllm_like, FakeBackend("http://h:2")]).supports_vllm_metrics is True


def test_fanout_rejects_mixed_backend_types():
    a, b = FakeBackend("http://h:1"), FakeBackend("http://h:2")
    b.name = "other"
    with pytest.raises(ValueError, match="same backend type"):
        FanoutBackend([a, b])


def test_fanout_closes_every_member():
    members = [FakeBackend(f"http://h:{i}") for i in range(3)]
    run(FanoutBackend(members).close())
    assert all(m.closed for m in members)


def test_distribution_reports_the_actual_split():
    members = [FakeBackend(f"http://h:{i}") for i in range(2)]
    fan = FanoutBackend(members)

    async def drive():
        for _ in range(4):
            async for _c in fan.complete_stream(max_tokens=1):
                pass

    run(drive())
    dist = fan.distribution()
    assert dist["dispatch_counts"] == [2, 2]
    assert dist["dispatch_share"] == [0.5, 0.5]


# --- nginx config ---


def test_nginx_conf_disables_proxy_buffering():
    """Non-negotiable: with buffering on, nginx batches the SSE stream and every TTFT/ITL
    number downstream becomes an artifact of its buffer size."""
    from llmbench.suite.lb import nginx
    from llmbench.suite.plan import build_plan
    from llmbench.suite.spec import SuiteSpec
    from tests.test_suite_spec_plan import base_spec

    spec = SuiteSpec.from_dict(base_spec(
        lb={"kind": "nginx", "port": 18081},
        deployment={"backend": ["llamacpp"], "instances": [3]}))
    dep = build_plan(spec, make_topology()).groups[0][0]
    conf = nginx.render_conf(dep, spec, conf_dir=__import__("pathlib").Path("/tmp/c"),
                             log_dir=__import__("pathlib").Path("/tmp/l"))

    assert "proxy_buffering off;" in conf
    assert "proxy_request_buffering off;" in conf
    assert "least_conn;" in conf
    assert conf.count("server 127.0.0.1:81") == 3
    assert "listen 18081" in conf
    assert "proxy_next_upstream off;" in conf     # a silent retry would hide a failing backend
    assert conf.count("{") == conf.count("}")


def test_nginx_missing_binary_gives_an_actionable_error():
    from llmbench.suite.lb.nginx import NginxUnavailable, resolve_nginx

    with pytest.raises(NginxUnavailable) as e:
        resolve_nginx("definitely-not-a-real-binary-xyz")
    assert "apt-get install" in str(e.value)
    assert "kind: client" in str(e.value)


# --- offline tool process handling ---


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_timed_out_offline_tool_takes_its_whole_process_group_with_it(tmp_path):
    """`subprocess.run(timeout=...)` kills only the direct child. Combined with
    start_new_session=True that leaves every grandchild alive -- and `vllm bench` runs its
    engine in a child process, so a timed-out offline trial would strand an engine holding
    the cores and the KV allocation for the rest of the sweep, silently poisoning every
    measurement after it.

    The shell here stands in for that shape: `sh` is the direct child, `sleep` the grandchild.
    """
    import time

    from llmbench.suite.offline.base import run_tool

    pid_file = tmp_path / "grandchild.pid"
    run = run_tool(
        ["sh", "-c", f"sleep 120 & echo $! > {pid_file}; wait"],
        env_overrides={}, log_path=tmp_path / "tool.log", timeout_s=1.0,
    )

    assert not run.ok
    assert "timed out" in run.stderr
    grandchild = int(pid_file.read_text().strip())
    for _ in range(50):                      # SIGTERM -> exit is not instantaneous
        if not _alive(grandchild):
            break
        time.sleep(0.1)
    assert not _alive(grandchild), (
        f"grandchild {grandchild} survived the timeout -- the process group was not signalled"
    )


def test_offline_tool_that_cannot_be_executed_is_a_recorded_failure_not_a_crash(tmp_path):
    from llmbench.suite.offline.base import run_tool

    run = run_tool(["/nonexistent/llama-bench"], env_overrides={},
                   log_path=tmp_path / "tool.log", timeout_s=5.0)
    assert not run.ok
    assert "failed to execute" in run.stderr
    assert (tmp_path / "tool.log").exists()


# --- prefix-cache evidence is normalised across backends ---


def test_vllm_nested_cached_tokens_is_lifted_to_the_flat_key():
    """vLLM reports prefix-cache hits as usage.prompt_tokens_details.cached_tokens (OpenAI
    schema); llama.cpp reports a flat equivalent. The recorder reads one flat key for both,
    so without normalisation the field was None on every vLLM row -- making a prefix-caching
    experiment unfalsifiable, since a hit and a miss looked identical."""
    from llmbench.backends.vllm import VllmBackend

    b = VllmBackend("http://127.0.0.1:8200", None, 60.0)
    chunk = b._parse_event({
        "choices": [{"text": "x"}],
        "usage": {"prompt_tokens": 512, "completion_tokens": 8,
                  "prompt_tokens_details": {"cached_tokens": 256}},
    })
    assert chunk.usage["cached_tokens"] == 256


def test_a_flat_cached_tokens_is_left_alone():
    from llmbench.backends.vllm import VllmBackend

    b = VllmBackend("http://127.0.0.1:8200", None, 60.0)
    chunk = b._parse_event({"choices": [{"text": "x"}],
                            "usage": {"prompt_tokens": 512, "cached_tokens": 99}})
    assert chunk.usage["cached_tokens"] == 99


def test_usage_without_cache_details_is_untouched():
    from llmbench.backends.vllm import VllmBackend

    b = VllmBackend("http://127.0.0.1:8200", None, 60.0)
    chunk = b._parse_event({"choices": [{"text": "x"}], "usage": {"prompt_tokens": 512}})
    assert chunk.usage["prompt_tokens"] == 512
    assert chunk.usage.get("cached_tokens") is None


def test_llamacpp_cache_n_is_exposed_as_cached_tokens():
    """llama.cpp reports prompt-cache reuse as timings.cache_n, vLLM as a nested usage field.
    The recorder reads one flat usage["cached_tokens"], and the report has a column for that
    and none for server_cache_n -- so llama.cpp's cache column was blank even when the server
    had reported the hit."""
    from llmbench.backends.llamacpp import LlamaCppBackend

    b = LlamaCppBackend("http://127.0.0.1:8100", None, 60.0)
    chunk = b._parse_event({
        "choices": [{"text": "x"}],
        "usage": {"prompt_tokens": 1024, "completion_tokens": 1},
        "timings": {"cache_n": 897, "prompt_n": 127, "prompt_ms": 550.0},
    })
    assert chunk.usage["cached_tokens"] == 897
    assert chunk.server_timings["cache_n"] == 897        # native field left intact


def test_llamacpp_timings_without_usage_still_carry_the_cache_hit():
    from llmbench.backends.llamacpp import LlamaCppBackend

    b = LlamaCppBackend("http://127.0.0.1:8100", None, 60.0)
    chunk = b._parse_event({"choices": [{"text": "x"}], "timings": {"cache_n": 513}})
    assert chunk.usage["cached_tokens"] == 513


def test_llamacpp_zero_cache_n_is_reported_as_zero_not_dropped():
    """A miss must read as 0, not as 'not measured' -- those mean different things when the
    whole point of the row is proving the cache did not hit."""
    from llmbench.backends.llamacpp import LlamaCppBackend

    b = LlamaCppBackend("http://127.0.0.1:8100", None, 60.0)
    chunk = b._parse_event({"choices": [{"text": "x"}], "timings": {"cache_n": 0}})
    assert chunk.usage["cached_tokens"] == 0


def test_llamacpp_without_timings_is_untouched():
    from llmbench.backends.llamacpp import LlamaCppBackend

    b = LlamaCppBackend("http://127.0.0.1:8100", None, 60.0)
    chunk = b._parse_event({"choices": [{"text": "x"}], "usage": {"prompt_tokens": 8}})
    assert chunk.usage == {"prompt_tokens": 8}
