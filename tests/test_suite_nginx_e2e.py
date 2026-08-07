"""End-to-end test of the nginx load-balancing path, against fake SSE upstreams.

Fake upstreams rather than real backends on purpose. This host is shared and frequently
saturated by other people's jobs, so a throughput comparison against real inference is not
reproducible here -- but the properties that actually matter for the nginx path are not
throughput at all:

  * does nginx start unprivileged, from a generated config, and route to the fleet?
  * does it spread requests across instances?
  * does the token stream survive the proxy intact -- every chunk delivered, and TTFT/ITL
    still reflecting the delays the upstream actually injected rather than proxy artifacts?
  * is nginx actually stopped on teardown, and its port released?

All of these are load-independent, which is exactly why they are tested this way.

What these tests deliberately do NOT claim: that they detect `proxy_buffering`. That was
tried and does not work. Sweeping a fake SSE upstream from 40ms down to 0.2ms inter-token
intervals on nginx 1.18, buffering on and off produced identical median ITL to two decimal
places -- nginx forwards `text/event-stream` as it arrives so long as the client keeps up, so
there is no buffering signature to detect. The ITL assertion below therefore verifies
*fidelity* (the proxy did not distort the stream), which is the property that matters, and
does not pretend to be a regression test for the buffering directives.
"""
from __future__ import annotations

import asyncio
import shutil
import socket

import pytest

from llmbench.suite.deploy import port_is_free
from llmbench.suite.lb import make_backend
from llmbench.suite.lb.nginx import nginx_version, resolve_nginx
from llmbench.suite.plan import DeploymentPlan, InstancePlan
from llmbench.suite.spec import BackendSpec, SuiteSpec
from llmbench.suite.topology import AllocationPlan, CoreSet
from tests.fake_server.server import FakeServerConfig, FakeSSEServer

pytestmark = pytest.mark.skipif(
    shutil.which("nginx") is None and not any(
        __import__("pathlib").Path(d, "nginx").is_file()
        for d in ("/usr/sbin", "/usr/local/sbin", "/sbin")
    ),
    reason="nginx not installed",
)

TTFT_MS = 120.0
ITL_MS = 40.0
N_TOKENS = 12


class CountingFakeServer(FakeSSEServer):
    """Counts completion requests so we can see how nginx distributed them."""

    def __init__(self, config=None):
        super().__init__(config)
        self.completions = 0

    async def _handle(self, reader, writer):
        peeked = getattr(self, "_peek", None)
        await super()._handle(_CountingReader(reader, self), writer)


class _CountingReader:
    """Wraps the request reader to spot completion requests without touching the server."""

    def __init__(self, reader, owner):
        self._reader = reader
        self._owner = owner

    async def readline(self):
        line = await self._reader.readline()
        if line and b"/completions" in line or (line and b"/completion " in line):
            self._owner.completions += 1
        return line

    def __getattr__(self, name):
        return getattr(self._reader, name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_deployment(ports: list[int], lb_port: int) -> DeploymentPlan:
    """A DeploymentPlan pointing at already-running upstreams, with no CPU pinning.

    lb_placement is left None so nginx runs unpinned -- this test is about proxy behaviour,
    and pinning it would make the result depend on which cores happen to be free.
    """
    cores = CoreSet(index=0, cpus=[0], physical_cpus=[0], nodes=[0], sockets=[0],
                    ccds=[0], membind=[])
    instances = [InstancePlan(index=i, port=p, cores=cores) for i, p in enumerate(ports)]
    return DeploymentPlan(
        id="dtest", backend="llamacpp",
        backend_spec=BackendSpec(name="llamacpp", model="m", server_bin="x"),
        n_instances=len(ports), n_ctx=4096, n_parallel=8, batch=2048, ubatch=512,
        instances=instances,
        allocation=AllocationPlan(instances=[cores], budget=[0]),
        lb_kind="nginx", lb_placement=None, lb_port=lb_port,
    )


def make_spec(**kw) -> SuiteSpec:
    return SuiteSpec.from_dict({
        "name": "ngx", "out_dir": "/tmp/ignored",
        "lb": {"kind": "nginx", **kw.pop("lb", {})},
        "backends": {"llamacpp": {"model": "m", "server_bin": "x"}},
        "deployment": {"backend": ["llamacpp"], "instances": [2]},
        "request_timeout_s": 60,
        **kw,
    })


async def _drive(dep, spec, n_requests: int, concurrency: int = 1):
    """Send n_requests through the load balancer using the real backend client."""
    backend = make_backend(dep, spec)
    results = []

    async def one():
        import time
        t0 = time.perf_counter_ns()
        first = None
        stamps = []
        async for chunk in backend.complete_stream(
            token_ids=[1, 2, 3], max_tokens=N_TOKENS, ignore_eos=True, model="fake-model",
        ):
            if chunk.text:
                now = time.perf_counter_ns()
                if first is None:
                    first = now
                stamps.append(now)
        results.append({
            "ttft_ms": (first - t0) / 1e6 if first else None,
            "itls_ms": [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])],
            "n_chunks": len(stamps),
        })

    sem = asyncio.Semaphore(concurrency)

    async def bounded():
        async with sem:
            await one()

    await asyncio.gather(*(bounded() for _ in range(n_requests)))
    await backend.close()
    return results


@pytest.mark.asyncio
async def test_nginx_starts_routes_and_preserves_the_token_stream(tmp_path):
    """TTFT and ITL measured *through nginx* must still match what the upstream injected.

    This is a stream-fidelity check: it catches a proxy that drops chunks, coalesces them, or
    adds material latency of its own. It is not a test of the buffering directives (see the
    module docstring for why that cannot be tested this way)."""
    from llmbench.suite.lb import start_load_balancer

    cfg = FakeServerConfig(ttft_ms=TTFT_MS, itl_ms=ITL_MS, n_tokens=N_TOKENS)
    async with CountingFakeServer(cfg) as a, CountingFakeServer(cfg) as b:
        lb_port = free_port()
        dep = make_deployment([a.port, b.port], lb_port)
        spec = make_spec()

        mp = await start_load_balancer(dep, spec, out_dir=tmp_path)
        try:
            assert mp.alive
            assert not port_is_free(lb_port), "nginx did not bind its listen port"

            results = await _drive(dep, spec, n_requests=6, concurrency=2)
            assert len(results) == 6
            assert all(r["n_chunks"] == N_TOKENS for r in results), \
                f"stream truncated through proxy: {[r['n_chunks'] for r in results]}"

            # TTFT through the proxy must still reflect the upstream's injected delay.
            ttfts = [r["ttft_ms"] for r in results]
            assert all(t is not None for t in ttfts)
            assert min(ttfts) >= TTFT_MS * 0.8, f"TTFT too fast to be real: {ttfts}"
            assert max(ttfts) <= TTFT_MS * 4, f"nginx inflated TTFT: {ttfts}"

            # Stream fidelity: inter-token gaps through the proxy must still cluster around
            # the interval the upstream injected. A proxy that coalesced or re-timed chunks
            # would move this median away from ITL_MS in one direction or the other.
            all_itls = [x for r in results for x in r["itls_ms"]]
            assert all_itls
            median = sorted(all_itls)[len(all_itls) // 2]
            assert ITL_MS * 0.5 <= median <= ITL_MS * 3, \
                f"median ITL through the proxy is {median:.1f}ms, but the upstream injected " \
                f"{ITL_MS}ms -- the proxy is distorting the stream"

            # Both upstreams should have seen traffic.
            assert a.completions > 0 and b.completions > 0, \
                f"nginx did not spread across the fleet: {a.completions}/{b.completions}"
            assert a.completions + b.completions == 6
        finally:
            mp.terminate()

        assert not mp.alive
        for _ in range(60):
            if port_is_free(lb_port):
                break
            await asyncio.sleep(0.25)
        assert port_is_free(lb_port), "nginx port still bound after teardown"


@pytest.mark.asyncio
async def test_nginx_single_instance_uniform_routing(tmp_path):
    """With lb.uniform the 1-instance layout also traverses the proxy, so its latency is
    comparable with the N-instance layouts rather than being the only one without a hop."""
    from llmbench.suite.lb import start_load_balancer

    cfg = FakeServerConfig(ttft_ms=50.0, itl_ms=10.0, n_tokens=5)
    async with CountingFakeServer(cfg) as a:
        lb_port = free_port()
        dep = make_deployment([a.port], lb_port)
        spec = make_spec()
        mp = await start_load_balancer(dep, spec, out_dir=tmp_path)
        try:
            results = await _drive(dep, spec, n_requests=3)
            assert a.completions == 3
            assert all(r["n_chunks"] == 5 for r in results)
        finally:
            mp.terminate()


@pytest.mark.asyncio
async def test_nginx_reports_upstream_failure_rather_than_masking_it(tmp_path):
    """proxy_next_upstream is off, so a dead instance must surface as an error rather than
    being silently retried onto a healthy peer and timed as one slow request."""
    from llmbench.suite.lb import start_load_balancer

    cfg = FakeServerConfig(ttft_ms=20.0, itl_ms=5.0, n_tokens=3)
    async with CountingFakeServer(cfg) as a:
        dead_port = free_port()          # nothing listening here
        lb_port = free_port()
        dep = make_deployment([a.port, dead_port], lb_port)
        spec = make_spec()
        mp = await start_load_balancer(dep, spec, out_dir=tmp_path)
        try:
            backend = make_backend(dep, spec)
            statuses = []
            for _ in range(6):
                try:
                    n = 0
                    async for chunk in backend.complete_stream(
                        token_ids=[1], max_tokens=3, ignore_eos=True, model="fake-model",
                    ):
                        if chunk.text:
                            n += 1
                    statuses.append(n)
                except Exception as e:  # noqa: BLE001
                    statuses.append(type(e).__name__)
            await backend.close()
            # Some requests hit the dead upstream; those must fail visibly, not be masked.
            assert any(isinstance(s, str) or s == 0 for s in statuses), \
                f"a dead upstream produced no visible failure: {statuses}"
        finally:
            mp.terminate()


def test_nginx_version_detection():
    exe = resolve_nginx("nginx")
    version = nginx_version(exe)
    assert version and len(version) == 3, f"could not parse nginx version from {exe}"


def test_dash_e_flag_matches_nginx_version(tmp_path):
    """`-e` only exists from 1.19.5; passing it to an older nginx is a hard failure
    ('invalid option: "e"'), which is how this was found on the 1.18 build here."""
    from llmbench.suite.lb import nginx as ngx

    exe = resolve_nginx("nginx")
    dep = make_deployment([9001], 9000)
    argv = ngx.build_argv(exe, tmp_path / "nginx.conf", tmp_path, dep, make_spec())
    if nginx_version(exe) >= ngx._MIN_DASH_E:
        assert "-e" in argv
    else:
        assert "-e" not in argv


# --- contention detection ---


def test_contention_snapshot_reads_real_cpu_state():
    """Sanity check against /proc/stat: percentages must be in range and self-consistent."""
    from llmbench.suite.contention import snapshot

    s = snapshot([0, 1, 2, 3], window_s=0.3)
    assert s.n_cpus == 4
    assert 0.0 <= s.busy_pct <= 100.5          # tiny overshoot possible from tick rounding
    assert s.foreign_cores <= s.busy_cores + 1e-6
    assert s.window_s > 0


def test_contention_attributes_our_own_load_correctly():
    """A CPU burner we own must land in `our_cores`, not be reported as foreign -- otherwise
    every busy benchmark would flag itself as contaminated."""
    import multiprocessing as mp
    import os
    import time

    from llmbench.suite.contention import ContentionMonitor

    def burn(stop_after: float):
        end = time.monotonic() + stop_after
        while time.monotonic() < end:
            pass

    proc = mp.Process(target=burn, args=(1.5,))
    proc.start()
    try:
        allowed = sorted(os.sched_getaffinity(0))
        mon = ContentionMonitor(cpus=allowed, pids=[proc.pid])
        mon.start()
        time.sleep(1.0)
        sample = mon.sample()
        # The burner is ours, so it must be attributed to us rather than inflating foreign.
        assert sample.our_cores > 0.3, f"own load not attributed: {sample.to_dict()}"
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


def test_contention_warning_threshold():
    from llmbench.suite.contention import ContentionSample

    quiet = ContentionSample(n_cpus=96, busy_cores=50.0, our_cores=48.0, foreign_cores=2.0)
    assert quiet.warning(15.0) is None

    contended = ContentionSample(n_cpus=96, busy_cores=90.0, our_cores=10.0, foreign_cores=80.0)
    msg = contended.warning(15.0)
    assert msg and "contaminated" in msg and "80.0 core(s)" in msg
