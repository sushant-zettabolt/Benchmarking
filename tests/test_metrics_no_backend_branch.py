"""Enforces docs/contract.md's single-implementation rule: metrics.py must never branch on
backend identity. See llmbench/metrics.py's module docstring.

Checks precise code patterns (imports, attribute access, backend-name literals), not the
English word "backend"/"backends" in prose -- this module's own docstring legitimately
discusses the rule it enforces.
"""
import ast
import re
from pathlib import Path

METRICS_PATH = Path(__file__).parent.parent / "llmbench" / "metrics.py"


def _code_only(src: str) -> str:
    """Strip the module docstring (first triple-quoted string) so prose mentioning
    "backend" doesn't trip the literal-string checks below."""
    tree = ast.parse(src)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        docstring_end = tree.body[0].end_lineno
        return "\n".join(src.splitlines()[docstring_end:])
    return src


def test_metrics_module_does_not_import_backends():
    tree = ast.parse(METRICS_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or "backends" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "backends" not in alias.name


def test_metrics_module_never_reads_a_dot_backend_attribute():
    code = _code_only(METRICS_PATH.read_text())
    assert not re.search(r"\.backend\b", code), "metrics.py must not read a `.backend` field"


def test_metrics_module_has_no_backend_name_literals():
    code = _code_only(METRICS_PATH.read_text())
    for literal in ('"llamacpp"', "'llamacpp'", '"vllm"', "'vllm'"):
        assert literal not in code, f"metrics.py must not branch on the literal {literal}"


def test_metrics_module_has_no_if_backend_conditionals():
    tree = ast.parse(METRICS_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.dump(node.test)
            assert "backend" not in test_src.lower(), "found an `if` branching on backend identity"


def test_cached_tokens_falls_back_to_the_server_side_field():
    """The two servers name the same quantity differently, and records written before the
    backend-side normalisation existed carry only one of the fields. One report column has to
    stay correct for both, including for runs already on disk."""
    from llmbench import metrics

    base = dict(t_send_ns=0, t_first_token_ns=1_000_000, t_end_ns=2_000_000,
                n_prompt_actual=1024, n_gen_actual=1)
    only_server = [{**base, "server_cache_n": 897}]
    only_usage = [{**base, "cached_tokens": 512}]
    neither = [dict(base)]

    assert metrics.aggregate_client(only_server, "pp1024").cached_tokens_mean == 897
    assert metrics.aggregate_client(only_usage, "pp1024").cached_tokens_mean == 512
    assert metrics.aggregate_client(neither, "pp1024").cached_tokens_mean is None


def test_a_zero_cache_hit_is_not_mistaken_for_missing_data():
    """0 means 'measured, and nothing was cached' -- the whole point of a miss arm. It must
    not fall through to the other field or read as unmeasured."""
    from llmbench import metrics

    base = dict(t_send_ns=0, t_first_token_ns=1_000_000, t_end_ns=2_000_000,
                n_prompt_actual=1024, n_gen_actual=1)
    assert metrics.aggregate_client([{**base, "cached_tokens": 0}], "pp1024").cached_tokens_mean == 0
    assert metrics.aggregate_client([{**base, "server_cache_n": 0}], "pp1024").cached_tokens_mean == 0


def test_prometheus_cached_tokens_is_per_request_not_a_trial_total():
    """vllm:prompt_tokens_cached_total is a counter over the whole trial. Written undivided
    into a per-request field it reads as one request having cached tens of thousands of
    tokens -- measured live: 34048 on a 1024-token prompt, which is 896 x 38 requests. Every
    other quantity in _attach_vllm_metrics_delta was already divided by the request count."""
    from llmbench.records import RawRecord
    from llmbench.runner import _attach_vllm_metrics_delta

    before = {"vllm:request_prefill_time_seconds_count": 0, "vllm:prompt_tokens_cached_total": 0,
              "vllm:request_prefill_time_seconds_sum": 0.0}
    after = {"vllm:request_prefill_time_seconds_count": 38,
             "vllm:prompt_tokens_cached_total": 34048,
             "vllm:request_prefill_time_seconds_sum": 19.0}

    rec = RawRecord()
    _attach_vllm_metrics_delta(rec, before, after)
    assert rec.cached_tokens == 896, "expected the per-request average, not the trial total"


def test_prometheus_cache_counter_without_a_request_count_is_not_guessed():
    from llmbench.records import RawRecord
    from llmbench.runner import _attach_vllm_metrics_delta

    rec = RawRecord()
    _attach_vllm_metrics_delta(rec, {"vllm:prompt_tokens_cached_total": 0},
                               {"vllm:prompt_tokens_cached_total": 7680})
    assert rec.cached_tokens is None


def test_a_cache_count_larger_than_the_prompt_is_flagged():
    """A request cannot reuse more cached prompt tokens than its prompt contains. When the
    number says otherwise the field is not per-request, and the value is plausible enough at
    a glance to be quoted as a result -- a trial-wide counter landing in a per-request field
    read as a 3300% hit rate on a 1024-token prompt."""
    from llmbench import metrics

    base = dict(t_send_ns=0, t_first_token_ns=1_000_000, t_end_ns=2_000_000,
                n_prompt_actual=1024, n_gen_actual=1)
    bad = metrics.aggregate_client([{**base, "cached_tokens": 34048}], "pp1024")
    assert "cached_tokens_exceeds_prompt" in bad.flags

    ok = metrics.aggregate_client([{**base, "cached_tokens": 896}], "pp1024")
    assert "cached_tokens_exceeds_prompt" not in ok.flags


def test_a_full_prompt_cache_hit_is_not_flagged():
    """cached == prompt is legitimate: a repeated identical prompt is entirely reusable."""
    from llmbench import metrics

    row = metrics.aggregate_client([dict(
        t_send_ns=0, t_first_token_ns=1_000_000, t_end_ns=2_000_000,
        n_prompt_actual=1024, n_gen_actual=1, cached_tokens=1024)], "pp1024")
    assert "cached_tokens_exceeds_prompt" not in row.flags
