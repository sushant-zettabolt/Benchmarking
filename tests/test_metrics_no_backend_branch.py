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
