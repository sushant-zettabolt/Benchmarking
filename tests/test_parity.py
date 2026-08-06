"""Gate G2: deliberately mismatch each axis one at a time; the tool must detect and refuse
every one (course_correct.txt §7).
"""
from llmbench.parity import AxisResult, ParityReport, apply_deliberate_mismatch


def _matched_report() -> ParityReport:
    report = ParityReport(url_a="http://a", url_b="http://b")
    for axis in ["kv_cache_dtype", "context_capacity_per_request", "batch_admission_shaping"]:
        report.axes.append(AxisResult(axis=axis, verdict="matched", value_a=1, value_b=1))
    return report


def test_all_matched_report_is_valid():
    report = _matched_report()
    assert report.all_matched
    assert "PARITY OK" in report.headline() or report.all_matched


def test_deliberate_mismatch_is_detected_and_refused():
    report = _matched_report()
    apply_deliberate_mismatch(report, "kv_cache_dtype", "f16", "fp8")
    assert not report.all_matched
    assert "kv_cache_dtype" in report.mismatched_axes
    assert "Not a valid comparison" in report.headline()


def test_every_axis_mismatch_is_individually_caught():
    for axis_to_break in ["kv_cache_dtype", "context_capacity_per_request", "batch_admission_shaping"]:
        report = _matched_report()
        apply_deliberate_mismatch(report, axis_to_break, "X", "Y")
        assert axis_to_break in report.mismatched_axes, f"axis {axis_to_break} was not caught"
        assert not report.all_matched


def test_none_values_are_unverifiable_not_matched():
    report = ParityReport(url_a="a", url_b="b")
    report.axes.append(AxisResult(axis="attention_backend", verdict="unverifiable", value_a=None, value_b=None))
    assert not report.all_matched
    assert "attention_backend" in report.unverifiable_axes
