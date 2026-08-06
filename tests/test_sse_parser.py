"""Unit tests for the SSE parser (spec §14): frames split across TCP reads, [DONE], empty
chunks, malformed lines. This is the unit-level counterpart to the fake-server integration
test; see docs/reference-notes.md §3 for the framing rules being replicated.
"""
from llmbench.backends.openai_common import DONE, SSEParser


def test_single_event_single_feed():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\n\n')
    assert events == [{"a": 1}]


def test_event_split_across_two_reads():
    p = SSEParser()
    assert p.feed(b'data: {"a"') == []
    assert p.feed(b': 1}\n\n') == [{"a": 1}]


def test_multibyte_utf8_split_across_reads():
    import json as _json

    text = "héllo wörld"
    raw = _json.dumps({"text": text}).encode("utf-8")
    frame = b"data: " + raw + b"\n\n"
    p = SSEParser()
    events = []
    for i in range(len(frame)):
        events.extend(p.feed(frame[i : i + 1]))
    assert events == [{"text": text}]


def test_done_marker():
    p = SSEParser()
    events = p.feed(b"data: [DONE]\n\n")
    assert events == [DONE]


def test_empty_chunk_ignored():
    p = SSEParser()
    events = p.feed(b"\n\n")
    assert events == []


def test_comment_line_skipped():
    p = SSEParser()
    events = p.feed(b': keep-alive\n\ndata: {"a": 2}\n\n')
    assert events == [{"a": 2}]


def test_malformed_json_waits_for_more_bytes():
    p = SSEParser()
    # A "complete" \n\n frame whose data is not valid JSON alone.
    events = p.feed(b'data: {"a": \n\n')
    assert events == []  # held, not surfaced as garbage


def test_multiple_events_one_read():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\n\ndata: {"a": 2}\n\n')
    assert events == [{"a": 1}, {"a": 2}]


def test_flush_handles_trailing_event_without_terminator():
    p = SSEParser()
    assert p.feed(b'data: {"a": 1}') == []
    assert p.flush() == [{"a": 1}]


def test_multiline_data_field_joined_with_newline():
    p = SSEParser()
    events = p.feed(b'data: {"a":\ndata: 1}\n\n')
    assert events == [{"a": 1}]
