"""Shared SSE streaming parser and OAI-compatible request building.

SSE framing replicates vLLM's own reference client (StreamedResponseHandler,
vllm/benchmarks/lib/endpoint_request_func.py:23-62, see docs/reference-notes.md §3):
incremental UTF-8 decoding (safe against multi-byte chars split across TCP reads), "\\n\\n"
event framing, JSON-completeness gating (a message is only yielded once its data: payload
parses cleanly), `[DONE]` and comment-line (":"-prefixed) handling, whitespace-only reads
dropped. This is the unit-test target called out in spec §14 as the most valuable test.
"""
from __future__ import annotations

import codecs
import json
from typing import Union

DONE = "[DONE]"


class SSEParser:
    """Feed raw byte chunks (as they arrive off the wire); get back parsed events.

    Does not assume line-buffering from the HTTP client -- chunks may split a UTF-8
    codepoint, a "data:" line, or the "\\n\\n" event terminator at an arbitrary byte
    boundary. Malformed/incomplete JSON is never surfaced as a bad event; the parser holds
    the partial frame and waits for more bytes.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buf = ""

    def feed(self, chunk: bytes) -> list[Union[dict, str]]:
        events: list[Union[dict, str]] = []
        text = self._decoder.decode(chunk)
        if not text:
            return events
        self._buf += text
        while "\n\n" in self._buf:
            raw_event, rest = self._buf.split("\n\n", 1)
            raw_event_stripped = raw_event.strip()
            if not raw_event_stripped:
                self._buf = rest
                continue
            data_lines = []
            for line in raw_event_stripped.split("\n"):
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            if not data_lines:
                self._buf = rest
                continue
            data = "\n".join(data_lines)
            if data == DONE:
                events.append(DONE)
                self._buf = rest
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                # Incomplete JSON despite a complete \n\n frame (rare, e.g. a proxy that
                # re-chunks mid-payload). Wait for more bytes rather than emit garbage.
                break
            events.append(obj)
            self._buf = rest
        return events

    def flush(self) -> list[Union[dict, str]]:
        """Call once the stream has ended, in case the final event lacked a trailing \\n\\n."""
        events: list[Union[dict, str]] = []
        raw_event = self._buf.strip()
        self._buf = ""
        if not raw_event:
            return events
        data_lines = []
        for line in raw_event.split("\n"):
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if not data_lines:
            return events
        data = "\n".join(data_lines)
        if data == DONE:
            return [DONE]
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            pass
        return events


def build_completions_payload(
    *,
    model: str,
    prompt: list[int] | str,
    max_tokens: int,
    ignore_eos: bool,
    stream: bool = True,
    extra: dict | None = None,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": ignore_eos,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if extra:
        payload.update(extra)
    return payload
