"""Partial-message streaming -> `reasoning` transcript events.

With `--include-partial-messages` the Claude Code CLI streams extended
thinking as raw Anthropic `stream_event` lines (content_block_start /
content_block_delta / content_block_stop) BEFORE the complete assistant
message. These tests feed representative stream-json lines through the parser
and assert the reasoning TEXT is surfaced, without duplicating the complete
`thinking` block the CLI still delivers at message end, and without breaking
the existing agent_message / tool_call / tool_result / result parsing.
"""
from __future__ import annotations

import json

import pytest

from zevo.engine.agent.drivers.claude_cli import (
    _THINKING_BUFFER,
    _THINKING_EMITTED,
    _line_to_events,
)


@pytest.fixture(autouse=True)
def _reset_thinking_state():
    # Parser keeps module-level buffers; isolate every test.
    _THINKING_BUFFER.clear()
    _THINKING_EMITTED.clear()
    yield
    _THINKING_BUFFER.clear()
    _THINKING_EMITTED.clear()


def _events(*objs: dict) -> list[dict]:
    out: list[dict] = []
    for obj in objs:
        out.extend(_line_to_events(json.dumps(obj)))
    return out


def _stream(event: dict) -> dict:
    return {"type": "stream_event", "event": event}


def _thinking_start(index: int = 0) -> dict:
    return _stream({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "thinking", "thinking": ""},
    })


def _thinking_delta(text: str, index: int = 0) -> dict:
    return _stream({
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "thinking_delta", "thinking": text},
    })


def _block_stop(index: int = 0) -> dict:
    return _stream({"type": "content_block_stop", "index": index})


def test_partial_thinking_deltas_flush_one_reasoning_event() -> None:
    events = _events(
        _thinking_start(),
        _thinking_delta("Let me "),
        _thinking_delta("check the schema."),
        _block_stop(),
    )
    reasoning = [e for e in events if e["type"] == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["payload"]["text"] == "Let me check the schema."


def test_no_reasoning_until_block_stops() -> None:
    # Deltas alone accumulate silently; nothing surfaces until the flush.
    events = _events(_thinking_start(), _thinking_delta("half a thought"))
    assert [e for e in events if e["type"] == "reasoning"] == []
    stop = _line_to_events(json.dumps(_block_stop()))
    assert [e for e in stop if e["type"] == "reasoning"][0]["payload"]["text"] == (
        "half a thought"
    )


def test_complete_thinking_block_after_partials_is_not_duplicated() -> None:
    full = "Let me check the schema."
    events = _events(
        _thinking_start(),
        _thinking_delta("Let me "),
        _thinking_delta("check the schema."),
        _block_stop(),
        # The CLI ALSO delivers the complete assistant message at the end.
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": full},
            {"type": "text", "text": "Done."},
        ]}},
    )
    reasoning = [e for e in events if e["type"] == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["payload"]["text"] == full
    # The complete block's non-thinking content still parses.
    assert [e["payload"]["message"] for e in events if e["type"] == "agent_message"] == [
        "Done."
    ]


def test_complete_thinking_block_without_partials_still_emits() -> None:
    # Back-compat: if partial streaming is absent, the complete block path must
    # still surface the reasoning text (original behaviour).
    events = _events({"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "no partials here"},
    ]}})
    reasoning = [e for e in events if e["type"] == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["payload"]["text"] == "no partials here"


def test_partial_text_and_signature_deltas_are_swallowed() -> None:
    # Non-thinking partial shapes must not produce agent_message duplicates or
    # crash the parser; the complete assistant message carries the real text.
    events = _events(
        _stream({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "partial text"},
        }),
        _stream({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "abc123"},
        }),
        _stream({"type": "message_start", "message": {"role": "assistant"}}),
        _stream({"type": "message_stop"}),
    )
    assert events == []


def test_existing_event_types_still_parse_alongside_partials() -> None:
    events = _events(
        {"type": "system", "subtype": "init"},
        _thinking_start(),
        _thinking_delta("thinking about it"),
        _block_stop(),
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Running a tool"},
            {"type": "tool_use", "id": "tu_1", "name": "Bash",
             "input": {"command": "ls", "description": "List files"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": [{"type": "text", "text": "file.txt"}]},
        ]}},
        {"type": "result", "subtype": "success", "result": "all good",
         "usage": {"input_tokens": 10, "output_tokens": 5}},
    )
    kinds = [e["type"] for e in events]
    assert "reasoning" in kinds
    assert "agent_message" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "turn_completed" in kinds
    assert "task_complete" in kinds

    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["payload"]["tool"] == "Bash"
    assert tool_call["payload"]["description"] == "List files"

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["payload"]["output"] == "file.txt"
    assert tool_result["payload"]["is_error"] is False


def test_interleaved_thinking_blocks_dedupe_independently() -> None:
    # Two separate thinking blocks (indices reused across messages) each flush
    # once and each dedupe against their own complete block.
    events = _events(
        _thinking_start(0),
        _thinking_delta("first block", 0),
        _block_stop(0),
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "first block"},
        ]}},
        _thinking_start(0),
        _thinking_delta("second block", 0),
        _block_stop(0),
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "second block"},
        ]}},
    )
    texts = [e["payload"]["text"] for e in events if e["type"] == "reasoning"]
    assert texts == ["first block", "second block"]
