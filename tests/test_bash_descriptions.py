from __future__ import annotations

import pytest

from zevo.engine.agent.drivers._agent_loop import BASH_TOOL, emit_tool_call
from zevo.engine.agent.drivers._bash_description import describe_bash_call
from zevo.engine.agent.drivers.claude_cli import _normalise_claude_event
from zevo.engine.agent.drivers.codex_cli import _normalise_codex_event


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -n 'status' src tests", "Inspect files and state"),
        ("python3 - <<'PY'\nprint('ok')\nPY", "Run an inline Python script"),
        (
            "ssh -i /secret/key user@host <<'EOF'\nnvidia-smi\nEOF",
            "Check remote GPU availability",
        ),
        (
            "ssh user@host <<'EOF'\npython3 /remote/train.py\nEOF",
            "Run training on the assigned GPU",
        ),
        (
            "post_message() { echo x; }\npost_message 'Starting: validation'",
            "Post a ticket update",
        ),
        (
            "curl -fsS -X POST http://backend:8000/api/tickets/t-1/messages",
            "Post a ticket update",
        ),
        ("bash -lc 'ls -la /tmp/work'", "Inspect files and state"),
    ],
)
def test_bash_description_fallbacks(command: str, expected: str) -> None:
    assert describe_bash_call(command) == expected


def test_agent_description_wins_and_is_bounded() -> None:
    assert describe_bash_call("echo ignored", "  Validate   prediction schema  ") == (
        "Validate prediction schema"
    )
    assert len(describe_bash_call("echo ignored", "x" * 300)) == 160


def test_fallback_does_not_expose_ssh_credentials_or_paths() -> None:
    description = describe_bash_call(
        "ssh -i /secret/private-key.pem root@sensitive.example -- echo ok"
    )
    assert description == "Run a command on the assigned remote host"
    assert "secret" not in description
    assert "sensitive.example" not in description


def test_sdk_bash_schema_requires_description() -> None:
    assert set(BASH_TOOL["input_schema"]["required"]) == {"command", "description"}


def test_shared_tool_event_keeps_raw_input_and_adds_fallback() -> None:
    events: list[dict] = []
    args = {"command": "python3 - <<'PY'\nprint('ok')\nPY", "timeout_sec": 60}
    emit_tool_call(events.append, item_id="tool-1", tool="run_bash", args=args)
    payload = events[0]["payload"]
    assert payload["input"] == args
    assert payload["command"] == args["command"]
    assert payload["description"] == "Run an inline Python script"
    assert payload["description_source"] == "fallback"


def test_shared_tool_event_preserves_agent_description() -> None:
    events: list[dict] = []
    args = {"command": "python3 check.py", "description": "Validate output schema"}
    emit_tool_call(events.append, item_id="tool-1", tool="run_bash", args=args)
    payload = events[0]["payload"]
    assert payload["input"] == args
    assert payload["description"] == "Validate output schema"
    assert payload["description_source"] == "agent"


def test_claude_bash_event_adds_fallback_without_rewriting_input() -> None:
    raw_input = {"command": "ssh user@host <<'EOF'\nnvidia-smi\nEOF"}
    events = _normalise_claude_event({
        "type": "assistant",
        "message": {"content": [{
            "type": "tool_use", "id": "use-1", "name": "Bash",
            "input": raw_input,
        }]},
    })
    payload = events[0]["payload"]
    assert payload["input"] == raw_input
    assert "description" not in payload["input"]
    assert payload["description"] == "Check remote GPU availability"
    assert payload["description_source"] == "fallback"


def test_claude_bash_event_uses_authored_description() -> None:
    events = _normalise_claude_event({
        "type": "assistant",
        "message": {"content": [{
            "type": "tool_use", "id": "use-2", "name": "Bash",
            "input": {
                "command": "python3 validate.py",
                "description": "Validate generated artifacts",
            },
        }]},
    })
    assert events[0]["payload"]["description"] == "Validate generated artifacts"
    assert events[0]["payload"]["description_source"] == "agent"


def test_codex_command_event_gets_deterministic_fallback() -> None:
    events = _normalise_codex_event({
        "type": "item.started",
        "item": {
            "id": "item-1", "type": "command_execution",
            "command": "bash -lc 'rg -n status src'",
        },
    })
    payload = events[0]["payload"]
    assert payload["command"] == "bash -lc 'rg -n status src'"
    assert payload["description"] == "Inspect files and state"
    assert payload["description_source"] == "fallback"
