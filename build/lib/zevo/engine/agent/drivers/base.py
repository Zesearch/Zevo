"""Driver protocol -- how the harness invokes an agent.

A Driver wraps an underlying LLM CLI / SDK. Implementations:
  - ClaudeCliDriver -- shells out to `claude -p ... --output-format stream-json` (default).
  - BedrockDriver   -- AWS Bedrock Converse API via boto3.
  - StubDriver      -- test-only; returns canned artifacts, no LLM call.

The Driver doesn't know about the database or scheduler. It just runs the
agent and returns its structured output (or raises).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint


@dataclass
class DriverRunResult:
    """What every driver returns."""

    output: BaseModel
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: int = 0
    driver: str = ""
    model: str = ""


class Driver(Protocol):
    """All concrete drivers implement this."""

    name: str

    async def run_agent(
        self,
        *,
        blueprint: AgentBlueprint,
        input_payload: BaseModel,
        workspace_dir: str,
        stdout_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        conversation: list[dict[str, Any]] | None = None,
        max_turns: int = 60,
        model_override: str = "",
        sandbox_mode: str = "",   # per-agent 'none'|'openshell' override; CLI drivers only
    ) -> DriverRunResult: ...
    # event_sink, if supplied, gets one dict per structured event:
    #   {"type": "agent_message" | "tool_call" | "phase" | "progress" | ...,
    #    "payload": {...}}
    # Drivers that don't emit JSON events (bare-stdout drivers) MAY
    # ignore event_sink entirely; the harness falls back to stdout_sink in
    # that case.
