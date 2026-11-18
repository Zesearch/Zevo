"""Driver protocol -- how the harness invokes an agent.

A Driver wraps an underlying LLM CLI / SDK. Implementations:
  - ClaudeCliDriver -- shells out to `claude -p ... --output-format stream-json` (default).
  - BedrockDriver   -- AWS Bedrock Converse API via boto3.
  - StubDriver      -- test-only; returns canned artifacts, no LLM call.

The Driver doesn't know about the database or scheduler. It just runs the
agent and returns its structured output (or raises).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
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

    def __post_init__(self) -> None:
        from zevo.contracts.agent_submission import AgentSubmission
        if isinstance(self.output, AgentSubmission):
            self.output = self.output.to_result()


def submission_blueprint(blueprint: AgentBlueprint, input_payload: BaseModel) -> AgentBlueprint:
    """Bind the wire contract before both prompting and parsing, on every LLM driver."""
    from zevo.contracts._base import AgentResult
    from zevo.contracts.agent_submission import submission_schema
    schema = blueprint.output_schema
    if schema is None or not issubclass(schema, AgentResult):
        return blueprint
    operation = str(getattr(input_payload, "operation", ""))
    wire = submission_schema(schema, operation)
    return replace(blueprint, output_schema=wire, instructions=blueprint.instructions + (
        "\n\n# Final submission contract\n"
        "Use the operation-specific submission schema supplied with this activation. "
        "It replaces the full backend Result shape shown in historical transcripts or examples. "
        "Do not copy fields absent from this schema, even with empty/default values. "
        "Engine finalization fields are not agent output. "
        "Execution retry fields, when needed, belong INSIDE failure_recovery with "
        "scope=current_task, never at top level. Review scopes have no execution retry fields. "
        "Infra provision success reports device_info_path and null failure_resource; "
        "failed provision may report failure_resource; release reports released_resource."
    ))


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
