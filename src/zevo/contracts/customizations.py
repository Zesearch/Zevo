"""Customized Pipeline — per-agent user customizations.

A *Customized Pipeline* (mode="customized_pipeline") is a guided pipeline: the
USER fills in one customization block per configurable agent up front to pin
or guide the responsible Specialist's execution configuration. Strict blocks are binding;
advisory blocks may use a documented fallback.

The deterministic Evaluation runner is intentionally absent: its only inputs
come from the frozen scoring contract on the Run. `RunCustomizations` is
persisted on the Run and handed to the orchestrator. At Ticket creation,
strict `parameters` become `configuration_pins` and advisory
ones become `configuration_suggestions`; instructions/paths/output/enforcement
remain in the Ticket's customization block. No execution value is stored twice.

Every block shares the universal fields (`instructions`, `input_paths`,
`output_dir`, and `enforcement`); agent-specific knobs live in a centrally
validated `parameters` dict. Known parameter keys per agent:

  infrastructure : no agent-specific parameters; it derives the resource plan
                   from Run context while provider/GPU maximum remain Run-owned
  data           : method_ids, target_size
  train          : batch_size, learning_rate, num_epochs,
                   lora_r, lora_alpha, max_seq_len
  inference      : prompt/mapping/decoding pins use UserRequest's canonical
                   fields and are displayed inside the Inference section
  registry       : no agent-specific parameters; model identity is Run-owned
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The Agents a customization block may target (pipeline order).
CUSTOMIZABLE_AGENTS: tuple[str, ...] = (
    "infrastructure", "data", "train", "inference", "registry",
)

# `training_method`, `base_model`, `data_query`, runtime, and scoring assets
# deliberately do not appear here: each has one canonical home in UserRequest
# or the CreateRunRequest envelope. Customizations only hold Agent-local knobs.
CUSTOMIZATION_PARAMETER_KEYS: dict[str, frozenset[str]] = {
    "infrastructure": frozenset(),
    "data": frozenset({"method_ids", "target_size"}),
    "train": frozenset({
        "batch_size", "learning_rate", "num_epochs", "lora_r",
        "lora_alpha", "max_seq_len",
    }),
    "inference": frozenset(),
    "registry": frozenset(),
}

_POSITIVE_INT_PARAMETERS = {"target_size", "batch_size", "num_epochs", "max_seq_len"}
_NONNEGATIVE_INT_PARAMETERS = {"lora_r", "lora_alpha"}


class AgentCustomization(BaseModel):
    """The user's marching orders for one Agent in a Customized Pipeline."""

    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(
        "",
        description="Free-text instructions for this Agent.",
    )
    input_paths: list[str] = Field(
        default_factory=list,
        description="Files/folders the agent should use (uploaded zips are extracted; the path is recorded).",
    )
    output_dir: str = Field(
        "",
        description="Where the agent should write its outputs. Empty = the run's default per-ticket workspace.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific knobs (e.g. train: batch_size/learning_rate/lora_r). Passed through verbatim.",
    )
    enforcement: Literal["strict", "advisory"] = Field(
        "strict",
        description="strict rejects deviations; advisory permits a documented fallback.",
    )

    def is_empty(self) -> bool:
        return not (
            self.instructions.strip() or self.input_paths or
            self.output_dir.strip() or self.parameters
        )


class RunCustomizations(BaseModel):
    """Per-Agent customization blocks, keyed by Agent id."""

    model_config = ConfigDict(extra="forbid")

    agents: dict[str, AgentCustomization] = Field(default_factory=dict)
    note: str = Field("", description="Optional global note shown to the orchestrator.")

    @model_validator(mode="after")
    def validate_agent_parameters(self) -> "RunCustomizations":
        unknown_agents = sorted(set(self.agents) - set(CUSTOMIZABLE_AGENTS))
        if unknown_agents:
            raise ValueError(f"unknown customization agent(s): {unknown_agents}")
        for agent_id, block in self.agents.items():
            allowed = CUSTOMIZATION_PARAMETER_KEYS[agent_id]
            unknown = sorted(set(block.parameters) - allowed)
            if unknown:
                raise ValueError(
                    f"unsupported {agent_id} customization parameter(s): {unknown}; "
                    f"allowed: {sorted(allowed)}"
                )
            for key, value in block.parameters.items():
                if key == "method_ids":
                    normalized = (
                        [item.strip() for item in value]
                        if isinstance(value, list)
                        and all(isinstance(item, str) for item in value)
                        else []
                    )
                    if (
                        not normalized or any(not item for item in normalized)
                        or len(normalized) != len(set(normalized))
                    ):
                        raise ValueError("data method_ids must be a non-empty unique string list")
                    block.parameters[key] = normalized
                elif key in _POSITIVE_INT_PARAMETERS:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                        raise ValueError(f"{key} must be an integer >= 1")
                elif key in _NONNEGATIVE_INT_PARAMETERS:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"{key} must be an integer >= 0")
                elif key == "learning_rate":
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or float(value) <= 0
                    ):
                        raise ValueError("learning_rate must be a number > 0")
        return self

    def for_agent(self, agent_id: str) -> AgentCustomization | None:
        d = self.agents.get(agent_id)
        return d if (d and not d.is_empty()) else None
