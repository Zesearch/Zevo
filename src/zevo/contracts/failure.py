"""A Specialist's recovery proposal; the engine retains routing authority."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


RecoveryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FailureRecovery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: Literal[
        "current_task", "upstream_input", "resource_plan", "fixed_configuration", "run_decision",
    ] = Field(description=(
        "Authority needed for the next action. current_task covers implementation repair "
        "or a bounded diagnostic within the existing work order, including executing a "
        "corrected workload again on the same resources (use execution_retry_reason). "
        "Replacing unusable compute within the same requirements uses execution_retry_action=replace_resource. resource_plan means the resource requirements must CHANGE; needing "
        "a new submit contract alone is current_task. Other scopes request "
        "Orchestrator review; they do not authorize changing resources, pins or decisions."
    ))
    reason: RecoveryText
    attempted_actions: list[RecoveryText] = Field(default_factory=list)
    evidence: list[RecoveryText] = Field(min_length=1, description=(
        "Concrete observations and their artifact/log references. Distinguish verified "
        "facts from hypotheses; do not include held-out answers or scores."
    ))
    next_action: RecoveryText = Field(description=(
        "Specific repair, discriminating diagnostic, or decision requested, and what it "
        "will resolve. Repeating the same failed action is not a recovery plan."
    ))
