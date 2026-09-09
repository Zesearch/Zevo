"""Shared 'freeform' input contract for all agents.

A freeform ticket carries a natural-language request from a human plus
optional attachments. The runner dispatches it as a FreeformInput regardless
of which agent it is assigned to; the agent's freeform.md defines how to
interpret it.

The agent still returns its existing typed *Result schema so downstream
ref-resolution + the orchestrator's reactive loop keep working unchanged.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from zevo.contracts.memory import AgentMemoryContext


class FreeformInput(BaseModel):
    """Uniform worker input for a freeform Ticket."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(..., description="e.g. 'task-001'.")
    agent_id: str = Field(..., description="The agent this ticket is assigned to (echoed for the agent's convenience).")
    request: str = Field(
        ...,
        description=(
            "The user's natural-language request, verbatim. The agent should "
            "interpret it, post a one-line 'I understood you want X' message "
            "as its first action, then execute."
        ),
    )
    attachments: list[str] = Field(
        ...,
        description=(
            "Absolute paths to user-uploaded files inside the container "
            "(typically under /app/data/uploads/<uuid>/). Empty list "
            "if the user attached nothing."
        ),
    )
    work_dir: str = Field(
        ...,
        description="Per-ticket work dir the agent must write artifacts into.",
    )
    run_id: str = Field(..., description="Parent run id; '' if the ticket is standalone.")
    memory: AgentMemoryContext = Field(
        default_factory=AgentMemoryContext,
        description="Relevant memory from earlier Tickets of this Agent in this Run.",
    )
