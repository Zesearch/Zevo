"""Run-scoped Agent memory contracts.

Provider conversations are caches, not durable experiment state.  These models
define the small, structured lessons Zevo persists and injects across distinct
Tickets belonging to the same Run.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MemoryKind = Literal[
    "verified_fact",
    "pitfall",
    "runtime_finding",
    "experiment_finding",
    "artifact_reference",
    "recommendation",
]
MemoryVisibility = Literal["agent_local", "shared_candidate"]


class MemoryDetail(BaseModel):
    """One portable evidence item in a worker-authored memory update.

    Open-ended JSON objects cannot be represented by provider-enforced strict
    output schemas.  A bounded string pair keeps the wire contract identical
    across Claude, Codex, Bedrock, and OpenRouter while the persistence layer
    can still expose the established dictionary-shaped audit view.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=6000)

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        key = value.strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", key):
            raise ValueError(
                "detail key must start with a letter and contain only lowercase "
                "letters, numbers, underscore, dot, or dash"
            )
        return key

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("detail value must not be blank")
        return normalized


class MemoryUpdate(BaseModel):
    """One durable lesson reported by a successful worker activation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: MemoryKind
    key: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Stable snake/dot/dash key. A later update with the same kind/key "
            "and applicability supersedes this one without deleting history."
        ),
    )
    summary: str = Field(
        min_length=1,
        max_length=1000,
        description="Concise verified lesson, pitfall, or recommendation.",
    )
    details: list[MemoryDetail] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Small structured evidence as strict key/value string pairs; never "
            "an open JSON object, raw transcript, or secret."
        ),
    )
    visibility: MemoryVisibility = "agent_local"

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        key = value.strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", key):
            raise ValueError(
                "key must start with a letter and contain only lowercase "
                "letters, numbers, underscore, dot, or dash"
            )
        return key

    @field_validator("summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        summary = " ".join(value.split())
        if not summary:
            raise ValueError("summary must not be blank")
        return summary

    @model_validator(mode="after")
    def bound_details(self) -> "MemoryUpdate":
        keys = [item.key for item in self.details]
        if len(keys) != len(set(keys)):
            raise ValueError("details must not contain duplicate keys")
        try:
            encoded = json.dumps(
                [item.model_dump() for item in self.details],
                sort_keys=True,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("details must be JSON serializable") from exc
        if len(encoded) > 6000:
            raise ValueError("details must serialize to at most 6000 characters")
        if self.visibility == "shared_candidate" and self.kind not in {
            "verified_fact", "experiment_finding", "recommendation",
        }:
            raise ValueError(
                "shared_candidate is reserved for verified cross-Agent facts, "
                "experiment findings, or recommendations; keep pitfalls/runtime "
                "details/artifact references agent_local"
            )
        return self


class MemoryEntryView(BaseModel):
    """A validated database entry supplied to an Agent or API reader."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    lane: Literal["optimization", "held_out_test"]
    iteration: int = Field(ge=0)
    kind: MemoryKind
    key: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    visibility: MemoryVisibility
    applies_to: dict[str, str] = Field(default_factory=dict)
    source_ticket_id: str
    created_at: str = ""


class AgentMemoryContext(BaseModel):
    """Bounded Run-local memory injected into one Agent activation."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["run"] = "run"
    run_id: str = ""
    agent_id: str = ""
    lane: Literal["optimization", "held_out_test"] = "optimization"
    entries: list[MemoryEntryView] = Field(default_factory=list)
