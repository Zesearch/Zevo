"""Run-scoped Agent memory retrieval and persistence.

Tickets stay immutable execution records.  This module supplies the separate
continuity layer that lets one worker remember verified lessons from its prior
Tickets in the same Run without relying on a provider conversation/session.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from zevo.contracts.memory import AgentMemoryContext, MemoryEntryView, MemoryUpdate
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db import AgentMemoryEntry, Run, Ticket, TicketNotice


_SECRET = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|oauth[_ -]?token|password|"
    r"client[_ -]?secret|private[_ -]?key|begin [a-z ]*private key)"
)
_PRIORITY = {
    "pitfall": 0,
    "verified_fact": 1,
    "runtime_finding": 2,
    "experiment_finding": 3,
    "artifact_reference": 4,
    "recommendation": 5,
}
_MAX_ENTRIES = 24
_MAX_CONTEXT_CHARS = 10_000


def _fingerprint(value: object) -> str:
    if value in (None, {}, [], ""):
        return ""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def applicability_for(run: Run, ticket: Ticket) -> dict[str, str]:
    """Return semantic identity only; transient runtime state is not memory.

    Artifact bindings and canonical Ticket payloads are the identity.
    """
    payload = dict(ticket.payload or {})
    inputs = dict(ticket.inputs or {})
    agent_id = ticket.agent_id

    if agent_id == "infrastructure":
        return {}
    if agent_id == "data":
        return {
            "data_source": _fingerprint({
                key: payload.get(key) for key in (
                    "dataset", "dataset_split", "dataset_config", "data_query",
                )
            }),
        }
    if agent_id == "train":
        return {
            "base_model": str(payload.get("base_model") or ""),
        }
    if agent_id == "inference":
        return {
            "base_model": str(payload.get("base_model") or ""),
        }
    if agent_id == "evaluation":
        return {
            "metric": str((ticket.payload or {}).get("metric") or ""),
            "evaluator": _fingerprint({
                key: (ticket.payload or {}).get(key)
                for key in ("evaluation_script", "answer_fields", "evaluation_config")
            }),
        }
    if agent_id == "registry":
        return {
            "metric": str(run.validation_metric or ""),
            "metric_direction": str(run.validation_metric_direction or ""),
            "base_model": str(payload.get("base_model") or ""),
        }
    return {}


def _applicable(stored: dict[str, Any], current: dict[str, str]) -> bool:
    """An empty stored dimension is general; every concrete one must match."""
    return all(not str(value or "") or current.get(key) == value
               for key, value in dict(stored or {}).items())


def _view(row: AgentMemoryEntry) -> MemoryEntryView:
    return MemoryEntryView(
        id=row.id,
        agent_id=row.agent_id,
        lane=row.lane,
        iteration=int(row.iteration or 0),
        kind=row.kind,
        key=row.key,
        summary=row.summary,
        details=dict(row.details or {}),
        visibility=row.visibility,
        applies_to={str(k): str(v) for k, v in dict(row.applies_to or {}).items()},
        source_ticket_id=row.source_ticket_id,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


async def load_memory_context(
    session: AsyncSession, *, run: Run, ticket: Ticket,
    current_applicability: dict[str, str] | None = None,
) -> AgentMemoryContext:
    """Load bounded, relevant memory for one activation.

    Held-out workers deliberately receive none: even lane-private adaptation
    would let later held-out measurements react to earlier Test behavior.  A
    terminal Run is audit-only and no longer injects memory into reruns.
    """
    empty = AgentMemoryContext(
        run_id=run.id, agent_id=ticket.agent_id, lane=ticket.lane, entries=[],
    )
    if ticket.lane != "optimization" or run.status in TERMINAL_RUN_STATUSES:
        return empty

    query = select(AgentMemoryEntry).where(
        AgentMemoryEntry.run_id == run.id,
        AgentMemoryEntry.lane == "optimization",
        AgentMemoryEntry.status == "active",
    )
    if ticket.agent_id == "orchestrator":
        query = query.where(AgentMemoryEntry.visibility == "shared_candidate")
    else:
        query = query.where(AgentMemoryEntry.agent_id == ticket.agent_id)
    rows = (await session.execute(
        query.order_by(desc(AgentMemoryEntry.created_at)).limit(100)
    )).scalars().all()

    current = current_applicability or applicability_for(run, ticket)
    if ticket.agent_id != "orchestrator":
        rows = [row for row in rows if _applicable(row.applies_to or {}, current)]
    rows.sort(key=lambda row: (
        _PRIORITY.get(row.kind, 99),
        -(row.created_at.timestamp() if row.created_at else 0),
    ))

    entries: list[MemoryEntryView] = []
    used = 0
    for row in rows:
        view = _view(row)
        size = len(view.model_dump_json())
        if entries and used + size > _MAX_CONTEXT_CHARS:
            continue
        entries.append(view)
        used += size
        if len(entries) >= _MAX_ENTRIES:
            break
    return empty.model_copy(update={"entries": entries})


def _forbidden_reason(update: MemoryUpdate, run: Run) -> str:
    text = json.dumps(update.model_dump(), ensure_ascii=False, default=str)
    if _SECRET.search(text):
        return "memory may not contain credentials or secrets"

    lowered = text.lower()
    if any(token in lowered for token in (
        "held_out_test", "heldout_test", "held-out test score",
        "baseline_test_score", "champion_test_score",
    )):
        return "optimization memory may not contain held-out Test information"
    # `Run.holdout` also carries validation metadata.  Validation belongs to
    # the optimization loop, so only compare against the four actual Test
    # assets instead of rejecting a perfectly valid validation finding.
    holdout = dict(run.holdout or {})
    for field in (
        "test_set", "test_public", "test_sample_submission",
    ):
        value = holdout.get(field)
        if not isinstance(value, str) or not value:
            continue
        path = value.replace("\\", "/")
        parts = [part for part in path.split("/") if part]
        needles = {path}
        if len(parts) >= 2:
            needles.add("/".join(parts[-2:]))
        if any(needle and needle.lower() in lowered for needle in needles):
            return "optimization memory may not contain a held-out Test path"
    return ""


async def persist_memory_updates(
    session: AsyncSession,
    *,
    run: Run,
    ticket: Ticket,
    updates: Iterable[MemoryUpdate],
    current_applicability: dict[str, str] | None = None,
) -> tuple[int, list[str]]:
    """Validate and append worker lessons, superseding matching active keys."""
    if (
        ticket.agent_id == "orchestrator"
        or ticket.lane != "optimization"
        or run.status in TERMINAL_RUN_STATUSES
    ):
        return 0, []

    applicable = current_applicability or applicability_for(run, ticket)
    existing = (await session.execute(select(AgentMemoryEntry).where(
        AgentMemoryEntry.run_id == run.id,
        AgentMemoryEntry.agent_id == ticket.agent_id,
        AgentMemoryEntry.lane == ticket.lane,
        AgentMemoryEntry.status == "active",
    ))).scalars().all()

    stored = 0
    rejected: list[str] = []
    seen_updates: set[tuple[str, str]] = set()
    for raw in updates:
        update = raw if isinstance(raw, MemoryUpdate) else MemoryUpdate.model_validate(raw)
        identity = (update.kind, update.key)
        if identity in seen_updates:
            rejected.append(
                f"{update.kind}/{update.key}: duplicate kind/key in one result"
            )
            continue
        seen_updates.add(identity)
        reason = _forbidden_reason(update, run)
        if reason:
            rejected.append(f"{update.kind}/{update.key}: {reason}")
            continue
        superseded = next((
            row for row in existing
            if row.kind == update.kind
            and row.key == update.key
            and dict(row.applies_to or {}) == applicable
        ), None)
        if superseded is not None:
            superseded.status = "superseded"
        entry = AgentMemoryEntry(
            run_id=run.id,
            agent_id=ticket.agent_id,
            lane=ticket.lane,
            iteration=int(ticket.iteration or 0),
            kind=update.kind,
            key=update.key,
            summary=update.summary,
            # Agent results use a strict list of key/value pairs so every LLM
            # provider can enforce the same output schema.  Keep the durable
            # DB/API audit shape unchanged: existing readers consume a mapping.
            details={item.key: item.value for item in update.details},
            visibility=update.visibility,
            applies_to=applicable,
            status="active",
            source_ticket_id=ticket.id,
            supersedes_id=superseded.id if superseded is not None else None,
        )
        session.add(entry)
        existing.append(entry)
        stored += 1

    for reason in rejected:
        session.add(TicketNotice(
            ticket_id=ticket.id,
            code="memory.rejected",
            severity="warning",
            body=f"**Agent memory update rejected.** {reason}"[:8000],
        ))
    return stored, rejected
