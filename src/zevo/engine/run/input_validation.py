"""Versioned review events for work orders rejected before Agent execution."""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import select

from zevo.db import AgentWakeupRequest, Ticket, TicketNotice
from zevo.contracts.tickets import TERMINAL_RUN_STATUSES

NOTICE_CODE = "ticket.input_validation_failed"
EVENTS_KEY = "input_validation_failures"


def merge_input_reviews(existing, incoming):
    """Keep every review identity without making unrelated wakes skippable."""
    existing, incoming = existing or {}, incoming or {}
    events = list(existing.get(EVENTS_KEY, []))
    for event in incoming.get(EVENTS_KEY, []):
        if event not in events:
            events.append(event)
    if not events:
        return existing
    return {**existing, EVENTS_KEY: events, "input_validation_review_only": bool(
        existing.get(EVENTS_KEY) and incoming.get(EVENTS_KEY)
        and existing.get("input_validation_review_only", True)
        and incoming.get("input_validation_review_only", True)
    )}


def input_signature(ticket: Ticket) -> str:
    envelope = {key: getattr(ticket, key, None) for key in (
        "agent_id", "input_format", "payload", "inputs", "customization",
    )}
    return hashlib.sha256(json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


async def report_input_failure(session, ticket, run, message):
    """Stop automatic dispatch and notify once per rejected input version.

    The ticket lock also covers creating the notice and enqueueing its review;
    queue_wakeup commits both, so a crash cannot leave only half the handoff.
    """
    await session.execute(select(Ticket.id).where(
        Ticket.id == ticket.id,
    ).with_for_update())
    await session.refresh(ticket)
    await session.refresh(run)
    reusable_supervisor = ticket.agent_id == "orchestrator" and ticket.id == run.supervisor_ticket_id
    if (ticket.status == "cancelled"
            or (ticket.status in {"succeeded", "degraded"} and not reusable_supervisor)
            or run.status in TERMINAL_RUN_STATUSES or run.cancel_requested_at is not None):
        await session.commit()
        return
    signature = input_signature(ticket)
    notices = (await session.scalars(select(TicketNotice).where(
        TicketNotice.ticket_id == ticket.id, TicketNotice.code == NOTICE_CODE,
    ))).all()
    notice = next((n for n in notices if json.loads(n.body).get("input_signature") == signature), None)
    if notice is None:
        notice = TicketNotice(ticket_id=ticket.id, code=NOTICE_CODE, severity="error",
                              body=json.dumps({"input_signature": signature, "message": message}))
        session.add(notice)
        await session.flush()
    ticket.status = "failed"
    can_review = (ticket.agent_id != "orchestrator" and ticket.lane != "held_out_test"
                  and bool(run.supervisor_ticket_id))
    ticket.repair_route = "orchestrator" if can_review else "terminal"
    ticket.error_message = message[:2000]
    ticket.summary = "Input validation failed before execution; work order requires correction"
    if can_review:
        event = {"ticket_id": ticket.id, "input_signature": signature, "notice_id": notice.id}
        prior = (await session.scalars(select(AgentWakeupRequest).where(
            AgentWakeupRequest.ticket_id == run.supervisor_ticket_id,
            AgentWakeupRequest.status.in_(("queued", "running", "completed", "coalesced")),
        ))).all()
        if not any(event in (w.payload or {}).get(EVENTS_KEY, []) for w in prior):
            from zevo.engine.run.wakeup import queue_wakeup
            await queue_wakeup(
                session, agent_id="orchestrator", ticket_id=run.supervisor_ticket_id,
                source="handoff", trigger_detail=f"payload_validation_failed:{ticket.id}"[:128],
                reason=f"child ticket {ticket.id} could not start; correct its work order. {message}"[:2000],
                payload={EVENTS_KEY: [event]},
            )
    await session.commit()


async def obsolete_input_review(session, wakeup, target) -> bool:
    """Check only validation-review events; unrelated new events stay runnable."""
    if target is None or target.agent_id != "orchestrator" or wakeup.source != "handoff":
        return False
    events = (wakeup.payload or {}).get(EVENTS_KEY)
    if (wakeup.payload or {}).get("input_validation_review_only") is False:
        return False
    if not events:
        # Old queue rows have no input identity. Only retire them when the
        # referenced child is unambiguously no longer awaiting review.
        prefix = "payload_validation_failed:"
        if not wakeup.trigger_detail.startswith(prefix):
            return False
        child = await session.get(Ticket, wakeup.trigger_detail[len(prefix):])
        return child is None or child.status in {"cancelled", "succeeded", "degraded"}
    completed = (await session.scalars(select(AgentWakeupRequest).where(
        AgentWakeupRequest.ticket_id == target.id,
        AgentWakeupRequest.id != wakeup.id,
        AgentWakeupRequest.status == "completed",
    ))).all()
    for event in events:
        if any(event in (prior.payload or {}).get(EVENTS_KEY, []) for prior in completed):
            continue
        child = await session.get(Ticket, event["ticket_id"])
        if (child is not None and child.run_id == target.run_id
                and child.status == "failed" and child.repair_route == "orchestrator"
                and input_signature(child) == event["input_signature"]):
            return False
    return True
