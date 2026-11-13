"""Bounded, durable waits for an explicitly rejected cloud acquisition."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from zevo.db import HeartbeatRun, InfraInstance, Run, TicketMessage, TicketNotice

KEY = "cloud_capacity_waits"
MAX_ATTEMPTS = 6


def next_attempt(run, ticket_id):
    value = (run.lifecycle or {}).get(KEY, {}).get(ticket_id, {}).get("next_at")
    return datetime.fromisoformat(value) if value else None


async def handle_failure(session, ticket, inp, heartbeat_id):
    """None: ordinary failure; True: wait; False: bounded wait exhausted."""
    if ticket.agent_id != "infrastructure" or inp is None or not heartbeat_id:
        return None
    path = Path(getattr(inp, "work_dir", "")) / "acquire_state.json"
    try:
        state = json.loads(path.read_text())
        attempted = datetime.fromisoformat(state["attempted_at"])
        if attempted.tzinfo is None:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    hb = await session.get(HeartbeatRun, heartbeat_id)
    if (hb is None or attempted < hb.started_at.replace(tzinfo=timezone.utc)
            or state.get("ticket_id") != ticket.id or state.get("run_id") != ticket.run_id
            or state.get("phase") != "create_failed"):
        return None
    session.add(TicketNotice(ticket_id=ticket.id, code="cloud.acquisition_failed",
        severity="warning", body=json.dumps(state)))
    if state.get("failure_kind") != "capacity" or state.get("confirmed_not_created") is not True:
        # Unknown creation must be reconciled, never re-run by a generic repair.
        ticket.status = "failed"
        ticket.repair_route = "orchestrator"
        return False
    active = await session.scalar(select(InfraInstance.id).where(
        InfraInstance.ticket_id == ticket.id, InfraInstance.released_at.is_(None),
    ).limit(1))
    if active:
        ticket.status = "failed"
        ticket.repair_route = "orchestrator"
        return False
    run = (await session.scalars(select(Run).where(Run.id == ticket.run_id)
        .with_for_update().execution_options(populate_existing=True))).one()
    from zevo.engine.cost.budget import snapshot_for_run
    budget = await snapshot_for_run(session, run.id)
    from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS
    now = datetime.now(timezone.utc)
    previous = (run.lifecycle or {}).get(KEY, {}).get(ticket.id, {})
    first = datetime.fromisoformat(previous["first_at"]) if previous else now
    count = int(previous.get("attempts", 0)) + 1
    delay = min(60 * 2 ** min(count - 1, 5), 900)
    deadline = first + timedelta(hours=min(float(run.max_queue_wait_hours or 1), 1))
    due = now + timedelta(seconds=delay)
    if (count >= MAX_ATTEMPTS or due > deadline or budget.over_budget or budget.over_time_limit
            or run.status != "running" or run.cancel_requested_at is not None
            or int(ticket.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS):
        ticket.status = "failed"
        ticket.repair_route = "orchestrator"
        ticket.summary = "Cloud capacity retry limit reached; supervisor review required"
        return False
    waits = dict((run.lifecycle or {}).get(KEY, {}))
    waits[ticket.id] = {"attempts": count, "first_at": first.isoformat(),
        "next_at": due.isoformat(), "heartbeat_id": heartbeat_id}
    run.lifecycle = {**(run.lifecycle or {}), KEY: waits}
    ticket.status = "repairing"
    ticket.repair_route = "self"
    ticket.repair_attempts = int(ticket.repair_attempts or 0) + 1
    ticket.summary = f"Waiting for cloud capacity; retry {count}/{MAX_ATTEMPTS} at {due.isoformat()}"
    session.add(TicketMessage(ticket_id=ticket.id, author="system", body=(
        f"Cloud provider explicitly rejected creation without allocating an instance. {ticket.summary}. "
        "Retry the same acquisition helper on this Ticket. Preserve the approved resource plan, "
        "provider, region, market, offer, disk and budget. Do not create a replacement Ticket, "
        "switch to on-demand, or bypass the acquisition journal. Unknown creation requires reconciliation."
    )))
    return True
