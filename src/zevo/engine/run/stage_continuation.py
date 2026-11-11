"""Normal execution-window expiry, independent of specialist and provider.

This authorizes another execution, not successful completion or reuse of an
artifact. Workers still verify their durable batch/benchmark receipts; Train
must additionally verify its full optimizer checkpoint before submission.
"""
from __future__ import annotations


def expired(ticket, row):
    if row is None or row.ticket_id != ticket.id or row.run_id != ticket.run_id:
        return False
    meta = row.meta or {}
    if not row.instance_id or not (meta.get("resource_request") or meta.get("stage_job")):
        return False
    if meta.get("superseded_by_instruction_id"):
        return False
    state = str(meta.get("scheduler_state") or "").upper().split("+")[0].split(" ")[0]
    code = str(meta.get("scheduler_exit_code") or "").split(":")[0]
    return state in {"TIMEOUT", "DEADLINE"} or (state == "FAILED" and code == "75")


def authorized(ticket, row):
    from zevo.engine.run.training_continuation import authorized as train_authorized
    if train_authorized(ticket, row):
        return True
    if row is None or not (row.meta or {}).get("stage_continuation"):
        return False
    if ticket.agent_id == "train" or not expired(ticket, row):
        return False
    proof = (row.meta or {}).get("stage_continuation") or {}
    return (proof.get("source_resource_id") == row.id
            and proof.get("ticket_id") == ticket.id)


async def queue_continuation(session, ticket, row):
    """Retire a finite allocation once, or reuse a direct host's healthy lease."""
    if ticket.agent_id == "train" or not expired(ticket, row):
        raise ValueError("Stage continuation requires the current expired execution")
    if authorized(ticket, row):
        return  # Duplicate watcher deliveries cannot enqueue another activation.
    row.meta = {**dict(row.meta or {}), "stage_continuation": {
        "source_resource_id": row.id, "source_job_id": row.instance_id,
        "ticket_id": ticket.id,
    }}
    if row.provider == "cluster":
        from zevo.engine.run.resource_recovery import request_replacement
        await request_replacement(session, ticket, row, kind="continuation",
                                  reason="Continue the same Ticket after its execution window expired")
    else:
        await wake_continuation(session, ticket, row)


async def wake_continuation(session, ticket, row):
    if not authorized(ticket, row):
        raise ValueError("Continuation no longer belongs to this execution")
    if ticket.agent_id == "train":
        from zevo.engine.run.training_continuation import wake_continuation as wake_train
        await wake_train(session, ticket, row, row.meta["training_continuation"])
        return
    from zevo.engine.run.wakeup import queue_wakeup
    ticket.status = "queued"
    ticket.repair_route = ""
    ticket.error_message = ""
    ticket.summary = "Continuing after execution window expiry; restoring verified progress"
    await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id,
        source="stage_continuation",
        reason="Continue the same workload and configuration. Reuse only verified durable progress; repeat unfinished work. Do not publish partial results as complete.",
        payload={"continuation_resource_id": row.id})
