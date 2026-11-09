"""Confirm lost cloud allocations and hand resource recovery to Infrastructure.

Provider API errors and SSH failures are never proof of loss. Missing inventory
requires two observations in the same credential/region scope that previously
observed this exact allocation. All recovery evidence survives controller restarts.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path

from sqlalchemy import select

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES, TERMINAL_TICKET_STATUSES
from zevo.db import InfraInstance, Run, Ticket, TicketNotice
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS

log = logging.getLogger(__name__)


def _aware(value):
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


async def binding_lost(session, run_id, device_info_path):
    """Fence late responses using this activation's allocation, not its Ticket's history."""
    if not device_info_path:
        return False
    try:
        info = json.loads(Path(device_info_path).read_text())
    except (OSError, ValueError):
        return False
    if info.get("provider") != "cloud" or not info.get("instance_id"):
        return False
    rows = (await session.scalars(select(InfraInstance).where(InfraInstance.run_id == run_id,
        InfraInstance.provider == "cloud", InfraInstance.instance_id == info["instance_id"],
        InfraInstance.released_at.is_not(None)).execution_options(populate_existing=True))).all()
    return any((row.meta or {}).get("cloud_health", {}).get("state")
               in {"terminated", "deleted", "destroyed", "not-found"} for row in rows)


async def inherited_repairs(session, run, agent_id, iteration, payload):
    """Replacing a host/Ticket does not reset the same work order's repair cap."""
    if not (run.lifecycle or {}).get("cloud_resource_recoveries"):
        return 0
    previous = (await session.scalars(select(Ticket).join(TicketNotice, TicketNotice.ticket_id == Ticket.id).where(
        Ticket.run_id == run.id, Ticket.agent_id == agent_id, Ticket.iteration == iteration,
        Ticket.lane == "optimization", TicketNotice.code.in_(["resource.allocation_lost", "resource.repair_inherited"])))).all()
    identity = ("operation", "purpose", "base_model", "model_source", "training_method", "training_method_pin")
    return max((int(ticket.repair_attempts or 0) for ticket in previous if all(
        (ticket.payload or {}).get(key, "") == payload.get(key, "") for key in identity)), default=0)


async def recovery_context(session, run):
    """Allocation facts only: never forward private scoring diagnostics."""
    events = list((run.lifecycle or {}).get("cloud_resource_recoveries", {}).values())
    if not events:
        return []
    rows = (await session.scalars(select(InfraInstance).where(InfraInstance.run_id == run.id,
        InfraInstance.provider == "cloud", InfraInstance.released_at.is_(None)))).all()
    result = []
    for event in events:
        replacements = [row for row in rows if not (row.meta or {}).get("allocation_owner_row_id")
            and row.id != event["allocation_id"] and row.status == "ready"
            and _aware(row.created_at) > datetime.fromisoformat(event["observed_at"])]
        if replacements:
            replacement = min(replacements, key=lambda row: _aware(row.created_at))
            event = {**event, "replacement_allocation_id": replacement.id,
                "next_action": "Replacement allocation is ready. Reuse its registered device_info; do not provision again. "
                    "Restore durable inputs and verified checkpoints, or rebuild missing remote-only Data before retrying work."}
        result.append(event)
    return result


async def observe(session, run, owner, health):
    from zevo.engine.run.resource_cleanup import _cloud_provider, cloud_recovery_keys
    keys = await cloud_recovery_keys(session, run, owner)
    backend = (owner.meta or {}).get("backend") or (owner.meta or {}).get("cloud_backend") or run.cloud_backend
    if run.cloud_backend and backend != run.cloud_backend:
        raise ValueError("cloud allocation backend differs from the Run")
    key_name = {"aws": "AWS_ACCESS_KEY_ID", "vastai": "VASTAI_API_KEY", "lambda": "LAMBDA_API_KEY"}.get(backend)
    key = keys.get(key_name, "") if key_name else ""
    secret = keys.get("AWS_SECRET_ACCESS_KEY", "") if backend == "aws" else ""
    region = str((owner.meta or {}).get("region") or health.get("region") or keys.get("AWS_REGION", "")) if backend == "aws" else ""
    if backend == "aws" and not region:
        raise ValueError("AWS allocation region is unknown; cannot infer loss from another region")
    if not key:
        raise ValueError("cloud observation credentials are unavailable")
    token = keys.get("AWS_SESSION_TOKEN", "") if backend == "aws" else ""
    # Persist a digest, never credentials. Rotation/account changes invalidate
    # negative-inventory evidence until this resource is seen in the new scope.
    identity = hashlib.sha256(json.dumps([backend, region, key, secret, token]).encode()).hexdigest()
    provider = _cloud_provider(backend, key, secret, **({"region": region} if region else {}),
                               **({"session_token": token} if token else {}))
    if backend == "aws":
        item = await provider.get_instance(owner.instance_id)
        if str(item.get("id", "")) != owner.instance_id or item.get("region") != region:
            raise ValueError("cloud observation returned another allocation or region")
        state = str(item.get("status") or "unknown").lower()
    else:
        items = await provider.list_instances()
        if not isinstance(items, list) or any(not isinstance(item, dict) or "id" not in item for item in items):
            raise ValueError("cloud inventory response is incomplete")
        matching = [item for item in items if str(item["id"]) == owner.instance_id]
        if len(matching) > 1:
            raise ValueError("cloud inventory has conflicting allocation identities")
        state = str(matching[0].get("status") or matching[0].get("actual_status") or "unknown").lower() if matching else "not-found"
    if health.get("scope") != identity:
        health.pop("seen", None)
        health["missing_count"] = 0
    health.update(scope=identity, backend=backend, region=region, state=state)
    if state == "not-found":
        health["missing_count"] = int(health.get("missing_count", 0)) + 1
        return bool(health.get("seen") and health["missing_count"] >= 2)
    if state != "unknown":
        health.update(seen=True, missing_count=0)
    return state in {"terminated", "deleted", "destroyed"}


async def handoff(session, run, owner, health):
    from zevo.engine.cost.budget import snapshot_for_run
    events = dict((run.lifecycle or {}).get("cloud_resource_recoveries", {}))
    if owner.id in events:
        return False
    now = datetime.now(timezone.utc)
    rows = (await session.scalars(select(InfraInstance).where(InfraInstance.run_id == run.id))).all()
    affected = [row for row in rows if (row.meta or {}).get("allocation_owner_row_id") == owner.id]
    # Include preparation and collection activations, which may have no live
    # execution row. Otherwise their late errors enqueue a second review of
    # the same lost resource and can overwrite the committed handoff.
    candidates = (await session.scalars(select(Ticket).where(Ticket.run_id == run.id,
        Ticket.agent_id.in_(["data", "train", "inference", "evaluation"]),
        Ticket.status.notin_(list(TERMINAL_TICKET_STATUSES))).with_for_update())).all()
    affected_ids = set()
    from zevo.engine.run.remote_jobs import _device_info_for_ticket
    for ticket in candidates:
        info = await _device_info_for_ticket(session, ticket)
        if info and info.provider == "cloud" and info.instance_id == owner.instance_id:
            affected_ids.add(ticket.id)
        elif info is None:
            executions = [row for row in rows if row.ticket_id == ticket.id and (row.meta or {}).get("direct_workload")]
            latest = max(executions, key=lambda row: row.created_at, default=None)
            if latest is not None and (latest.meta or {}).get("allocation_owner_row_id") == owner.id:
                affected_ids.add(ticket.id)
    owner.released_at, owner.status = now, "released"
    owner.release_reason = "cloud provider confirmed allocation lost"
    for row in affected:
        if row.released_at is not None and row.ticket_id not in affected_ids:
            continue  # completed history stays immutable
        row.released_at, row.status = row.released_at or now, "failed"
        row.release_reason = owner.release_reason
        row.meta = {**(row.meta or {}), "scheduler_state": "NODE_FAIL", "scheduler_reason": owner.release_reason,
                    "allocation_lost": owner.id, "monitor_terminal": True, "collect_wakeup_queued": True}
    await session.flush()
    budget = await snapshot_for_run(session, run.id)
    attempts = int((run.lifecycle or {}).get("cloud_resource_recovery_attempts", 0))
    blocked = ("Run budget exhausted" if budget.over_budget or budget.over_time_limit else
               "Run is finalizing" if (run.lifecycle or {}).get("finalization") else "")
    active = [ticket for ticket in candidates if ticket.id in affected_ids]
    eligible = {ticket.id for ticket in active if int(ticket.repair_attempts or 0) < MAX_REPAIR_ATTEMPTS}
    if active and not eligible:
        blocked = blocked or "Ticket repair limit reached"
    allowed = not blocked
    if allowed:
        attempts += 1
    action = (
        "The engine will provision replacement compute through Infrastructure with the unchanged resource plan "
        "and bind the original Tickets to its device_info. Reuse durable artifacts only; verify any full-state "
        "checkpoint on storage accessible from the new host before resuming. Remote-only paths on the lost "
        "instance are invalid: rebuild missing Data or report that recovery is impossible. Preserve repair "
        "counts on the same work orders; do not restart completed expensive work when durable outputs exist."
        if allowed else "Do not provision a replacement: " + blocked
    )
    event = {"allocation_id": owner.id, "instance_id": owner.instance_id,
        "backend": health["backend"], "region": health.get("region", ""), "provider_state": health["state"],
        "observed_at": now.isoformat(), "recovery_attempt": attempts,
        "replacement_allowed": allowed, "blocked_reason": blocked, "next_action": action}
    events[owner.id] = event
    run.lifecycle = {**(run.lifecycle or {}), "cloud_resource_recoveries": events,
                     "cloud_resource_recovery_attempts": attempts}
    for ticket in active:
        ticket_allowed = allowed and ticket.id in eligible
        if ticket_allowed:
            ticket.repair_attempts = int(ticket.repair_attempts or 0) + 1
        ticket.status = "failed"
        ticket.repair_route = "terminal"
        ticket.error_message = owner.release_reason
        ticket.summary = "Cloud allocation lost; " + ("resource recovery requested" if ticket_allowed else "recovery stopped")
        if not ticket_allowed:
            ticket.repair_route = "terminal"
        session.add(TicketNotice(ticket_id=ticket.id, code="resource.allocation_lost", severity="error",
                                body=json.dumps(event)))
        if ticket_allowed:
            from zevo.engine.run.resource_recovery import request_replacement
            from zevo.engine.run.runner import _latest_slurm_resource_request
            await request_replacement(session, ticket, await _latest_slurm_resource_request(session, ticket),
                                      owner=owner, reason=owner.release_reason)
    # Recovery stays within the engine, including private stages. No private
    # ticket/diagnostic is handed to the optimization supervisor.
    await session.commit()
    return True


async def reconcile(session):
    owners = (await session.scalars(select(InfraInstance).join(Run, Run.id == InfraInstance.run_id).where(
        InfraInstance.provider == "cloud", InfraInstance.released_at.is_(None),
        Run.status.notin_(list(TERMINAL_RUN_STATUSES)), Run.cancel_requested_at.is_(None)))).all()
    recovered = 0
    owner_ids = [row.id for row in owners]
    for owner_id in owner_ids:
        candidate = await session.get(InfraInstance, owner_id)
        if (candidate.meta or {}).get("allocation_owner_row_id") or (candidate.meta or {}).get("direct_workload"):
            continue
        if candidate.status != "ready" and candidate.ready_at is None:
            continue  # provisioning/not-yet-visible instances are not lost
        run = (await session.scalars(select(Run).where(Run.id == candidate.run_id).with_for_update(skip_locked=True)
                                     .execution_options(populate_existing=True))).one_or_none()
        if run is None:
            continue
        await session.refresh(candidate)
        if run.status in TERMINAL_RUN_STATUSES or run.cancel_requested_at or candidate.released_at:
            await session.commit()
            continue
        health = dict((candidate.meta or {}).get("cloud_health", {}))
        now = datetime.now(timezone.utc)
        if health.get("next_at") and datetime.fromisoformat(health["next_at"]) > now:
            await session.commit()
            continue
        try:
            lost = await asyncio.wait_for(observe(session, run, candidate, health), timeout=40)
            health.update(error="", failures=0)
            health["next_at"] = (now + timedelta(seconds=30 if health["state"] == "not-found" else 60)).isoformat()
            candidate.meta = {**(candidate.meta or {}), "cloud_health": health}
        except Exception as exc:
            lost = False
            failures = int(health.get("failures", 0)) + 1
            health.update(failures=failures, missing_count=0,
                          error=f"Cloud status could not be verified ({type(exc).__name__})")
            health["next_at"] = (now + timedelta(seconds=min(300, 15 * 2 ** min(failures, 5)))).isoformat()
        health["checked_at"] = now.isoformat()
        candidate.meta = {**(candidate.meta or {}), "cloud_health": health}
        try:
            if lost:
                recovered += await handoff(session, run, candidate, health)
            await session.commit()
        except Exception:
            # Never commit a released allocation without its durable recovery
            # notification. A provider-confirmed loss is retried next pass.
            await session.rollback()
            log.exception("Could not hand off confirmed cloud allocation loss: %s", owner_id)
    return recovered
