"""Durable admission queue for stages sharing a per-run Slurm allocation.

An InfraInstance with an empty instance_id is a prepared request, not a GPU
allocation. Ticket completion releases the turn, not workload exit/release.
The Run row lock serializes admission and publication across both lanes.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import select

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES, TERMINAL_TICKET_STATUSES
from zevo.db import InfraInstance, Run, Ticket


async def blocking_stage(session, run, ticket):
    if run.gpu_allocation_mode != "per_run":
        return None
    from zevo.engine.run.gpu_controller import checkpoint_execution

    rows = (await session.execute(
        select(InfraInstance, Ticket).join(Ticket, Ticket.id == InfraInstance.ticket_id)
        .where(InfraInstance.run_id == run.id, InfraInstance.provider == "cluster",
               Ticket.status.notin_(list(TERMINAL_TICKET_STATUSES)))
        .order_by(InfraInstance.created_at, InfraInstance.id)
    )).all()
    claims = [(row, owner) for row, owner in rows if
              (row.meta or {}).get("resource_request") or
              (row.meta or {}).get("controller_directory")]
    if not claims:
        return None
    # A yielded Train retains its turn. Its verified checkpoint children borrow
    # that turn; unrelated stages cannot jump in between validation and resume.
    first_owner = claims[0][1]
    if first_owner.id == ticket.id:
        return None
    checkpoint = await checkpoint_execution(session, run, ticket)
    from zevo.engine.run.lifecycle import aware
    if checkpoint is not None and checkpoint.ticket_id == first_owner.id:
        if any(owner.id == first_owner.id and row.id != checkpoint.id
               and aware(row.created_at) > aware(checkpoint.created_at) and row.released_at is None
               for row, owner in claims):
            return first_owner
        other = next((owner for row, owner in claims
                      if owner.id not in {first_owner.id, ticket.id}
                      and row.instance_id and row.released_at is None), None)
        return other
    return first_owner


def waiting_submission(*, blocker, contract, info, heartbeat_id, sha256, continuation):
    def snapshot(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return json.loads(json.dumps(value, default=vars))
    return "", {
        "scheduler_state": "WAITING_RESOURCES",
        "stage_queue": {
            "blocked_by_ticket_id": blocker.id,
            "contract": snapshot(contract), "device_info": snapshot(info),
            "heartbeat_id": heartbeat_id, "sha256": sha256,
            "continuation": continuation or {},
        },
    }


async def dispatch_waiting_stages(session):
    """Publish prepared stages without another Agent activation or repair."""
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo, SlurmStageJobContract
    from zevo.engine.run.gpu_controller import submit_workload
    from zevo.engine.run.steering import instruction_gate_active
    from zevo.engine.run.runner import _commit_slurm_submission, _slurm_stage_implementation_sha256

    ids = (await session.scalars(select(InfraInstance.id).where(
        InfraInstance.provider == "cluster", InfraInstance.instance_id == "",
        InfraInstance.released_at.is_(None),
    ).order_by(InfraInstance.created_at, InfraInstance.id))).all()
    dispatched = 0
    for row_id in ids:
        row = await session.get(InfraInstance, row_id)
        if not (row.meta or {}).get("stage_queue"):
            continue
        # Refresh after taking the lock: another scheduler may have dispatched
        # this exact request while this scheduler was waiting for the lock.
        run = (await session.execute(select(Run).where(Run.id == row.run_id)
                                     .with_for_update().execution_options(populate_existing=True))).scalar_one()
        await session.refresh(row)
        ticket = (await session.execute(select(Ticket).where(Ticket.id == row.ticket_id)
                                        .with_for_update().execution_options(populate_existing=True))).scalar_one()
        if row.instance_id or row.released_at or not (row.meta or {}).get("stage_queue"):
            await session.commit()
            continue
        if run.status in TERMINAL_RUN_STATUSES or ticket.status in TERMINAL_TICKET_STATUSES:
            row.released_at = datetime.now(timezone.utc)
            row.status = "released"
            row.release_reason = "queued stage ended before resource admission"
            await session.commit()
            continue
        started = (row.meta or {}).get("stage_queue_dispatch_started_at")
        if started:
            # A prior process may have published and crashed before recording
            # the JOBID. Do not replay sbatch, even after a scheduler restart.
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
            if elapsed > 120 and ticket.status == "waiting_external":
                ticket.status = "awaiting_input"
                ticket.error_message = "Queued stage submission was interrupted; reconcile the remote request before retrying."
            await session.commit()
            continue
        if (run.cancel_requested_at is not None or ticket.status != "waiting_external" or
            not (row.meta or {}).get("submission_committed") or
            (ticket.lane == "optimization" and await instruction_gate_active(session, run.id)) or
            await blocking_stage(session, run, ticket)):
            await session.commit()
            continue
        pending = row.meta["stage_queue"]
        publication = {"attempted": False}
        try:
            contract = SlurmStageJobContract.model_validate(pending["contract"])
            info = InfrastructureDeviceInfo.model_validate(pending["device_info"])
            if hashlib.sha256(Path(contract.script_path).read_bytes()).hexdigest() != pending["sha256"]:
                raise ValueError("queued workload script changed after validation")
            expected_implementation = (row.meta or {}).get("implementation_sha256")
            if expected_implementation and _slurm_stage_implementation_sha256(contract) != expected_implementation:
                raise ValueError("queued stage implementation changed after validation")
            # Persist intent before the remote side effect. The Run lock is
            # reacquired after commit; other dispatchers see the durable marker
            # and cannot publish this request again.
            row.meta = {**dict(row.meta or {}),
                        "stage_queue_dispatch_started_at": datetime.now(timezone.utc).isoformat()}
            await session.commit()
            run = (await session.execute(select(Run).where(Run.id == row.run_id)
                .with_for_update().execution_options(populate_existing=True))).scalar_one()
            ticket = (await session.execute(select(Ticket).where(Ticket.id == row.ticket_id)
                .with_for_update().execution_options(populate_existing=True))).scalar_one()
            await session.refresh(row)
            if (run.status in TERMINAL_RUN_STATUSES or run.cancel_requested_at is not None
                or ticket.status != "waiting_external"
                or (ticket.lane == "optimization" and await instruction_gate_active(session, run.id))):
                meta = dict(row.meta or {})
                meta.pop("stage_queue_dispatch_started_at", None)
                row.meta = meta
                await session.commit()
                continue
            job_id, controller_meta = await submit_workload(
                session, run=run, ticket=ticket, contract=contract, info=info,
                heartbeat_id=pending["heartbeat_id"], sha256=pending["sha256"],
                publication=publication, continuation=pending["continuation"],
            )
            if not job_id:
                meta = dict(row.meta or {})
                meta.pop("stage_queue_dispatch_started_at", None)
                row.meta = meta
                await session.commit()
                continue
            meta = dict(row.meta or {})
            meta.pop("stage_queue", None)
            meta.pop("stage_queue_dispatch_started_at", None)
            meta.pop("stage_queue_error", None)
            meta.update(controller_meta)
            meta["scheduler_state"] = "PENDING"
            # Deferred inference has the same effective controller envelope as
            # immediate submission, so later inference replay stays accurate.
            if meta.get("inference_launch"):
                from zevo.engine.run.inference_launch import effective_snapshot
                meta["inference_launch"] = effective_snapshot(
                    meta["inference_launch"], controller_meta["controller_script_snapshot"],
                    controller_meta["controller_workload_launch"])
            row.instance_id = job_id
            row.meta = meta
            _commit_slurm_submission(row, heartbeat_id=pending["heartbeat_id"],
                stdout_path=contract.stdout_path, stderr_path=contract.stderr_path)
            await session.commit()
            dispatched += 1
        except Exception as exc:
            await session.rollback()
            row = await session.get(InfraInstance, row_id)
            # Publication can be ambiguous on SSH failure. Never automatically
            # replay a possibly submitted command (same rule as normal submit).
            row.meta = {**dict(row.meta or {}), "stage_queue_error": str(exc)[:2000]}
            if publication["attempted"]:
                row.meta = {**row.meta, "stage_queue_publication_uncertain": True}
            ticket = await session.get(Ticket, row.ticket_id)
            if ticket.status not in TERMINAL_TICKET_STATUSES:
                ticket.status = "awaiting_input"
                ticket.error_message = "Queued stage publication needs reconciliation: " + str(exc)[:1500]
            await session.commit()
    return dispatched


async def preparation_already_queued(session, run, ticket, activation_payload):
    """Old alarms must not rerun the Agent or mutate a validated queued script."""
    rows = (await session.scalars(select(InfraInstance).where(
        InfraInstance.ticket_id == ticket.id, InfraInstance.provider == "cluster",
        InfraInstance.instance_id == "", InfraInstance.released_at.is_(None),
    ))).all()
    pending = [row for row in rows if (row.meta or {}).get("stage_queue")]
    if not pending:
        return False
    if not (activation_payload.get("run_instruction_id") and
            activation_payload.get("instruction_action") == "restart_activation"):
        return True
    await session.execute(select(Run).where(Run.id == run.id).with_for_update())
    for row in pending:
        await session.refresh(row)
        if row.instance_id or (row.meta or {}).get("stage_queue_dispatch_started_at"):
            return True  # Reconcile an admitted/ambiguous remote request first.
        row.released_at = datetime.now(timezone.utc)
        row.status = "released"
        row.release_reason = "queued preparation superseded by instruction"
    await session.commit()
    return False
