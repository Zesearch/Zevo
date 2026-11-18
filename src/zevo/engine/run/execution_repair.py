"""Coordinator-reviewed repair of an existing work order, without budget resets."""
from __future__ import annotations

import json
from typing import Literal

from fastapi import HTTPException
from pydantic import Field
from sqlalchemy import select

from zevo.contracts._base import StrictBody
from zevo.contracts.failure import RecoveryText, TrainingConfigurationChange
from zevo.db import HeartbeatRun, Run, Ticket, TicketMessage, TicketNotice
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS
from zevo.engine.run.repair_progress import record_execution_repair


class RepairTicketRequest(StrictBody):
    action: Literal["prepare", "provision", "collect", "resubmit", "replace_resource", "resume_inputs", "reconfigure"]
    training_configuration_change: TrainingConfigurationChange | None = None
    reason: RecoveryText = Field(max_length=2000)
    evidence: list[RecoveryText] = Field(min_length=1, max_length=20)
    execution_attempt: int = Field(ge=0, description="Exact registered attempt; zero only for preparation before any execution")
    train_config_path: str = Field("", description="Candidate local train_config.yaml for Train resubmit; critical changes require reconfigure.")
    input_signature: str = Field("", description="Exact rejected input signature for prepare")
    input_products: dict[str, str] = Field(default_factory=dict, description="For prepare: same upstream Ticket; for resume_inputs: reviewed successful repair dependency. Maps input names to exact WorkProduct IDs.")


async def repair_ticket(session, ticket: Ticket, request: RepairTicketRequest):
    from zevo.engine.cost.budget import snapshot_for_run
    from zevo.engine.run.runner import _latest_slurm_resource_request, _SLURM_TERMINAL_STATES
    from zevo.engine.run.steering import instruction_gate_active
    from zevo.engine.run.wakeup import queue_wakeup

    # Same admission lock as workload publication; recheck all mutable state.
    run = (await session.scalars(select(Run).where(Run.id == ticket.run_id)
                                .with_for_update().execution_options(populate_existing=True))).one()
    await session.refresh(ticket)
    if run.status != "running" or run.cancel_requested_at is not None:
        raise HTTPException(409, "Repair requires an active Run")
    if await instruction_gate_active(session, run.id):
        raise HTTPException(409, "Run instruction must be resolved before repair")
    if (ticket.lane != "optimization" or ticket.agent_id not in {"infrastructure", "data", "train", "inference", "evaluation", "registry"}
            or ticket.status not in {"failed", "awaiting_input"} or ticket.repair_route != "orchestrator"):
        raise HTTPException(409, "Only a failed or waiting optimization workload referred to Orchestrator can be repaired")
    supervisor = await session.get(Ticket, run.supervisor_ticket_id)
    supervisor_heartbeat = await session.scalar(select(HeartbeatRun.id).where(
        HeartbeatRun.ticket_id == run.supervisor_ticket_id,
        HeartbeatRun.agent_id == "orchestrator", HeartbeatRun.finished_at.is_(None),
    ).order_by(HeartbeatRun.started_at.desc()).limit(1))
    if not supervisor or supervisor.status != "running" or not supervisor_heartbeat:
        raise HTTPException(409, "Repair requires a live Run supervisor decision")
    if int(ticket.repair_attempts or 0) >= MAX_REPAIR_ATTEMPTS:
        raise HTTPException(409, "Ticket repair limit reached")
    if request.action in {"resume_inputs", "reconfigure"}:
        from zevo.engine.run.reviewed_recovery import waiting_ancestors
        blocked_chain = await waiting_ancestors(session, ticket)
        active_other = await session.scalar(select(Ticket.id).where(
            Ticket.run_id == run.id, Ticket.lane == "optimization", Ticket.id.notin_(blocked_chain),
            Ticket.agent_id != "orchestrator",
            Ticket.status.in_(("queued", "running", "repairing", "waiting_external", "awaiting_input")),
        ).limit(1))
        if active_other:
            raise HTTPException(409, "Wait for the upstream repair Specialist to finish before resuming")
    if ticket.status == "awaiting_input" and request.action in {"resubmit", "collect"}:
        from zevo.engine.run.reviewed_recovery import latest_request
        review = await latest_request(session, ticket)
        if review.get("request", {}).get("scope") != "fixed_configuration":
            raise HTTPException(409, "Upstream-input waits must resume with explicitly reviewed input_products")
    if await session.scalar(select(HeartbeatRun.id).where(
        HeartbeatRun.ticket_id == ticket.id, HeartbeatRun.finished_at.is_(None),
    ).limit(1)):
        raise HTTPException(409, "Previous activation is still finishing")
    budget = await snapshot_for_run(session, run.id)
    if budget.over_budget or budget.over_time_limit:
        raise HTTPException(409, "Run budget prevents repair")
    row = await _latest_slurm_resource_request(session, ticket)
    meta = dict(row.meta or {}) if row else {}
    state = str(meta.get("scheduler_state") or "").split(" ")[0].upper().rstrip("+")
    if request.training_configuration_change is not None and request.action != "reconfigure":
        raise HTTPException(422, "Configuration approval requires reconfigure")
    if ticket.status == "awaiting_input" and request.action not in {"resume_inputs", "reconfigure", "resubmit", "collect"}:
        raise HTTPException(409, "Waiting review requires resume_inputs or reconfigure")
    if ticket.agent_id == "infrastructure" and request.action not in {"provision", "resume_inputs"}:
        raise HTTPException(422, "Infrastructure recovery requires provision, preserving the existing plan")
    if request.action == "resume_inputs":
        expected_attempt = int(meta.get("execution_attempt") or 1) if row else 0
        if request.execution_attempt != expected_attempt or row and state not in _SLURM_TERMINAL_STATES:
            raise HTTPException(409, "Resume inputs must name the latest terminal execution or zero before execution")
        from zevo.engine.run.reviewed_recovery import restore_inputs
        await restore_inputs(session, run, ticket, request, row)
    elif request.action == "provision":
        failure = await session.scalar(select(TicketNotice).where(
            TicketNotice.ticket_id == ticket.id, TicketNotice.code == "cloud.acquisition_failed",
        ).order_by(TicketNotice.created_at.desc()).limit(1))
        try:
            rejected = json.loads(failure.body) if failure else {}
        except (ValueError, TypeError):
            rejected = {}
        if rejected.get("confirmed_not_created") is not True:
            raise HTTPException(409, "Provision repair requires confirmed absence; reconcile unknown creation first")
        from zevo.db import InfraInstance
        active = await session.scalar(select(InfraInstance.id).where(
            InfraInstance.run_id == run.id, InfraInstance.ticket_id == ticket.id,
            InfraInstance.released_at.is_(None),
        ).limit(1))
        if (ticket.agent_id != "infrastructure" or request.execution_attempt != 0
                or row is not None or active is not None
                or request.input_products or request.input_signature):
            raise HTTPException(409, "Provision repair requires Infrastructure with no active allocation or execution")
    elif request.action == "prepare":
        from zevo.engine.run.input_validation import input_signature, NOTICE_CODE
        from zevo.contracts.tickets import validate_stored_payload
        from zevo.db import WorkProduct
        notices = (await session.scalars(select(TicketNotice).where(
            TicketNotice.ticket_id == ticket.id, TicketNotice.code == NOTICE_CODE,
        ))).all()
        signature = input_signature(ticket)
        if (row is not None or request.execution_attempt != 0 or request.input_signature != signature
            or not any(json.loads(n.body).get("input_signature") == signature for n in notices)):
            raise HTTPException(409, "Preparation repair requires the exact rejected input version and no execution")
        bindings = dict(ticket.inputs or {})
        for name, product_id in request.input_products.items():
            old = bindings.get(name, {})
            product = await session.get(WorkProduct, product_id)
            source = await session.get(Ticket, product.ticket_id) if product else None
            if (not source or source.run_id != ticket.run_id or source.lane != ticket.lane
                or source.status != "succeeded" or product.ticket_id != old.get("source_ticket_id")
                or product.role != old.get("artifact_role")):
                raise HTTPException(409, "Preparation repair must preserve upstream lineage and artifact roles")
            bindings[name] = {**old, "work_product_id": product.id, "path": product.path}
        try:
            validate_stored_payload(agent_id=ticket.agent_id, input_format=ticket.input_format, payload=ticket.payload)
            from zevo.engine.run.scheduler.bindings import resolve_input_bindings
            bindings = await resolve_input_bindings(session, run_id=run.id, inputs=bindings)
        except ValueError as exc:
            raise HTTPException(422, f"Preparation inputs still invalid: {exc}") from exc
        ticket.inputs = bindings
    elif request.input_products or request.input_signature:
        raise HTTPException(422, "Input correction is only available before execution")
    elif (row is None or state not in _SLURM_TERMINAL_STATES
            or int(meta.get("execution_attempt") or 1) != request.execution_attempt):
        raise HTTPException(409, "Repair must name the latest terminal execution attempt")

    if request.action == "reconfigure":
        from zevo.engine.run.reviewed_recovery import approve_configuration
        await approve_configuration(session, ticket, row, request, supervisor_heartbeat)

    if ticket.agent_id == "train" and request.action in {"resubmit", "replace_resource"}:
        from zevo.db import HeartbeatResult
        from zevo.engine.run.repair_progress import verify_repair_training_config
        candidate = request.train_config_path or meta.get("training_config_local_path")
        if not candidate:
            result = await session.scalar(select(HeartbeatResult).where(
                HeartbeatResult.ticket_id == ticket.id
            ).order_by(HeartbeatResult.created_at.desc()).limit(1))
            candidate = (result.output or {}).get("train_config_path") if result else None
        if not candidate:
            raise HTTPException(409, "Train resubmit requires train_config_path for configuration review")
        try:
            verify_repair_training_config(row, candidate)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    # Input repairs and configuration changes are explicit reviewed actions.
    # Reviewed reconfiguration starts afresh; other repairs preserve resumable state.
    if row is not None:
        record_execution_repair(row, action="resubmit" if request.action in {"reconfigure", "resume_inputs"} else request.action, reason=request.reason,
                                heartbeat_id=supervisor_heartbeat)
    decision = await session.get(HeartbeatRun, supervisor_heartbeat)
    targets = list((decision.resolved_config or {}).get("repair_target_ticket_ids", []))
    decision.resolved_config = {**(decision.resolved_config or {}), "repair_target_ticket_ids": sorted(set(targets + [ticket.id]))}
    ticket.repair_attempts = int(ticket.repair_attempts or 0) + 1
    ticket.repair_route = "self"
    ticket.status = "repairing"
    ticket.summary = f"Repair {ticket.repair_attempts}/{MAX_REPAIR_ATTEMPTS}: {request.reason}"
    session.add(TicketNotice(ticket_id=ticket.id, code="repair.authorized", severity="info",
        body=json.dumps({**request.model_dump(), "heartbeat_id": supervisor_heartbeat,
                         "resource_id": row.id if row else "", "repair_attempt": ticket.repair_attempts,
                         "training_config_review": (row.meta or {}).get("training_config_retry_audit") if row else None})))
    guidance = (
        "Use only the approved proposed_config_path and SHA-256. Start a fresh execution from the original parent model, "
        "with a fresh attempt-local plan and optimizer state; never resume a prior attempt checkpoint. "
        "Preserve data, model, method, inference protocol and resource requirements."
        if request.action == "reconfigure" else
        "Re-read the repaired input bindings and validate them before execution. Preserve all other work-order requirements."
        if request.action == "resume_inputs" else
        "Preserve training semantics, inputs, plan, resources and verified resumable checkpoints. "
        "Only engine-recognized execution-only YAML changes are permitted; checkpoint continuation remains byte-identical."
    )
    session.add(TicketMessage(ticket_id=ticket.id, author="system", body=(
        f"__REPAIR__\n\nOrchestrator approved {request.action} on this same Ticket "
        f"(repair {ticket.repair_attempts}/{MAX_REPAIR_ATTEMPTS}). {request.reason}\n"
        f"Evidence: {json.dumps(request.evidence)}\n"
        f"Follow the engine-supplied execution contract. {guidance}"
    )))
    session.add(TicketNotice(ticket_id=ticket.id,
        code="ticket.awaiting_supervisor_completion", severity="info",
        body=json.dumps({"heartbeat_id": supervisor_heartbeat, "repair": True})))
    await session.commit()
    return {"ticket_id": ticket.id, "status": ticket.status,
            "repair_attempts": ticket.repair_attempts, "action": request.action}


async def release_reviewed_repairs(session, supervisor, heartbeat_id):
    """Release only repairs belonging to this committed, validated decision."""
    from zevo.engine.run.wakeup import queue_wakeup
    from zevo.engine.run.resource_recovery import request_replacement
    from zevo.db import InfraInstance
    run = await session.scalar(select(Run).where(Run.id == supervisor.run_id)
        .with_for_update().execution_options(populate_existing=True))
    await session.refresh(supervisor)
    decision = await session.get(HeartbeatRun, heartbeat_id, populate_existing=True)
    if (run is None or run.status != "running" or run.cancel_requested_at is not None
            or decision is None or decision.finished_at is None
            or decision.ticket_id != supervisor.id or run.supervisor_ticket_id != supervisor.id
            or supervisor.status not in {"succeeded", "degraded"}):
        return
    from zevo.db import HeartbeatResult
    from zevo.engine.run.steering import instruction_gate_active
    from zevo.engine.cost.budget import snapshot_for_run
    result = await session.scalar(select(HeartbeatResult).where(
        HeartbeatResult.heartbeat_id == heartbeat_id,
        HeartbeatResult.ticket_id == supervisor.id,
        HeartbeatResult.agent_id == "orchestrator",
        HeartbeatResult.status.in_(["succeeded", "degraded"])))
    if result is None or await instruction_gate_active(session, run.id):
        return
    budget = await snapshot_for_run(session, run.id)
    if budget.over_budget or budget.over_time_limit:
        return
    # Notices are durable even when the runner's cached heartbeat predates the
    # approval API transaction, or a later config write overwrote its target list.
    candidates = (await session.scalars(select(TicketNotice).join(
        Ticket, Ticket.id == TicketNotice.ticket_id).where(
        Ticket.run_id == run.id, Ticket.status == "repairing",
        TicketNotice.code == "ticket.awaiting_supervisor_completion"))).all()
    targets = set((decision.resolved_config or {}).get("repair_target_ticket_ids", []))
    for notice in candidates:
        try:
            data = json.loads(notice.body)
        except (ValueError, TypeError):
            continue
        if data.get("heartbeat_id") == heartbeat_id and data.get("repair") is True:
            targets.add(notice.ticket_id)
    for ticket_id in sorted(targets):
        # queue_wakeup commits, so acquire the admission lock for each target.
        run = await session.scalar(select(Run).where(Run.id == run.id)
            .with_for_update().execution_options(populate_existing=True))
        if run.status != "running" or run.cancel_requested_at is not None:
            return
        ticket = await session.get(Ticket, ticket_id, populate_existing=True)
        if ticket is None or ticket.run_id != run.id or ticket.status != "repairing":
            continue
        notices = (await session.scalars(select(TicketNotice).where(
            TicketNotice.ticket_id == ticket_id,
            TicketNotice.code.in_(["repair.authorized", "ticket.awaiting_supervisor_completion"]),
        ))).all()
        authorization = None
        gates = []
        for notice in notices:
            try:
                data = json.loads(notice.body)
            except (ValueError, TypeError):
                continue
            if data.get("heartbeat_id") != heartbeat_id:
                continue
            if notice.code == "repair.authorized":
                authorization = data
            else:
                gates.append(notice)
        if (not authorization or not gates
                or authorization.get("repair_attempt") != ticket.repair_attempts
                or ticket.lane != "optimization"
                or await session.scalar(select(HeartbeatRun.id).where(
                    HeartbeatRun.ticket_id == ticket.id, HeartbeatRun.finished_at.is_(None)).limit(1))):
            continue
        if authorization.get("resource_id"):
            row = await session.get(InfraInstance, authorization["resource_id"], populate_existing=True)
            meta = (row.meta or {}) if row else {}
            repair = meta.get("execution_repair") or {}
            if (row is None or row.ticket_id != ticket.id
                    or repair.get("heartbeat_id") != heartbeat_id
                    or int(meta.get("execution_attempt") or 1) != authorization.get("execution_attempt")):
                continue
        if authorization["action"] == "reconfigure":
            row = await session.get(InfraInstance, authorization["resource_id"])
            approval = dict((row.meta or {}).get("training_configuration_approval") or {}) if row else {}
            if approval.get("heartbeat_id") != heartbeat_id:
                continue
            row.meta = {**row.meta, "training_configuration_approval": {**approval, "released": True}}
        for gate in gates:
            gate.code = "ticket.supervisor_handoff_released"
        if authorization["action"] == "replace_resource":
            row = await session.get(InfraInstance, authorization["resource_id"])
            await request_replacement(session, ticket, row)
            await session.commit()
        else:
            await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id,
                source="retry", reason="Validated Orchestrator repair decision completed",
                payload={"repair_attempt": ticket.repair_attempts, "review_heartbeat_id": heartbeat_id})


async def reconcile_reviewed_repairs(session):
    """Recover a committed approval if the consumer died before releasing it."""
    notices = (await session.scalars(select(TicketNotice).join(
        Ticket, Ticket.id == TicketNotice.ticket_id).where(
        Ticket.status == "repairing", Ticket.lane == "optimization",
        TicketNotice.code == "ticket.awaiting_supervisor_completion"))).all()
    decisions = set()
    for notice in notices:
        try:
            data = json.loads(notice.body)
        except (ValueError, TypeError):
            continue
        if data.get("repair") is True and data.get("heartbeat_id"):
            decisions.add(data["heartbeat_id"])
    for heartbeat_id in sorted(decisions):
        hb = await session.get(HeartbeatRun, heartbeat_id, populate_existing=True)
        if hb is None or hb.finished_at is None:
            continue
        supervisor = await session.get(Ticket, hb.ticket_id, populate_existing=True)
        if supervisor is not None:
            await release_reviewed_repairs(session, supervisor, heartbeat_id)
    await session.commit()
