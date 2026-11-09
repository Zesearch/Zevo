"""Durable, same-work-order resource replacement, shared by both execution lanes.

The Run lock serializes decisions. Provider release must be confirmed before a
replacement is provisioned. Infrastructure sees resource facts only; private
stage diagnostics and results never become coordinator inputs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import logging
from pathlib import Path
import shlex
from types import SimpleNamespace

from sqlalchemy import func, select

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db import Run, Ticket, InfraInstance, HeartbeatRun, WorkProduct, TicketMessage
from zevo.engine.run.failure_policy import MAX_REPAIR_ATTEMPTS
from zevo.engine.run.repair_progress import record_execution_repair

KEY = "stage_resource_recoveries"
log = logging.getLogger(__name__)


def save(run, ticket_id, event):
    run.lifecycle = {**(run.lifecycle or {}), KEY: {**(run.lifecycle or {}).get(KEY, {}), ticket_id: event}}


async def is_recovery_infrastructure(session, ticket):
    if ticket.agent_id != "infrastructure":
        return False
    run = await session.get(Run, ticket.run_id)
    return any(e.get("infra_ticket_id") == ticket.id for e in (run.lifecycle or {}).get(KEY, {}).values())


async def current_device_binding(session, run, inputs):
    """New downstream work follows a recovered route without rewriting history."""
    from zevo.engine.run.remote_jobs import _device_info_for_ticket
    replacements = (run.lifecycle or {}).get("device_replacements", {})
    if not replacements or not inputs.get("device_info"):
        return inputs
    info = await _device_info_for_ticket(session, SimpleNamespace(inputs=inputs))
    source = info.ticket_id if info else ""
    visited = set()
    result = dict(inputs)
    while source in replacements:
        if source in visited:
            raise ValueError("Resource replacement bindings contain a cycle")
        visited.add(source)
        product = await session.get(WorkProduct, replacements[source])
        owner = await session.get(Ticket, product.ticket_id) if product else None
        if not owner or owner.run_id != run.id or product.role != "device_info" or owner.status != "succeeded":
            raise ValueError("Replacement device product is not verified in this Run")
        result["device_info"] = {"source_ticket_id": owner.id, "work_product_id": product.id,
                                "artifact_role": "device_info", "path": product.path}
        source = owner.id
    return result


async def request_replacement(session, ticket, execution=None, *, owner=None, reason=""):
    run = (await session.scalars(select(Run).where(Run.id == ticket.run_id).with_for_update()
        .execution_options(populate_existing=True))).one()
    meta = (execution.meta or {}) if execution else {}
    identity = execution.id if execution else (owner.id if owner else "preparation")
    prior = (run.lifecycle or {}).get(KEY, {}).get(ticket.id, {})
    if prior.get("execution_id") == identity:
        return  # one replacement decision per failed execution, including after restart
    if owner is None and execution:
        owner = await session.get(InfraInstance, meta.get("allocation_owner_row_id") or execution.id)
    from zevo.engine.run.remote_jobs import _device_info_for_ticket
    info = await _device_info_for_ticket(session, ticket)
    save(run, ticket.id, {
        "execution_id": identity, "owner_id": owner.id if owner else "",
        "phase": "stopping", "reason": reason or meta.get("execution_repair", {}).get("reason", "Resource replacement requested"),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "resource_plan": info.resource_plan.model_dump(mode="json") if info else {},
    })
    ticket.status = "waiting_external"
    ticket.repair_route = "self"
    ticket.summary = "Waiting for resource recovery on this Ticket"
    # Cloud loss may arrive while a submit/collect Agent is still alive. Its
    # own consumer stops that process; the reconciler waits for it to close.
    if owner and (owner.meta or {}).get("cloud_health", {}).get("state") in {"terminated", "deleted", "destroyed", "not-found"}:
        live = (await session.scalars(select(HeartbeatRun).where(
            HeartbeatRun.ticket_id == ticket.id, HeartbeatRun.finished_at.is_(None),
        ))).all()
        for activation in live:
            activation.resolved_config = {**(activation.resolved_config or {}), "resource_recovery_requested": True}
    await session.flush()


async def _stop_owner(session, run, ticket, owner):
    from zevo.engine.run.remote_jobs import cancel_ticket_remote_job
    from zevo.engine.run.resource_cleanup import release_cloud, release_cluster
    if owner and owner.released_at:
        return True
    if owner:
        rows = (await session.scalars(select(InfraInstance).where(
            InfraInstance.run_id == run.id, InfraInstance.released_at.is_(None),
        ))).all()
        for row in rows:
            if row.ticket_id == ticket.id or row.id == owner.id:
                continue
            if (row.meta or {}).get("allocation_owner_row_id") == owner.id and not (row.meta or {}).get("monitor_terminal"):
                return False  # never terminate another stage's work
    stopped = await cancel_ticket_remote_job(session, ticket, preserve_allocation=True)
    if stopped.get("attempted") and not stopped.get("ok"):
        return False
    if owner:
        if owner.provider == "cloud":
            confirmed = await release_cloud(owner, run, session=session)
        elif owner.provider == "cluster":
            from zevo.engine.run.gpu_controller import _retained_allocation_state, _TERMINAL_ALLOCATION_STATES
            from zevo.engine.run.remote_jobs import _device_info_for_ticket
            info = await _device_info_for_ticket(session, ticket)
            state = await _retained_allocation_state(info, owner.instance_id)
            confirmed = state in _TERMINAL_ALLOCATION_STATES or await release_cluster(session, owner, run)
        else:
            # A fixed host is never destroyed. Its GPU lease is retained until
            # another configured route is verified, rather than blindly re-renting.
            confirmed = True
        if not confirmed:
            return False
        if owner.provider != "instance":
            owner.released_at = datetime.now(timezone.utc)
            owner.status = "released"
            owner.release_reason = "Confirmed release for resource recovery"
    return True


async def _replacement_product(session, run, infra_id):
    products = (await session.scalars(select(WorkProduct).join(Ticket, Ticket.id == WorkProduct.ticket_id).where(
        Ticket.run_id == run.id, Ticket.id == infra_id, Ticket.status == "succeeded",
        WorkProduct.role == "device_info",
    ).order_by(WorkProduct.created_at.desc()))).all()
    return products[0] if products else None


async def _verify_resume(session, run, ticket, previous, info):
    """Never restart progressed Train at zero because a new host lost its state."""
    from zevo.engine.run.remote_jobs import _ssh
    from zevo.engine.run.training_continuation import verify_resume_checkpoint
    if ticket.agent_id != "train":
        return
    request = (previous.meta or {}).get("training_continuation", {}) if previous else {}
    checkpoints = (run.lifecycle or {}).get("adaptive_checkpoints", {})
    for stage in checkpoints.values():
        candidate = stage.get("request", {})
        if candidate.get("ticket_id") == ticket.id and candidate.get("step", 0) > request.get("step", request.get("completed_steps", 0)):
            request = candidate
    if request:
        checkpoint = request.get("checkpoint") or request.get("checkpoint_path")
        if not checkpoint:
            raise ValueError("Recovery requires the committed full-state checkpoint path")
        code = inspect.getsource(verify_resume_checkpoint) + '\n' + (
            f"verify_resume_checkpoint({checkpoint!r},step={request.get('step', request.get('completed_steps'))!r},"
            f"planned_steps={request['planned_steps']!r},world_size={request.get('world_size', int(previous.gpu_count or 1) if previous else 1)!r})\n"
            "print('ZEVO_CHECKPOINT_VERIFIED')"
        )
        result = await _ssh(info, "python3 -c " + shlex.quote(code), timeout_seconds=60, stdout_limit=2000)
        if not result.get("ok") or "ZEVO_CHECKPOINT_VERIFIED" not in result.get("stdout", ""):
            raise ValueError("Committed full-state checkpoint is not verified on the replacement host; restore durable state before retrying")
    else:
        from zevo.db import ExecutionEvent
        progress = await session.scalar(select(ExecutionEvent.id).where(
            ExecutionEvent.ticket_id == ticket.id, ExecutionEvent.current_step > 0,
        ).limit(1))
        if progress:
            raise ValueError("Training made progress but has no verified resumable checkpoint; recovery needs input")


async def _advance(session, run, ticket, event):
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo
    from zevo.engine.run.remote_jobs import _device_info_for_ticket
    from zevo.engine.run.wakeup import queue_wakeup
    from zevo.engine.run.runner import _latest_slurm_resource_request
    owner = await session.get(InfraInstance, event["owner_id"]) if event.get("owner_id") else None
    previous = await _latest_slurm_resource_request(session, ticket)
    info = await _device_info_for_ticket(session, ticket)
    if info is None:
        raise ValueError("Resource recovery requires the original verified device_info")
    if event["phase"] == "stopping":
        if not await _stop_owner(session, run, ticket, owner):
            return
        event = {**event, "phase": "provisioning"}
        save(run, ticket.id, event)
    if info.provider == "cluster":
        # The next finite submission acquires a fresh allocation; paths remain
        # on the cluster filesystem, and publication verifies them again.
        replacement = None
    elif info.provider == "instance":
        # Instance mode has a fixed configured host. Only an already provisioned,
        # verified alternative may replace it; do not invent/rent another server.
        alternatives = (await session.scalars(select(WorkProduct).join(Ticket, Ticket.id == WorkProduct.ticket_id).where(
            Ticket.run_id == run.id, Ticket.agent_id == "infrastructure", Ticket.status == "succeeded",
            WorkProduct.role == "device_info",
        ).order_by(WorkProduct.created_at.desc()))).all()
        replacement = None
        for product in alternatives:
            candidate = InfrastructureDeviceInfo.model_validate_json(Path(product.path).read_text())
            if (candidate.provider == "instance" and candidate.run_id == run.id
                and (candidate.ssh.host, candidate.ssh.port, candidate.instance.visible_devices)
                    != (info.ssh.host, info.ssh.port, info.instance.visible_devices)):
                replacement = product
                break
        if replacement is None:
            ticket.status = "awaiting_input"
            ticket.summary = "Resource recovery needs a configured, verified replacement instance"
            return
    else:
        consumers = [ticket_id for ticket_id, state in (run.lifecycle or {}).get(KEY, {}).items()
                     if state.get("owner_id") == event.get("owner_id") and state.get("phase") not in {"resumed", "blocked"}]
        repair_floor = max(int(ticket.repair_attempts or 0), int(await session.scalar(
            select(func.max(Ticket.repair_attempts)).where(Ticket.run_id == run.id, Ticket.id.in_(consumers))) or 0))
        infra_id = event.get("infra_ticket_id")
        if not infra_id:
            # Deterministic identity coalesces all consumers of one lost owner.
            purpose = "train" if ticket.agent_id in {"data", "train"} else "inference"
            if info.resource_plan:
                purpose = info.resource_plan.purpose
            elif owner and owner.ticket_id:
                source = await session.get(Ticket, owner.ticket_id)
                if source and source.agent_id == "infrastructure":
                    purpose = (source.payload or {}).get("purpose", purpose)
            token = hashlib.sha256(f"{run.id}:{event['owner_id']}:{purpose}".encode()).hexdigest()[:24]
            infra_id = "infra-recovery-" + token
            event = {**event, "infra_ticket_id": infra_id}
            save(run, ticket.id, event)
            infra = await session.get(Ticket, infra_id)
            if infra is None:
                infra = Ticket(id=infra_id, run_id=run.id, agent_id="infrastructure", status="queued",
                    iteration=ticket.iteration, payload={"operation": "provision", "purpose": purpose},
                    repair_attempts=repair_floor)
                session.add(infra)
                await session.flush()
                await queue_wakeup(session, agent_id="infrastructure", ticket_id=infra_id,
                    source="resource_recovery", reason="Replace confirmed unusable compute within the unchanged Run resource plan")
                return
        infra = await session.get(Ticket, infra_id)
        infra.repair_attempts = max(int(infra.repair_attempts or 0), repair_floor)
        ticket.repair_attempts = max(int(ticket.repair_attempts or 0), int(infra.repair_attempts or 0))
        if infra.status in {"failed", "cancelled"}:
            raise ValueError("Replacement Infrastructure could not complete; inspect its recorded failure")
        replacement = await _replacement_product(session, run, infra_id)
        if replacement is None:
            return
    if replacement:
        new_info = InfrastructureDeviceInfo.model_validate_json(Path(replacement.path).read_text())
        if new_info.run_id != run.id or new_info.provider != info.provider:
            raise ValueError("Replacement belongs to another Run or provider")
        if new_info.ticket_id != replacement.ticket_id:
            raise ValueError("Replacement device_info does not belong to its Infrastructure Ticket")
        if info.provider == "cloud" and (new_info.instance_id == info.instance_id or new_info.cloud_backend != info.cloud_backend):
            raise ValueError("Replacement must be a different allocation on the same cloud backend")
        if (new_info.resource_plan.num_gpus != info.resource_plan.num_gpus
            or new_info.resource_plan.nodes != info.resource_plan.nodes
            or new_info.resource_plan.purpose != info.resource_plan.purpose
            or (info.resource_plan.gpu_type and new_info.resource_plan.gpu_type != info.resource_plan.gpu_type)
            or any(getattr(new_info.resource_plan, key) < getattr(info.resource_plan, key)
                   for key in ("min_vram_gb", "min_ram_gb", "min_cpus", "disk_gb"))):
            raise ValueError("Replacement changed the registered resource requirements")
        if info.provider == "cloud":
            registered = await session.scalar(select(InfraInstance.id).where(
                InfraInstance.run_id == run.id, InfraInstance.instance_id == new_info.instance_id,
                InfraInstance.provider == "cloud", InfraInstance.status == "ready", InfraInstance.released_at.is_(None),
            ).limit(1))
            if not registered:
                raise ValueError("Replacement cloud allocation is not registered and ready")
        await _verify_resume(session, run, ticket, previous, new_info)
        if info.provider == "instance":
            from zevo.db import GpuLease
            leases = (await session.scalars(select(GpuLease).where(
                GpuLease.run_id == run.id, GpuLease.released_at.is_(None),
            ))).all()
            new_cards = {lease.gpu_index for lease in leases
                         if lease.ticket_id == new_info.ticket_id and lease.jobid == new_info.instance_id}
            if new_info.ticket_id == info.ticket_id or new_cards != {d.index for d in new_info.gpu.devices}:
                raise ValueError("Replacement instance requires its own verified live GPU leases")
            for lease in leases:
                if lease.ticket_id == info.ticket_id:
                    lease.released_at = datetime.now(timezone.utc)
                    lease.release_reason = "Original workload stopped; verified replacement instance bound"
            if owner:
                owner.released_at = datetime.now(timezone.utc)
                owner.status = "released"
        ticket.inputs = {**ticket.inputs, "device_info": {"source_ticket_id": replacement.ticket_id,
            "work_product_id": replacement.id, "artifact_role": "device_info", "path": replacement.path}}
        run.lifecycle = {**run.lifecycle, "device_replacements": {
            **(run.lifecycle or {}).get("device_replacements", {}), info.ticket_id: replacement.id}}
    if previous:
        record_execution_repair(previous, action="resubmit", reason=event["reason"], heartbeat_id="")
    save(run, ticket.id, {**event, "phase": "resumed", "replacement_product_id": replacement.id if replacement else ""})
    ticket.status = "repairing"
    ticket.repair_route = "self"
    ticket.error_message = ""
    session.add(TicketMessage(ticket_id=ticket.id, author="system", body=(
        "__REPAIR__\n\nResource recovery is ready. Continue this same Ticket with its existing frozen configuration, "
        "prepared implementation and verified state. Validate all inputs on the current host. "
        "Remote-only artifacts missing from a lost host must be restored; never silently restart completed work."
    )))
    await queue_wakeup(session, agent_id=ticket.agent_id, ticket_id=ticket.id, source="retry",
        trigger_detail=f"repair_{ticket.repair_attempts}", reason="Continue after resource recovery",
        payload={"repair_attempt": ticket.repair_attempts})


async def reconcile(session):
    from zevo.engine.run.steering import instruction_gate_active
    from zevo.engine.cost.budget import snapshot_for_run
    ids = (await session.scalars(select(Run.id).where(Run.status.notin_(list(TERMINAL_RUN_STATUSES)),
        Run.cancel_requested_at.is_(None)))).all()
    resumed = 0
    for run_id in ids:
        run = (await session.scalars(select(Run).where(Run.id == run_id).with_for_update(skip_locked=True)
            .execution_options(populate_existing=True))).one_or_none()
        if run is None:
            continue
        if await instruction_gate_active(session, run.id):
            await session.commit()
            continue
        for ticket_id, event in list((run.lifecycle or {}).get(KEY, {}).items()):
            # Queue publication commits its state. Reacquire before processing
            # another consumer so a second scheduler or a pause cannot race it.
            run = (await session.scalars(select(Run).where(Run.id == run_id).with_for_update(skip_locked=True)
                .execution_options(populate_existing=True))).one_or_none()
            if run is None:
                break
            if run.status in TERMINAL_RUN_STATUSES or run.cancel_requested_at or await instruction_gate_active(session, run.id):
                break
            event = (run.lifecycle or {}).get(KEY, {}).get(ticket_id, event)
            if event["phase"] in {"resumed", "blocked"}:
                continue
            if event.get("next_check_at") and datetime.fromisoformat(event["next_check_at"]) > datetime.now(timezone.utc):
                continue
            ticket = await session.get(Ticket, ticket_id, populate_existing=True)
            if ticket is None or ticket.status in {"cancelled", "succeeded", "degraded"} or ticket.repair_route == "terminal":
                continue
            if await session.scalar(select(HeartbeatRun.id).where(HeartbeatRun.ticket_id == ticket.id,
                HeartbeatRun.finished_at.is_(None)).limit(1)):
                continue
            budget = await snapshot_for_run(session, run.id)
            if (budget.over_budget or budget.over_time_limit or (run.lifecycle or {}).get("finalization")
                or int(ticket.repair_attempts or 0) > MAX_REPAIR_ATTEMPTS):
                save(run, ticket.id, {**event, "phase": "blocked", "error": "Run budget or repair boundary reached"})
                ticket.status = "failed"
                ticket.repair_route = "terminal"
                continue
            try:
                await _advance(session, run, ticket, event)
                resumed += int((run.lifecycle or {})[KEY][ticket.id]["phase"] == "resumed")
                await session.commit()
            except Exception as exc:
                # Keep committed provisioning identity on retries; never issue
                # a second rental because SSH/verification failed.
                await session.rollback()
                log.exception("Resource recovery is waiting for %s", ticket_id)
                run = (await session.scalars(select(Run).where(Run.id == run_id).with_for_update()
                    .execution_options(populate_existing=True))).one()
                ticket = await session.get(Ticket, ticket_id, populate_existing=True)
                if (ticket is None or ticket.status in {"cancelled", "succeeded", "degraded"} or ticket.repair_route == "terminal"
                    or run.status in TERMINAL_RUN_STATUSES or run.cancel_requested_at
                    or await instruction_gate_active(session, run.id)):
                    await session.commit()
                    continue
                latest = (run.lifecycle or {}).get(KEY, {}).get(ticket_id, event)
                if latest.get("phase") in {"resumed", "blocked"}:
                    await session.commit()
                    continue
                if latest.get("infra_ticket_id"):
                    infra = await session.get(Ticket, latest["infra_ticket_id"], populate_existing=True)
                    if infra:
                        ticket.repair_attempts = max(int(ticket.repair_attempts or 0), int(infra.repair_attempts or 0))
                save(run, ticket_id, {**latest, "last_error": type(exc).__name__,
                    "next_check_at": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()})
                ticket.status = "awaiting_input"
                ticket.summary = str(exc)[:400]
                await session.commit()
        await session.commit()
    return resumed
