"""A resource failure before launch must enter recovery, not another diagnosis."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from tests.test_ssh_provider_contract import _device
from tests.test_direct_jobs import info_for
from tests.test_ticket_repair_state import session
from zevo.contracts._base import AgentResult
from zevo.contracts.infrastructure import InfrastructureDeviceInfo
from zevo.db import Run, Ticket, InfraInstance, AgentWakeupRequest, TicketNotice, WorkProduct
from zevo.engine.run import resource_recovery
from zevo.engine.run.runner import _apply_ticket_failure_policy, _queue_ticket_repair


async def prepared(session, tmp_path, provider, phase, agent="train", first=False):
    info = InfrastructureDeviceInfo.model_validate(_device("cluster")) if provider == "cluster" else info_for(provider)
    path = tmp_path / "device.json"
    path.write_text(info.model_dump_json())
    run = Run(id=info.run_id, metric="score", status="running", gpu_provider=provider)
    ticket = Ticket(id="stage", run_id=run.id, agent_id=agent, status="running",
                    iteration=2, repair_attempts=2, payload={"fixed": "plan"},
                    inputs={"device_info": {"path": str(path)}})
    owner = InfraInstance(id="owner", run_id=run.id, ticket_id=ticket.id,
        provider=provider, instance_id="123" if provider == "cluster" else info.instance_id, status="ready")
    execution = None
    if not first:
        meta = {"resource_request": True, "execution_attempt": 2,
                "scheduler_state": "COMPLETED", "monitor_terminal": True}
        if provider == "cluster":
            # A retained allocation is also the earlier training execution.
            execution = owner
            owner.meta = meta
        else:
            execution = InfraInstance(id="execution", run_id=run.id, ticket_id=ticket.id,
                provider=provider, instance_id="workload-2", status="released",
                released_at=datetime.now(timezone.utc),
                meta={**meta, "direct_workload": True, "allocation_owner_row_id": owner.id})
            session.add(execution)
    session.add_all([run, ticket, owner, WorkProduct(ticket_id=ticket.id, role="checkpoint",
                                                    path="/durable/checkpoint-1468")])
    await session.commit()
    collecting = phase == "collect"
    contract = SimpleNamespace(enabled=True, phase=phase,
        transport="slurm" if provider == "cluster" else "direct",
        attempt=1 if first else 2 if collecting else 3,
        bookkeeping_row_id=execution.id if execution and collecting else "",
        job_id=execution.instance_id if execution and collecting else "",
        retry_of_bookkeeping_row_id=execution.id if execution and not collecting else "",
        retry_of_job_id=execution.instance_id if execution and not collecting else "")
    inp = SimpleNamespace(slurm_job=contract, device_info_path=str(path))
    result = AgentResult(ticket_id=ticket.id, status="failed", notes="", error_message="Resource cannot run the assigned work",
        execution_retry_action="replace_resource", execution_retry_reason="Replace the assigned resource within the same plan",
        failure_recovery={"scope": "current_task", "reason": "The assigned resource failed its verified readiness check",
                          "evidence": ["resource-health.json: readiness check failed"],
                          "next_action": "Replace compute and resume the verified state"})
    return SimpleNamespace(run=run, ticket=ticket, owner=owner, execution=execution, inp=inp, result=result)


async def fail(session, state, **kwargs):
    return await _apply_ticket_failure_policy(session, state.ticket,
        error_message=state.result.error_message, result=state.result,
        inp=state.inp, heartbeat_id="failed-activation", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["cluster", "cloud", "instance"])
@pytest.mark.parametrize("phase", ["submit", "collect"])
@pytest.mark.parametrize("agent", ["data", "train", "inference", "evaluation"])
@pytest.mark.parametrize("lane", ["optimization", "held_out_test"])
async def test_replacement_routes_by_action_across_stages_phases_and_providers(session, tmp_path, provider, phase, agent, lane):
    s = await prepared(session, tmp_path, provider, phase, agent)
    s.ticket.lane = lane
    assert await fail(session, s)
    event = s.run.lifecycle[resource_recovery.KEY][s.ticket.id]
    assert event["owner_id"] == s.owner.id and event["phase"] == "stopping"
    assert s.ticket.status == "waiting_external" and s.ticket.repair_attempts == 3
    assert s.execution.meta["execution_repair"]["heartbeat_id"] == "failed-activation"
    await session.commit()
    await _queue_ticket_repair(session, s.ticket)
    await _queue_ticket_repair(session, s.ticket)
    assert not (await session.scalars(select(AgentWakeupRequest))).all()
    assert s.owner.released_at is None  # only the reconciler may stop/release it
    assert s.ticket.iteration == 2 and s.ticket.payload == {"fixed": "plan"}
    assert (await session.scalar(select(WorkProduct))).path == "/durable/checkpoint-1468"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["cloud", "instance"])
async def test_first_submission_replaces_assigned_host_without_execution_row(session, tmp_path, provider):
    s = await prepared(session, tmp_path, provider, "submit", first=True)
    assert await fail(session, s)
    event = s.run.lifecycle[resource_recovery.KEY][s.ticket.id]
    assert event["owner_id"] == event["execution_id"] == s.owner.id
    await _queue_ticket_repair(session, s.ticket)
    assert not (await session.scalars(select(AgentWakeupRequest))).all()
    assert len((await session.scalars(select(InfraInstance))).all()) == 1


@pytest.mark.asyncio
async def test_submit_after_host_replacement_targets_new_owner_and_preserves_old_execution(session, tmp_path):
    s = await prepared(session, tmp_path, "cloud", "submit")
    s.owner.released_at = datetime.now(timezone.utc)
    new_info = info_for("cloud").model_copy(update={"instance_id": "new-host"})
    new_path = tmp_path / "new-device.json"
    new_path.write_text(new_info.model_dump_json())
    s.ticket.inputs = {"device_info": {"path": str(new_path)}}
    s.inp.device_info_path = str(new_path)
    new_owner = InfraInstance(id="new-owner", run_id=s.run.id, provider="cloud",
                             instance_id="new-host", status="ready")
    session.add(new_owner)
    await session.commit()
    original = dict(s.execution.meta)
    assert await fail(session, s)
    event = s.run.lifecycle[resource_recovery.KEY][s.ticket.id]
    assert event["owner_id"] == event["execution_id"] == new_owner.id
    assert s.execution.meta == original
    await _queue_ticket_repair(session, s.ticket)
    assert not (await session.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["limit", "cancel", "isolation"])
async def test_replacement_cannot_bypass_failure_boundaries(session, tmp_path, boundary):
    s = await prepared(session, tmp_path, "cluster", "submit")
    kwargs = {}
    if boundary == "limit": s.ticket.repair_attempts = 20
    if boundary == "cancel": kwargs["cancelled"] = True
    if boundary == "isolation": kwargs["boundary"] = "held_out_isolation"
    count = s.ticket.repair_attempts
    assert not await fail(session, s, **kwargs)
    assert resource_recovery.KEY not in (s.run.lifecycle or {})
    assert s.ticket.repair_attempts == count and s.owner.released_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["new_execution", "job", "attempt", "released", "foreign_owner", "binding", "unassigned"])
async def test_unbound_replacement_is_reviewed_without_repair_loop(session, tmp_path, mismatch):
    s = await prepared(session, tmp_path, "cluster", "submit", first=mismatch == "unassigned")
    if mismatch == "new_execution":
        session.add(InfraInstance(id="new", run_id=s.run.id, ticket_id=s.ticket.id,
            provider="cluster", instance_id=s.owner.instance_id, status="ready",
            created_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            meta={"resource_request": True, "execution_attempt": 3}))
    if mismatch == "job": s.inp.slurm_job.retry_of_job_id = "other"
    if mismatch == "attempt": s.inp.slurm_job.attempt = 4
    if mismatch == "released": s.owner.released_at = datetime.now(timezone.utc)
    if mismatch == "foreign_owner":
        session.add_all([Run(id="foreign", metric="score"), InfraInstance(id="foreign-owner",
            run_id="foreign", provider="cluster", instance_id="foreign-job", status="ready")])
        s.execution.meta = {**s.execution.meta, "allocation_owner_row_id": "foreign-owner"}
    if mismatch == "binding": s.ticket.inputs = {}
    await session.commit()
    assert not await fail(session, s)
    assert s.ticket.status == "failed" and s.ticket.repair_route == "orchestrator"
    assert s.ticket.repair_attempts == 2
    assert resource_recovery.KEY not in (s.run.lifecycle or {})
    assert await session.scalar(select(TicketNotice.id).where(TicketNotice.code == "resource.replacement_rejected"))


@pytest.mark.asyncio
async def test_resubmit_remains_an_ordinary_repair_not_resource_replacement(session, tmp_path):
    s = await prepared(session, tmp_path, "cluster", "submit")
    s.result.execution_retry_action = "resubmit"
    assert await fail(session, s)
    assert s.ticket.status == "repairing" and s.ticket.repair_attempts == 3
    assert resource_recovery.KEY not in (s.run.lifecycle or {})
