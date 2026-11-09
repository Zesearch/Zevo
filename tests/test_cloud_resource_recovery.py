from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from tests.test_direct_jobs import database
from zevo.db import Agent, AgentWakeupRequest, InfraInstance, Ticket
from zevo.engine.run import cloud_recovery as recovery, direct_jobs, resource_cleanup


@pytest_asyncio.fixture
async def setup(tmp_path, monkeypatch):
    engine, db, run, ticket, owner, inp = await database(tmp_path)
    run.cloud_backend = "aws"
    run.supervisor_ticket_id = "orch"
    owner.meta = {"backend": "aws", "region": "us-west-2"}
    db.add(Ticket(id="orch", run_id=run.id, agent_id="orchestrator", status="succeeded"))
    keys = {"AWS_ACCESS_KEY_ID": "personal-key", "AWS_SECRET_ACCESS_KEY": "personal-secret", "AWS_REGION": "us-east-1"}
    calls = []
    provider = SimpleNamespace(state="running", error=None)
    async def scope(session, owning_run, allocation):
        assert session is db and owning_run is run and allocation is owner
        return keys
    async def get(instance_id):
        calls.append(instance_id)
        if provider.error:
            raise provider.error
        return {"id": instance_id, "status": provider.state, "region": "us-west-2"}
    provider.get_instance = get
    def factory(backend, key, secret, **kwargs):
        assert backend == "aws" and key == keys["AWS_ACCESS_KEY_ID"]
        assert secret == keys["AWS_SECRET_ACCESS_KEY"]
        assert kwargs == {"region": "us-west-2"}  # saved region wins over changed user default
        return provider
    async def remote(*args, **kwargs): return {}
    monkeypatch.setattr(resource_cleanup, "cloud_recovery_keys", scope)
    monkeypatch.setattr(resource_cleanup, "_cloud_provider", factory)
    monkeypatch.setattr(direct_jobs, "remote_call", remote)
    await db.commit()
    job = await direct_jobs.submit(db, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
    try:
        yield SimpleNamespace(db=db, run=run, ticket=ticket, owner=owner, job=job,
                              provider=provider, keys=keys, calls=calls)
    finally:
        await db.close()
        await engine.dispose()


async def tick(s):
    health = dict(s.owner.meta.get("cloud_health", {}))
    health.pop("next_at", None)
    s.owner.meta = {**s.owner.meta, "cloud_health": health}
    await s.db.commit()
    return await recovery.reconcile(s.db)


@pytest.mark.asyncio
async def test_terminated_allocation_hands_off_once_and_never_republishes(setup, monkeypatch):
    s = setup
    s.provider.state = "terminated"
    assert await tick(s) == 1
    assert s.owner.released_at is not None and s.job.released_at is not None
    assert s.job.meta["scheduler_state"] == "NODE_FAIL"
    assert s.ticket.status == "failed" and s.ticket.repair_route == "orchestrator"
    assert s.ticket.repair_attempts == 1
    assert s.run.lifecycle["cloud_resource_recovery_attempts"] == 1
    wakes = (await s.db.scalars(select(AgentWakeupRequest))).all()
    assert len(wakes) == 1 and wakes[0].source == "resource_recovery"
    assert wakes[0].payload["cloud_resource_recovery"]["replacement_allowed"]
    assert "checkpoint" in wakes[0].reason and "Remote-only" in wakes[0].reason
    assert await recovery.reconcile(s.db) == 0
    async def forbidden(*args, **kwargs): pytest.fail("lost allocation must not be republished")
    monkeypatch.setattr(direct_jobs, "remote_call", forbidden)
    assert await direct_jobs.reconcile(s.db) == (0, 0)
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
async def test_absence_requires_prior_positive_identity_and_two_observations(setup):
    s = setup
    s.provider.state = "not-found"
    assert await tick(s) == await tick(s) == 0
    assert s.owner.released_at is None
    s.provider.state = "running"
    assert await tick(s) == 0
    s.provider.state = "not-found"
    assert await tick(s) == 0
    # Backoff survives another reconciliation without forcing the deadline.
    calls = len(s.calls)
    assert await recovery.reconcile(s.db) == 0 and len(s.calls) == calls
    assert await tick(s) == 1


@pytest.mark.asyncio
async def test_changed_credentials_invalidate_old_negative_inventory_evidence(setup):
    s = setup
    await tick(s)
    s.keys["AWS_ACCESS_KEY_ID"] = "different-account"
    s.provider.state = "not-found"
    assert await tick(s) == await tick(s) == 0
    assert s.owner.released_at is None and s.ticket.status == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "pending", "stopped", "stopping", "shutting-down", "unknown"])
async def test_nonterminal_or_ambiguous_state_does_not_rent_a_replacement(setup, state):
    s = setup
    s.provider.state = state
    assert await tick(s) == 0
    assert s.owner.released_at is None and s.job.released_at is None
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
async def test_permission_error_is_not_absence_and_breaks_missing_streak(setup):
    s = setup
    await tick(s)
    s.provider.state = "not-found"
    await tick(s)
    s.provider.error = PermissionError("private credential details must not be saved")
    assert await tick(s) == 0
    health = s.owner.meta["cloud_health"]
    assert health["missing_count"] == 0 and "PermissionError" in health["error"]
    assert "private credential" not in str(s.owner.meta)
    s.provider.error = None
    assert await tick(s) == 0
    assert await tick(s) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["repair_limit", "budget", "finalization"])
async def test_replacement_respects_existing_repair_and_run_limits(setup, boundary):
    s = setup
    if boundary == "repair_limit": s.ticket.repair_attempts = 20
    if boundary == "budget":
        s.run.max_runtime_hours = 1
        s.run.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    if boundary == "finalization": s.run.lifecycle = {"finalization": {"reason": "limit"}}
    s.provider.state = "terminated"
    assert await tick(s) == 1
    event = s.run.lifecycle["cloud_resource_recoveries"][s.owner.id]
    assert not event["replacement_allowed"]
    assert s.ticket.repair_route == "terminal" and s.ticket.repair_attempts <= 20
    assert "Do not provision" in event["next_action"]


@pytest.mark.asyncio
async def test_cancelled_run_does_not_probe_or_recover(setup):
    s = setup
    s.run.cancel_requested_at = datetime.now(timezone.utc)
    s.provider.state = "terminated"
    assert await tick(s) == 0 and not s.calls


@pytest.mark.asyncio
async def test_new_ready_allocation_is_reused_in_coordinator_context(setup):
    s = setup
    s.provider.state = "terminated"
    await tick(s)
    replacement = InfraInstance(id="replacement", run_id=s.run.id,
        provider="cloud", instance_id="new-machine", status="ready",
        created_at=datetime.now(timezone.utc) + timedelta(seconds=1))
    s.db.add(replacement)
    await s.db.commit()
    context = await recovery.recovery_context(s.db, s.run)
    assert context[0]["replacement_allocation_id"] == replacement.id
    assert "do not provision again" in context[0]["next_action"]


@pytest.mark.asyncio
async def test_failed_handoff_rolls_back_release_and_retries(setup, monkeypatch):
    from zevo.engine.run import wakeup
    s = setup
    original = wakeup.queue_wakeup
    async def broken(*args, **kwargs): raise RuntimeError("queue unavailable")
    monkeypatch.setattr(wakeup, "queue_wakeup", broken)
    s.provider.state = "terminated"
    assert await tick(s) == 0
    await s.db.refresh(s.owner)
    await s.db.refresh(s.job)
    await s.db.refresh(s.run)
    assert s.owner.released_at is None and s.job.released_at is None
    assert not s.run.lifecycle.get("cloud_resource_recoveries")
    monkeypatch.setattr(wakeup, "queue_wakeup", original)
    assert await tick(s) == 1


@pytest.mark.asyncio
async def test_exhausted_recovery_blocks_provision_api(setup):
    from fastapi import HTTPException
    from zevo.api.routers.shared.tickets import create_ticket
    from zevo.contracts.tickets import CreateTicketBody
    s = setup
    s.ticket.repair_attempts = 20
    s.provider.state = "terminated"
    await tick(s)
    s.db.add(Agent(id="infrastructure", name="Infrastructure", title="Infra", identity_path=""))
    await s.db.commit()
    with pytest.raises(HTTPException) as caught:
        await create_ticket(CreateTicketBody(run_id=s.run.id, agent_id="infrastructure",
            payload={"operation": "provision"}), s.db)
    assert caught.value.status_code == 409 and "recovery stopped" in str(caught.value.detail)


@pytest.mark.asyncio
async def test_private_workload_failure_exposes_only_allocation_facts(setup):
    s = setup
    s.ticket.lane = "held_out_test"
    s.ticket.error_message = "private answer and score"
    s.provider.state = "terminated"
    await tick(s)
    assert s.ticket.repair_route == "terminal"
    wake = (await s.db.scalars(select(AgentWakeupRequest))).one()
    assert "private answer" not in str(wake.payload) + wake.reason
    assert s.ticket.id not in str(wake.payload)


@pytest.mark.asyncio
async def test_confirmed_lost_cloud_host_needs_no_ssh_cleanup(setup, monkeypatch):
    from zevo.engine.run import remote_jobs
    from tests.test_direct_jobs import info_for
    s = setup
    s.provider.state = "terminated"
    await tick(s)
    async def route(*args): return info_for("cloud")
    async def forbidden(*args, **kwargs): pytest.fail("do not connect to destroyed hosts")
    monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", route)
    monkeypatch.setattr(remote_jobs, "_ssh", forbidden)
    assert (await remote_jobs.cancel_ticket_remote_job(s.db, s.ticket))["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["lambda", "vastai"])
async def test_non_aws_inventory_uses_same_loss_confirmation(setup, monkeypatch, backend):
    s = setup
    s.run.cloud_backend = ""  # An unpinned Run uses its recorded backend.
    s.owner.meta = {"backend": backend}
    s.keys.update(LAMBDA_API_KEY="lambda-key", VASTAI_API_KEY="vast-key")
    inventory = [{"id": s.owner.instance_id, "actual_status": "running"}]
    async def instances(): return inventory
    def factory(name, key, secret):
        assert name == backend and key == s.keys[{"lambda": "LAMBDA_API_KEY", "vastai": "VASTAI_API_KEY"}[backend]]
        return SimpleNamespace(list_instances=instances)
    monkeypatch.setattr(resource_cleanup, "_cloud_provider", factory)
    assert await tick(s) == 0
    inventory.clear()
    assert await tick(s) == 0
    assert await tick(s) == 1
    assert s.run.lifecycle["cloud_resource_recoveries"][s.owner.id]["backend"] == backend


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "collect"])
async def test_loss_also_closes_agents_outside_a_live_workload(setup, tmp_path, phase):
    from tests.test_direct_jobs import info_for
    s = setup
    device = tmp_path / "bound-device.json"
    device.write_text(info_for("cloud").model_dump_json())
    s.ticket.inputs = {"device_info": {"path": str(device), "artifact_role": "device_info"}}
    if phase == "prepare":
        await s.db.delete(s.job)
    else:
        s.job.released_at = datetime.now(timezone.utc)
        s.job.meta = {**s.job.meta, "scheduler_state": "COMPLETED"}
    s.provider.state = "terminated"
    assert await tick(s) == 1
    assert s.ticket.status == "failed"
    assert await recovery.binding_lost(s.db, s.run.id, str(device))
    info = info_for("cloud").model_copy(update={"instance_id": "new-host"})
    device.write_text(info.model_dump_json())
    assert not await recovery.binding_lost(s.db, s.run.id, str(device))


@pytest.mark.asyncio
async def test_old_host_loss_does_not_fail_work_rebound_to_a_new_host(setup, tmp_path):
    from tests.test_direct_jobs import info_for
    s = setup
    device = tmp_path / "rebound.json"
    device.write_text(info_for("cloud").model_copy(update={"instance_id": "replacement-host"}).model_dump_json())
    s.ticket.inputs = {"device_info": {"path": str(device), "artifact_role": "device_info"}}
    s.job.released_at = datetime.now(timezone.utc)
    s.job.meta = {**s.job.meta, "scheduler_state": "COMPLETED"}
    s.provider.state = "terminated"
    assert await tick(s) == 1
    assert s.ticket.status == "running" and s.ticket.repair_attempts == 0
    assert s.job.meta["scheduler_state"] == "COMPLETED"


@pytest.mark.asyncio
async def test_unknown_aws_region_never_uses_ambient_default(setup, monkeypatch):
    s = setup
    s.owner.meta = {"backend": "aws"}
    s.keys.pop("AWS_REGION")
    def forbidden(*args, **kwargs): pytest.fail("must not query an unverified region")
    monkeypatch.setattr(resource_cleanup, "_cloud_provider", forbidden)
    assert await tick(s) == 0
    assert s.owner.released_at is None and s.owner.meta["cloud_health"]["error"]


@pytest.mark.asyncio
async def test_recovery_review_can_emit_infrastructure_through_normal_api(setup):
    from zevo.api.routers.shared.tickets import create_ticket
    from zevo.contracts.tickets import CreateTicketBody
    from zevo.db import HeartbeatRun
    s = setup
    s.provider.state = "terminated"
    await tick(s)
    s.db.add(Agent(id="infrastructure", name="Infrastructure", title="Infra", identity_path=""))
    s.db.add(HeartbeatRun(id="recovery-review", ticket_id="orch", agent_id="orchestrator", driver="test", model="test"))
    await s.db.commit()
    created = await create_ticket(CreateTicketBody(run_id=s.run.id, agent_id="infrastructure",
        payload={"operation": "provision", "purpose": "train"}), s.db)
    replacement = await s.db.get(Ticket, created.id)
    assert replacement.agent_id == "infrastructure" and replacement.payload["operation"] == "provision"
    # Infra dispatch still waits for this review's validated completion.
    assert not (await s.db.scalars(select(AgentWakeupRequest).where(
        AgentWakeupRequest.ticket_id == replacement.id))).all()


@pytest.mark.asyncio
async def test_new_ticket_id_preserves_repairs_but_new_iteration_has_own_allowance(setup):
    s = setup
    s.ticket.repair_attempts = 7
    s.ticket.payload = {"base_model": "model-a", "model_source": "base_model"}
    s.run.lifecycle = {"cloud_resource_recovery_attempts": 25}
    s.provider.state = "terminated"
    assert await tick(s) == 1  # no new Run-wide cap across unrelated iterations
    assert s.ticket.repair_attempts == 8
    assert await recovery.inherited_repairs(s.db, s.run, "inference", s.ticket.iteration, s.ticket.payload) == 8
    assert await recovery.inherited_repairs(s.db, s.run, "inference", s.ticket.iteration + 1, s.ticket.payload) == 0
    assert await recovery.inherited_repairs(s.db, s.run, "inference", s.ticket.iteration,
                                           {**s.ticket.payload, "base_model": "model-b"}) == 0
