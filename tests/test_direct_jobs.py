from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tests.test_ssh_provider_contract import _device
from zevo.contracts.infrastructure import InfrastructureDeviceInfo, SlurmStageJobContract
from zevo.db.models import Base, Run, Ticket, InfraInstance, HeartbeatRun, AgentWakeupRequest, ExecutionEvent
from zevo.engine.run import direct_jobs as direct


def info_for(provider="instance"):
    raw = _device("instance")
    if provider == "cloud":
        raw.update(provider="cloud", cloud_backend="aws", auto_release=True)
        raw["resource_plan"]["cloud_backend"] = "aws"
    return InfrastructureDeviceInfo.model_validate(raw)


def job_for(tmp_path, *, previous=None, retry=False):
    return direct.contract_for(info=info_for(), ticket=SimpleNamespace(id="infer-one"),
        work_dir=str(tmp_path), stage="inference", previous=previous, retry=retry)


@pytest.mark.parametrize("provider", ["instance", "cloud"])
def test_existing_host_contract_has_no_slurm_or_new_allocation(tmp_path, provider):
    job = direct.contract_for(info=info_for(provider), ticket=SimpleNamespace(id="infer-one"),
        work_dir=str(tmp_path), stage="inference", previous=None, retry=False)
    assert job.enabled and job.transport == "direct" and job.phase == "submit"
    assert job.script_path.endswith(".sh") and job.stdout_path.endswith("workload.out")
    assert not job.lifecycle_prologue and not job.runtime_prologue
    assert job.num_gpus == 1 and job.nodes == 1


def test_retry_and_collect_bind_exact_execution_and_preserve_history(tmp_path):
    initial = job_for(tmp_path)
    previous = SimpleNamespace(id="row", instance_id="execution", meta={
        "execution_contract": initial.model_dump(), "scheduler_state": "FAILED",
        "execution_attempt": 1, "implementation_sha256": "original-hash"})
    collected = job_for(tmp_path, previous=previous)
    assert collected.phase == "collect" and collected.bookkeeping_row_id == "row"
    assert collected.implementation_sha256 == "original-hash"
    retried = job_for(tmp_path, previous=previous, retry=True)
    assert retried.phase == "submit" and retried.attempt == 2
    assert retried.remote_work_dir.endswith("attempt-2")
    assert retried.previous_execution["implementation_sha256"] == "original-hash"
    assert retried.previous_execution["job_id"] == "execution"
    SlurmStageJobContract.model_validate(retried.previous_execution)


@pytest.mark.parametrize("change", [{"nodes": 2, "num_gpus": 2},
    {"status_path": "/elsewhere/status"}, {"script_path": "/tmp/job.sbatch"},
    {"phase": "collect"}, {"attempt": 2}])
def test_direct_contract_rejects_unowned_paths_and_unbound_attempts(tmp_path, change):
    data = job_for(tmp_path).model_dump()
    with pytest.raises(ValidationError):
        SlurmStageJobContract.model_validate({**data, **change})


async def database(tmp_path, provider="cloud"):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    info = info_for(provider)
    device = tmp_path / "device.json"
    device.write_text(info.model_dump_json())
    run = Run(id=info.run_id, metric="accuracy", gpu_provider=provider, status="running",
              gpu_allocation_mode="per_run", started_at=datetime.now(timezone.utc))
    ticket = Ticket(id="infer-one", run_id=run.id, agent_id="inference", status="running")
    owner = InfraInstance(id="allocation", run_id=run.id, provider=provider, instance_id=info.instance_id,
                          status="ready", gpu_count=1, gpu_name="A100", dph=2)
    hb = HeartbeatRun(id="submit-hb", ticket_id=ticket.id, agent_id="inference", driver="test", model="test",
                      activation_phase="submit", finished_at=datetime.now(timezone.utc), exit_code=0)
    session.add_all([run, ticket, owner, hb]); await session.commit()
    job = direct.contract_for(info=info, ticket=ticket, work_dir=str(tmp_path), stage="inference", previous=None, retry=False)
    Path(job.script_path).write_text("#!/bin/bash\npython predict.py\n")
    (tmp_path / "predict.py").write_text("print('work')\n")
    inp = SimpleNamespace(slurm_job=job, device_info_path=str(device))
    return engine, session, run, ticket, owner, inp


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["instance", "cloud"])
async def test_durable_publication_recovery_and_collection_are_idempotent(tmp_path, monkeypatch, provider):
    engine, session, run, ticket, owner, inp = await database(tmp_path, provider)
    calls = []
    async def failed_publication(*args, **kwargs):
        calls.append("ambiguous")
        raise RuntimeError("SSH response lost")
    monkeypatch.setattr(direct, "remote_call", failed_publication)
    try:
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        assert row.meta["submission_committed"] and row.meta["allocation_owner_row_id"] == owner.id
        assert owner.released_at is None and row.meta["monitor_error"] == "SSH response lost"
        ticket.status = "waiting_external"; await session.commit()
        async def recovered(info, manifest, action, **kwargs):
            calls.append((manifest["execution_id"], action))
            return {"execution_id": row.id, "state": "COMPLETED", "exit_code": 0}
        async def telemetry(session, row, ticket, info, meta):
            meta["telemetry_drained"] = True
        monkeypatch.setattr(direct, "remote_call", recovered)
        monkeypatch.setattr(direct, "read_telemetry", telemetry)
        assert await direct.reconcile(session) == (1, 1)
        assert ticket.status == "queued" and owner.released_at is None
        assert row.released_at is not None
        assert await direct.reconcile(session) == (0, 0)
        wakes = (await session.scalars(select(AgentWakeupRequest))).all()
        assert len(wakes) == 1 and wakes[0].source == "workload_watcher"
        assert wakes[0].payload["resource_row_id"] == row.id
        assert len((await session.scalars(select(InfraInstance))).all()) == 2
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
async def test_active_collection_and_cancelled_run_cannot_republish(tmp_path, monkeypatch):
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    async def remote(info, manifest, action, **kwargs):
        return {"execution_id": manifest["execution_id"], "state": "COMPLETED", "exit_code": 0}
    async def telemetry(session, row, ticket, info, meta): meta["telemetry_drained"] = True
    monkeypatch.setattr(direct, "remote_call", remote)
    monkeypatch.setattr(direct, "read_telemetry", telemetry)
    try:
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        session.add(HeartbeatRun(id="collect", ticket_id=ticket.id, agent_id="inference", driver="test",
                                model="test", activation_phase="collect"))
        await session.commit()
        assert await direct.reconcile(session) == (1, 0)
        assert not (await session.scalars(select(AgentWakeupRequest))).all()
        run.cancel_requested_at = datetime.now(timezone.utc); await session.commit()
        async def forbidden(*args, **kwargs): pytest.fail("cancelled Run must not publish")
        monkeypatch.setattr(direct, "remote_call", forbidden)
        assert await direct.reconcile(session) == (0, 0)
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
async def test_checkpoint_handoff_is_shared_by_direct_transport(tmp_path, monkeypatch):
    from zevo.engine.run.scheduler.reconciler import finish_observed_workload
    from zevo.engine.run import adaptive_training
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    ticket.agent_id = "train"
    seen = []
    async def checkpoint(db, r, t, job, meta):
        seen.append(job.id); meta["checkpoint_yield"] = "checkpoint-1"
        return True
    monkeypatch.setattr(adaptive_training, "handle_checkpoint_yield", checkpoint)
    try:
        meta = {"submission_committed": True}
        assert await finish_observed_workload(session, owner, run, ticket, meta, "COMPLETED", set()) == 0
        assert seen == [owner.id] and meta["checkpoint_yield"] == "checkpoint-1"
        assert not (await session.scalars(select(AgentWakeupRequest))).all()
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
async def test_orphaned_control_activation_does_not_block_remote_completion(tmp_path):
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    try:
        hb = await session.get(HeartbeatRun, "submit-hb")
        hb.finished_at = None; hb.started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        await session.commit()
        assert not await direct.live_activation(session, ticket)
        assert hb.finished_at is not None and ticket.status == "running"
    finally:
        await session.close(); await engine.dispose()


def test_direct_preflight_catches_static_helper_failure_before_publication(tmp_path):
    from zevo.engine.run.direct_validation import validate_launch
    job = job_for(tmp_path)
    script = Path(job.script_path); script.write_text("#!/bin/bash\npython predict.py\n")
    (tmp_path / "predict.py").write_text("print(missing_variable)\n")
    result = validate_launch(SimpleNamespace(slurm_job=job), script.read_text(), "predict.py")
    assert "undefined" in result or "missing_variable" in result


def test_direct_launch_reuse_does_not_inject_scheduler_directives(tmp_path):
    from zevo.engine.run.inference_launch import snapshot, render
    from tests.test_inference_launch import input_for
    old = input_for(tmp_path); old.slurm_job = job_for(Path(old.work_dir))
    new = input_for(tmp_path, "infer-2", "/checkpoint")
    new.slurm_job = direct.contract_for(info=info_for(), ticket=SimpleNamespace(id="infer-2"),
        work_dir=new.work_dir, stage="inference", previous=None, retry=False)
    body = "#!/bin/bash\npython predict.py --model org/model\n"
    rendered = render(snapshot(old, body), new)
    assert "#SBATCH" not in rendered and "--model /checkpoint" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["instance", "cloud"])
async def test_direct_checkpoint_exit_hands_verified_state_to_orchestrator(tmp_path, monkeypatch, provider):
    from tests.test_checkpoint_evaluation import bundle
    from zevo.engine.run import remote_jobs
    engine, session, run, ticket, owner, inp = await database(tmp_path, provider)
    path, digest = bundle(tmp_path / "validation")
    key = "a" * 32
    request = {"request_id": key, "ticket_id": ticket.id, "step": 20, "planned_steps": 100,
        "contract_sha256": digest, "metrics_path": f"/remote/{ticket.id}/adaptive/validation-20.json",
        "decision_path": f"/remote/{ticket.id}/adaptive/decision-{key}.json",
        "checkpoint": f"/remote/{ticket.id}/adaptive/checkpoint-20", "world_size": 1,
        "measurements": [], "decisions": [], "direction": "max"}
    async def remote(info, manifest, action, **kwargs):
        return {"execution_id": manifest["execution_id"], "state": "COMPLETED", "exit_code": 0}
    async def telemetry(session, row, ticket, info, meta): meta["telemetry_drained"] = True
    async def route(*args): return info_for(provider)
    async def verify(*args, **kwargs): return {"ok": True, "stdout": "checkpoint-verified"}
    monkeypatch.setattr(direct, "remote_call", remote)
    monkeypatch.setattr(direct, "read_telemetry", telemetry)
    monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", route)
    monkeypatch.setattr(remote_jobs, "_ssh", verify)
    try:
        ticket.agent_id = "train"
        run.supervisor_ticket_id = "orch"
        run.lifecycle = {"adaptive_mid_training_evaluation": True,
            "adaptive_validation_contracts": {ticket.id: digest},
            "adaptive_validation_contract_paths": {ticket.id: str(path)}}
        session.add(Ticket(id="orch", run_id=run.id, agent_id="orchestrator", status="succeeded"))
        inp.slurm_job = direct.contract_for(info=info_for(provider), ticket=ticket,
            work_dir=str(tmp_path), stage="train", previous=None, retry=False)
        Path(inp.slurm_job.script_path).write_text("python train.py\n")
        (tmp_path / "train.py").write_text("print('checkpoint')\n")
        (tmp_path / "train_config.yaml").write_text("training: {}\n")
        await session.commit()
        remote_config = str(Path(inp.slurm_job.remote_work_dir).parent / "frozen_train_config.yaml")
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb",
                                  remote_train_config_path=remote_config)
        assert row.meta["training_config_remote_path"] == remote_config
        assert row.meta["direct_manifest"]["files"][remote_config] == row.meta["training_config_sha256"]
        session.add(ExecutionEvent(ticket_id=ticket.id, heartbeat_id="submit-hb", attempt_id=row.id,
            event_type="progress", phase="adaptive_checkpoint", current_step=20,
            extras={"execution_purpose": "workload", "execution_id": row.id,
                    "adaptive_checkpoint_request": request}))
        await session.commit()
        assert await direct.reconcile(session) == (1, 0)
        assert ticket.status == "waiting_external" and owner.released_at is None
        assert row.meta["checkpoint_yield"] == key
        assert run.lifecycle["adaptive_checkpoints"][key]["job_row_id"] == row.id
        wakes = (await session.scalars(select(AgentWakeupRequest))).all()
        assert len(wakes) == 1 and wakes[0].agent_id == "orchestrator"
        assert await direct.reconcile(session) == (0, 0)
        assert len((await session.scalars(select(AgentWakeupRequest))).all()) == 1
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts,allowed", [(0, True), (19, True), (20, False)])
async def test_direct_failure_uses_shared_twenty_repair_limit(tmp_path, monkeypatch, attempts, allowed):
    from zevo.engine.run.runner import _activate_failed_workload_repair
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    async def remote(*args, **kwargs): return {}
    monkeypatch.setattr(direct, "remote_call", remote)
    try:
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        row.meta = {**row.meta, "scheduler_state": "FAILED", "scheduler_reason": "workload exit 1"}
        ticket.repair_attempts = attempts
        await session.commit()
        assert await _activate_failed_workload_repair(session, ticket) is allowed
        assert ticket.repair_attempts == attempts + int(allowed)
        assert await _activate_failed_workload_repair(session, ticket) is allowed
        assert ticket.repair_attempts == attempts + int(allowed)
        assert owner.released_at is None
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
async def test_unchanged_failed_direct_workload_cannot_relaunch(tmp_path, monkeypatch):
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    calls = []
    async def remote(*args, **kwargs): calls.append("publish"); return {}
    monkeypatch.setattr(direct, "remote_call", remote)
    try:
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        row.meta = {**row.meta, "scheduler_state": "FAILED"}
        row.released_at = datetime.now(timezone.utc)
        await session.commit()
        inp.slurm_job = direct.contract_for(info=info_for("cloud"), ticket=ticket,
            work_dir=str(tmp_path), stage="inference", previous=row, retry=True)
        with pytest.raises(ValueError, match="unchanged failed workload"):
            await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        assert calls == ["publish"]
        (tmp_path / "predict.py").write_text("print('repaired')\n")
        config = tmp_path / "suite" / "second" / "selected.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("max_tokens: 32\n")
        repaired = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb",
                                       prepared_config_paths=(str(config),))
        assert repaired.id != row.id and repaired.meta["execution_attempt"] == 2
        assert repaired.meta["retry_of_bookkeeping_row_id"] == row.id
        remote_config = str(Path(inp.slurm_job.remote_work_dir) / "suite/second/selected.yaml")
        assert repaired.meta["direct_manifest"]["files"][remote_config] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert owner.released_at is None
    finally:
        await session.close(); await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_ok", [False, True])
async def test_instruction_restart_waits_for_confirmed_direct_cleanup(tmp_path, monkeypatch, cleanup_ok):
    from zevo.engine.run import remote_jobs
    from zevo.engine.run.runner import _slurm_stage_job_contract
    engine, session, run, ticket, owner, inp = await database(tmp_path)
    actions = []
    async def remote(info, manifest, action, **kwargs): actions.append(action); return {}
    async def route(*args): return info_for("cloud")
    async def cleanup(*args, **kwargs): return {"ok": cleanup_ok, "error": ""}
    monkeypatch.setattr(direct, "remote_call", remote)
    monkeypatch.setattr(remote_jobs, "_device_info_for_ticket", route)
    monkeypatch.setattr(remote_jobs, "_ssh", cleanup)
    try:
        row = await direct.submit(session, run=run, ticket=ticket, inp=inp, heartbeat_id="submit-hb")
        row.meta = {**row.meta, "scheduler_state": "RUNNING", "superseded_by_instruction_id": "restart-1"}
        await session.commit()
        result = await remote_jobs.cancel_ticket_remote_job(session, ticket, preserve_allocation=True)
        assert result["ok"] is cleanup_ok and actions == ["publish", "cancel"]
        assert (row.released_at is not None) is cleanup_ok
        assert row.meta["scheduler_state"] == ("CANCELLED" if cleanup_ok else "RUNNING")
        assert owner.released_at is None
        if cleanup_ok:
            contract = await _slurm_stage_job_contract(run=run, ticket=ticket, work_dir=str(tmp_path),
                filename="infer.sbatch", device_info_path=inp.device_info_path, session=session, stage="inference")
            assert contract.phase == "submit" and contract.attempt == 2
            assert contract.restart_instruction_id == "restart-1"
            assert contract.retry_of_bookkeeping_row_id == row.id
    finally:
        await session.close(); await engine.dispose()
