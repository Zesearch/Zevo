"""Finite Slurm work is committed only by the engine after validation."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.infrastructure import (
    InfrastructureDeviceInfo,
    InfrastructureResourcePlan,
    SlurmStageJobContract,
    slurm_lifecycle_prologue,
    slurm_runtime_prologue,
)
from zevo.db.models import Base, InfraInstance, Run, Ticket
from zevo.engine.run import remote_jobs
from zevo.engine.run.runner import (
    _slurm_watcher_job_is_current,
    _submit_validated_slurm_stage,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['repair', 'remeasure', 'checkpoint', 'collect_retry', 'walltime'])
async def test_engine_submits_exact_uploaded_script_after_validation(tmp_path, monkeypatch, mode) -> None:
    remeasure = mode != "repair"
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    script = tmp_path / "train.sbatch"
    script.write_text("#!/bin/bash\necho validated\n", encoding="utf-8")
    train_program = tmp_path / "train.py"
    train_program.write_text("print('validated')\n", encoding="utf-8")
    device = InfrastructureDeviceInfo(
        run_id="run-1",
        ticket_id="infra-1",
        purpose="train",
        provider="cluster",
        auto_release=False,
        host="login.example",
        ssh={"host": "login.example", "port": 22, "user": "user", "key_path": "/key"},
        cluster={
            "requested_gpus": 4,
            "nodes": 1,
            "workdir": "/remote/run-1",
            "hf_cache": "/remote/cache",
            "gpu_constraints": {
                "min_gpus_per_job": 1,
                "allocation_step": 1,
                "gpus_per_node": 4,
                "source": "test scheduler geometry",
            },
        },
        cost={"dph_total": 0},
        resource_plan=InfrastructureResourcePlan(
            purpose="train",
            required_working_set_gib=40,
            host_memory_components_gib={"model_and_runtime": 40},
            host_memory_formula_gib=64,
            site_min_ram_gib=0,
            num_gpus=4,
            nodes=1,
            min_vram_gb=40,
            min_ram_gb=64,
            min_cpus=8,
            time_limit_hours=8,
            disk_gb=0,
            rationale="test plan",
        ),
        probe_source="none",
        probed_at="2026-09-19T00:00:00Z",
    )
    device_path = tmp_path / "device_info.json"
    device_path.write_text(device.model_dump_json(), encoding="utf-8")
    status_path = "/remote/run-1/train-1/.zevo-slurm-status"
    contract = SlurmStageJobContract(
        enabled=True,
        stage="train",
        estimated_gpus=2,
        resource_plan_source="test",
        resource_plan_rationale="test tier",
        script_path=str(script),
        remote_work_dir="/remote/run-1/train-1",
        remote_script_path="/remote/run-1/train-1/train.sbatch",
        job_name="zevo-train-1",
        status_path=status_path,
        stdout_path="/remote/run-1/train-1/slurm-%j.out",
        stderr_path="/remote/run-1/train-1/slurm-%j.err",
        lifecycle_prologue=slurm_lifecycle_prologue(status_path),
        runtime_prologue=slurm_runtime_prologue(),
        num_gpus=4,
    )
    commands: list[str] = []

    async def fake_ssh(_info, command, **_kwargs):
        commands.append(command)
        job_id = "98765" if len(commands) == 1 else "98766"
        return {
            "ok": True, "exit_code": 0,
            "stdout": f"{job_id};cluster\n", "error": "",
        }

    monkeypatch.setattr(remote_jobs, "_ssh", fake_ssh)
    async with Session() as session:
        run = Run(id="run-1", task_name="task", status="running", metric="accuracy")
        ticket = Ticket(
            id="train-1", run_id="run-1", agent_id="inference" if mode == "collect_retry" else "train", status="running",
            payload={}, inputs={}, customization={},
        )
        session.add_all([run, ticket])
        await session.commit()
        row = await _submit_validated_slurm_stage(
            session,
            run=run,
            ticket=ticket,
            inp=type("StageInput", (), {
                "slurm_job": contract,
                "device_info_path": str(device_path),
            })(),
            heartbeat_id="heartbeat-1",
        )
        assert row.instance_id == "98765"
        assert row.meta["submission_heartbeat_id"] == "heartbeat-1"
        assert row.meta["remote_script"] == contract.remote_script_path
        assert "submission_committed" not in row.meta
        assert commands and "sha256sum" in commands[0]
        assert "sbatch --parsable -- /remote/run-1/.zevo-allocations/heartbeat-1/controller.sbatch" in commands[0]

        row.status = "released"
        row.released_at = datetime.now(timezone.utc)
        row.meta = {**row.meta, "scheduler_state": "COMPLETED" if remeasure else "FAILED"}
        await session.flush()
        retry = contract.model_copy(update={
            "attempt": 2,
            "retry_of_bookkeeping_row_id": row.id,
            "retry_of_job_id": row.instance_id,
        })
        retry_input = type("StageInput", (), {
            "slurm_job": retry,
            "device_info_path": str(device_path),
        })()
        with pytest.raises(
            ValueError, match="requires the prior workload to be terminal" if remeasure else "only after its generated stage implementation changes",
        ):
            await _submit_validated_slurm_stage(
                session, run=run, ticket=ticket, inp=retry_input,
                heartbeat_id="heartbeat-2",
            )

        if mode == "collect_retry":
            row.meta = {**row.meta, "inference_execution_retry": {"execution_attempt": 1, "reason": "prediction output lost after job completed"}}
        elif mode == "walltime":
            row.meta = {**row.meta, "training_continuation": {
                "source_resource_id": row.id, "source_job_id": row.instance_id,
                "ticket_id": ticket.id, "verified": True,
                "config_sha256": "a" * 64, "completed_steps": 15, "planned_steps": 450,
            }}
        elif mode == "checkpoint":
            row.meta = {**row.meta, "checkpoint_yield": "request", "checkpoint_resume": "request"}
            run.lifecycle = {"adaptive_checkpoints": {"request": {
                "request": {"ticket_id": ticket.id, "step": 20, "planned_steps": 100},
                "job_row_id": row.id, "resume_queued": True, "metrics_delivered": True}},
                "adaptive_training_requests": {"request": {"status": "decided"}}}
        elif remeasure:
            # A deliberate budget replay can reuse a successful implementation;
            # it must be authorized by the engine's exact pending attempt.
            run.lifecycle = {'token_budget_remeasurements': [{
                'id': 'revision', 'status': 'running', 'reuse_tickets': True,
                'cohort': [{'inference_id': ticket.id, 'inference_attempt': {
                    'revision': 'revision', 'previous_resource_id': row.id,
                }}],
            }]}
        else:
            train_program.write_text("print('repaired')\n", encoding="utf-8")
        replacement = await _submit_validated_slurm_stage(
            session, run=run, ticket=ticket, inp=retry_input,
            heartbeat_id="heartbeat-2",
        )
        assert replacement.instance_id == "98766"
        assert replacement.meta["execution_attempt"] == 2
        assert replacement.meta["retry_of_bookkeeping_row_id"] == row.id
        assert replacement.meta["retry_of_job_id"] == "98765"
        assert len(commands) == 2

    await engine.dispose()


@pytest.mark.asyncio
async def test_old_watcher_job_cannot_collect_a_newer_running_job() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        session.add(Run(id="run-1", task_name="task", status="running", metric="accuracy"))
        ticket = Ticket(
            id="infer-1", run_id="run-1", agent_id="inference", status="queued",
            payload={}, inputs={}, customization={},
        )
        session.add(ticket)
        session.add(InfraInstance(
            id="old-job", provider="cluster", instance_id="100",
            run_id="run-1", ticket_id="infer-1", status="released",
            released_at=datetime.now(timezone.utc),
            meta={
                "resource_request": True,
                "submission_committed": True,
                "scheduler_state": "COMPLETED",
            },
        ))
        await session.flush()
        session.add(InfraInstance(
            id="new-job", provider="cluster", instance_id="200",
            run_id="run-1", ticket_id="infer-1", status="ready",
            meta={
                "resource_request": True,
                "submission_committed": True,
                "scheduler_state": "RUNNING",
            },
        ))
        await session.commit()

        assert not await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "100", "scheduler_state": "COMPLETED"},
        )
        assert not await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "200", "scheduler_state": "RUNNING"},
        )
        newer = await session.get(InfraInstance, "new-job")
        # Workload is done while the per-run allocation is still retained.
        newer.meta = {**newer.meta, "scheduler_state": "COMPLETED"}
        await session.commit()
        assert newer.released_at is None
        assert await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "200", "resource_row_id": "new-job"},
        )
        assert not await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "200", "resource_row_id": "old-job"},
        )
        newer.meta = {**newer.meta, "superseded_by_instruction_id": "next"}
        await session.commit()
        assert not await _slurm_watcher_job_is_current(session, ticket, {"job_id": "200"})
        newer.meta = {k: v for k, v in newer.meta.items() if k != "superseded_by_instruction_id"}
        newer.status = "released"
        newer.released_at = datetime.now(timezone.utc)
        await session.commit()
        assert await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "200", "scheduler_state": "COMPLETED"},
        )

    await engine.dispose()

@pytest.mark.parametrize('state,allowed', [
    ('PENDING', True), ('RUNNING', True), ('CONFIGURING', True), ('COMPLETING', True),
    ('COMPLETED', False), ('FAILED', False), ('CANCELLED by 123', False), ('', False),
])
def test_premature_collect_waits_only_for_live_workload(state, allowed):
    from types import SimpleNamespace
    from zevo.engine.run.runner import _deferred_collect_problem
    contract = SimpleNamespace(phase='collect', bookkeeping_row_id='row', job_id='123')
    row = SimpleNamespace(id='row', instance_id='123', meta={'scheduler_state': state})
    assert (_deferred_collect_problem(contract, row) is None) is allowed
    row.instance_id = 'another-job'
    assert _deferred_collect_problem(contract, row) is not None

@pytest.mark.parametrize('state', ['RUNNING', 'COMPLETED'])
def test_deferred_collect_rearms_watcher_without_rewriting_submission(state):
    from types import SimpleNamespace
    from zevo.engine.run.runner import _return_slurm_collect_to_watcher
    original = dict(scheduler_state=state, submission_committed=True,
                    submission_heartbeat_id='submit-hb', submission_committed_at='original-time',
                    log_path='/original/out', controller_request_id='submit-hb')
    row = SimpleNamespace(meta={**original, 'collect_wakeup_queued': True})
    # COMPLETED covers the job finishing between result validation and publication.
    assert _return_slurm_collect_to_watcher(row, heartbeat_id="collect-hb") == state
    assert row.meta == {**original, "watcher_handoff_heartbeat_id": "collect-hb"}


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['deadlock', 'persistent', 'after_publish', 'other_sql'])
async def test_submission_savepoint_recovers_without_replaying_remote_work(monkeypatch, fault):
    from sqlalchemy import event, select
    from sqlalchemy.exc import DBAPIError
    from zevo.engine.run import runner

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    calls, published, failures = [], [], []

    class DatabaseFault(Exception):
        sqlstate = '40P01' if fault != 'other_sql' else '23505'

    def fail_flush(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith('UPDATE infra_instances') and (not failures or fault == 'persistent'):
            failures.append(statement)
            raise DBAPIError(statement, parameters, DatabaseFault('injected failure'))

    async def submit_once(session, *, run, ticket, publication, **kwargs):
        calls.append(1)
        owner = await session.get(InfraInstance, 'expired-owner')
        assert owner.status == 'ready'
        if fault == 'after_publish':
            publication['attempted'] = True
            published.append('job')
        owner.status = 'released'
        await session.flush()  # failed ORM flush really poisons the savepoint
        publication['attempted'] = True
        published.append('job')
        return owner

    monkeypatch.setattr(runner, '_submit_validated_slurm_stage_once', submit_once)
    async with Session() as session:
        run = Run(id='expiry-run', task_name='task', status='running', metric='accuracy')
        ticket = Ticket(id='expiry-ticket', run_id=run.id, agent_id='inference', status='running')
        session.add_all([run, ticket, InfraInstance(
            id='expired-owner', run_id=run.id, ticket_id=ticket.id,
            instance_id='old-job', provider='cluster', status='ready',
        )])
        await session.commit()
        ticket.summary = 'agent already prepared the script'
        event.listen(engine.sync_engine, 'before_cursor_execute', fail_flush)
        kwargs = dict(run=run, ticket=ticket, inp=None, heartbeat_id='same-activation')
        if fault == 'deadlock':
            row = await runner._submit_validated_slurm_stage(session, **kwargs)
            assert row.status == 'released'
            assert len(calls) == 2
            assert published == ['job']
        else:
            with pytest.raises(DBAPIError):
                await runner._submit_validated_slurm_stage(session, **kwargs)
            assert len(calls) == (3 if fault == 'persistent' else 1)
            assert published == (['job'] if fault == 'after_publish' else [])
        # Caller can finish the heartbeat instead of hitting PendingRollbackError.
        await session.refresh(ticket)
        assert ticket.summary == 'agent already prepared the script'
        ticket.status = 'waiting_external' if fault == 'deadlock' else 'failed'
        await session.commit()
        assert await session.scalar(select(Ticket.status).where(Ticket.id == ticket.id)) == ticket.status
    await engine.dispose()
