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
async def test_engine_submits_exact_uploaded_script_after_validation(tmp_path, monkeypatch) -> None:
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
            id="train-1", run_id="run-1", agent_id="train", status="running",
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
        assert "sbatch --parsable -- /remote/run-1/train-1/train.sbatch" in commands[0]

        row.status = "released"
        row.released_at = datetime.now(timezone.utc)
        row.meta = {**row.meta, "scheduler_state": "FAILED"}
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
            ValueError, match="only after its generated stage implementation changes",
        ):
            await _submit_validated_slurm_stage(
                session, run=run, ticket=ticket, inp=retry_input,
                heartbeat_id="heartbeat-2",
            )

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
        newer.status = "released"
        newer.released_at = datetime.now(timezone.utc)
        newer.meta = {**newer.meta, "scheduler_state": "COMPLETED"}
        await session.commit()
        assert await _slurm_watcher_job_is_current(
            session, ticket, {"job_id": "200", "scheduler_state": "COMPLETED"},
        )

    await engine.dispose()
