from pathlib import Path

import pytest
from sqlalchemy import select

from tests.test_direct_jobs import database, info_for
from zevo.contracts.model_registry import RegisterTaskInput, RegisterResult
from zevo.db import AgentWakeupRequest
from zevo.engine.run import direct_jobs, runner


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['instance', 'cloud'])
async def test_registry_compute_submission_expiry_and_resume(tmp_path, monkeypatch, provider):
    engine, db, run, ticket, owner, original = await database(tmp_path, provider)
    try:
        ticket.agent_id = 'registry'
        info = info_for(provider)
        contract = direct_jobs.contract_for(info=info, ticket=ticket, work_dir=str(tmp_path),
            stage='registry', previous=None, retry=False)
        Path(contract.script_path).write_text('#!/bin/bash\npython register.py\n')
        (tmp_path / 'register.py').write_text('print("prepare owned output")\n')
        inp = RegisterTaskInput.model_construct(slurm_job=contract, device_info_path=original.device_info_path)
        result = RegisterResult(status='deferred', ticket_id=ticket.id, error_message='', notes='',
            registry_path='', register_script_path=str(tmp_path / 'register.py'), slurm_script_path=contract.script_path)
        assert runner._extract_summary_artifact_meta(result, inp=inp, agent_id='registry',
            work_dir=str(tmp_path), run_id=run.id)[0] == 'deferred'
        calls = []
        async def remote(info, manifest, action, **kwargs):
            calls.append(action)
            return {'execution_id': manifest['execution_id'], 'state': 'FAILED', 'exit_code': 75}
        async def telemetry(db, row, ticket, info, meta):
            meta['telemetry_drained'] = True
        monkeypatch.setattr(direct_jobs, 'remote_call', remote)
        monkeypatch.setattr(direct_jobs, 'read_telemetry', telemetry)
        row = await direct_jobs.submit(db, run=run, ticket=ticket, inp=inp, heartbeat_id='submit-hb')
        ticket.status = 'waiting_external'
        await db.commit()
        await direct_jobs.reconcile(db)
        assert not await runner._activate_failed_workload_repair(db, ticket)
        assert ticket.status == 'queued' and ticket.repair_attempts == 0
        assert owner.released_at is None
        wakes = (await db.scalars(select(AgentWakeupRequest))).all()
        assert any(wake.source == 'stage_continuation' for wake in wakes)
        retry = await runner._slurm_stage_job_contract(run=run, ticket=ticket, work_dir=str(tmp_path),
            filename='registry.sbatch', device_info_path=original.device_info_path, session=db, stage='registry')
        assert retry.phase == 'submit' and retry.attempt == 2
        assert retry.retry_of_bookkeeping_row_id == row.id
        inp.slurm_job = retry
        resumed = await direct_jobs.submit(db, run=run, ticket=ticket, inp=inp, heartbeat_id='continue-hb')
        assert resumed.id != row.id and resumed.meta['allocation_owner_row_id'] == owner.id
        assert ticket.repair_attempts == 0
    finally:
        await db.close()
        await engine.dispose()


def test_registry_gpu_cannot_report_success_during_preparation(tmp_path):
    from tests.test_direct_jobs import job_for
    job = job_for(tmp_path).model_copy(update={'stage': 'registry'})
    Path(job.script_path).write_text('#!/bin/bash\npython register.py\n')
    (tmp_path / 'register.py').write_text('print("prepare")\n')
    inp = RegisterTaskInput.model_construct(slurm_job=job)
    result = RegisterResult(status='succeeded', ticket_id='registry', error_message='', notes='',
        registry_path='', register_script_path='', slurm_script_path=job.script_path)
    assert runner._extract_summary_artifact_meta(result, inp=inp, agent_id='registry',
        work_dir=str(tmp_path), run_id='run')[0] == 'failed'
