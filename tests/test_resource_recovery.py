from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from tests.test_cloud_resource_recovery import setup, tick
from tests.test_direct_jobs import info_for
from tests.test_ticket_repair_state import session
from zevo.db import Run, Ticket, InfraInstance, WorkProduct, AgentWakeupRequest, RunInstruction, HeartbeatRun, ExecutionEvent, TicketMessage
from zevo.engine.run import resource_recovery as recovery
from zevo.engine.run.runner import _activate_failed_workload_repair


@pytest_asyncio.fixture
async def walltime(setup, tmp_path, monkeypatch):
    from tests.test_ssh_provider_contract import _device
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo
    from zevo.engine.run import remote_jobs, resource_cleanup, gpu_controller
    from zevo.engine.run.training_continuation import queue_continuation
    s = setup
    s.run.gpu_provider = s.owner.provider = s.job.provider = 'cluster'
    s.ticket.agent_id = 'train'
    s.ticket.iteration = 2
    s.ticket.repair_attempts = 20
    s.owner.ticket_id = s.ticket.id
    device = tmp_path / 'cluster.json'
    device.write_text(InfrastructureDeviceInfo.model_validate(_device('cluster')).model_dump_json())
    s.ticket.inputs = {'device_info': {'path': str(device), 'artifact_role': 'device_info'}}
    s.job.meta = {**s.job.meta, 'monitor_terminal': True, 'scheduler_state': 'COMPLETED'}
    proof = dict(verified=True, reason='walltime', ticket_id=s.ticket.id,
                 source_resource_id=s.job.id, source_job_id=s.job.instance_id,
                 completed_steps=1928, planned_steps=2447)
    hb = HeartbeatRun(ticket_id=s.ticket.id, agent_id='train', driver='test', model='test')
    s.db.add(hb)
    await queue_continuation(s.db, s.ticket, hb, s.job, proof)
    await s.db.commit()
    s.stop = AsyncMock(return_value={'attempted': False})
    s.state = AsyncMock(return_value='RUNNING')
    s.release = AsyncMock(return_value=False)
    monkeypatch.setattr(remote_jobs, 'cancel_ticket_remote_job', s.stop)
    monkeypatch.setattr(gpu_controller, '_retained_allocation_state', s.state)
    monkeypatch.setattr(resource_cleanup, 'release_cluster', s.release)
    s.proof, s.hb = proof, hb
    return s


@pytest.mark.asyncio
async def test_walltime_releases_exact_owner_before_normal_continuation(walltime):
    from zevo.engine.run.training_continuation import queue_continuation
    s = walltime
    assert s.ticket.status == 'waiting_external' and s.ticket.repair_route == ''
    assert s.hb.activation_phase == 'continue' and s.hb.finished_at
    assert await recovery.reconcile(s.db) == 0
    assert s.release.call_args.args[1] is s.owner
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    # Another scheduler pass cannot wake the Agent or reuse the retiring owner.
    assert await recovery.reconcile(s.db) == 0
    s.release.return_value = True
    assert await recovery.reconcile(s.db) == 1
    assert s.owner.released_at and s.owner.release_reason == 'Confirmed release for walltime continuation'
    assert s.ticket.status == 'queued' and s.ticket.repair_route == ''
    assert s.ticket.repair_attempts == 20 and s.ticket.iteration == 2
    assert 'execution_repair' not in s.job.meta
    assert not (await s.db.scalars(select(TicketMessage))).all()
    wake = (await s.db.scalars(select(AgentWakeupRequest))).one()
    assert wake.source == 'training_continuation'
    assert wake.payload['continuation_resource_id'] == s.job.id
    # Even after the wake is consumed, redelivery cannot enqueue it twice.
    wake.status = 'completed'
    await s.db.commit()
    await queue_continuation(s.db, s.ticket, s.hb, s.job, s.proof)
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('barrier', ['pause', 'cancel', 'live_agent', 'other_workload'])
async def test_walltime_handoff_respects_control_and_other_work(walltime, barrier):
    s = walltime
    if barrier == 'pause':
        s.db.add(RunInstruction(run_id=s.run.id, status='needs_input', body='Pause'))
    elif barrier == 'cancel':
        s.run.cancel_requested_at = datetime.now(timezone.utc)
    elif barrier == 'live_agent':
        s.db.add(HeartbeatRun(ticket_id=s.ticket.id, agent_id='train', driver='test', model='test'))
    else:
        s.db.add_all([Ticket(id='other', run_id=s.run.id, agent_id='evaluation', status='running'),
            InfraInstance(id='other-job', run_id=s.run.id, ticket_id='other', provider='cluster',
                          instance_id='456', meta={'allocation_owner_row_id': s.owner.id})])
    await s.db.commit()
    s.release.return_value = True
    assert await recovery.reconcile(s.db) == 0
    s.stop.assert_not_awaited()
    s.release.assert_not_awaited()
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
async def test_walltime_handoff_rejects_changed_execution(walltime):
    s = walltime
    event = s.run.lifecycle[recovery.KEY][s.ticket.id]
    recovery.save(s.run, s.ticket.id, {**event, 'execution_id': 'different-execution'})
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    s.release.assert_not_awaited()
    assert s.ticket.status == 'awaiting_input'
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['slurm_watcher', 'reconciler', 'training_continuation'])
async def test_old_wake_cannot_activate_train_during_walltime_handoff(walltime, source):
    from zevo.engine.run.runner import run_ticket
    s = walltime
    before = set((await s.db.scalars(select(HeartbeatRun.id))).all())
    assert await run_ticket(s.db, ticket_id=s.ticket.id, activation_source=source,
                            activation_payload={'continuation_resource_id': s.job.id}) is s.ticket
    assert s.ticket.status == 'waiting_external'
    assert set((await s.db.scalars(select(HeartbeatRun.id))).all()) == before


@pytest.mark.asyncio
async def test_stale_continuation_wake_cannot_resume_another_execution(walltime):
    from zevo.engine.run.runner import run_ticket
    s = walltime
    s.release.return_value = True
    assert await recovery.reconcile(s.db) == 1
    before = set((await s.db.scalars(select(HeartbeatRun.id))).all())
    assert await run_ticket(s.db, ticket_id=s.ticket.id, activation_source='training_continuation',
                            activation_payload={'continuation_resource_id': 'older-execution'}) is s.ticket
    assert s.ticket.status == 'queued'
    assert set((await s.db.scalars(select(HeartbeatRun.id))).all()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['optimization', 'held_out_test'])
async def test_cloud_recovery_rebinds_original_ticket_once(setup, tmp_path, lane):
    s = setup
    s.ticket.lane = lane
    s.ticket.iteration = 2
    s.ticket.repair_attempts = 3
    s.provider.state = 'terminated'
    await tick(s)
    await recovery.reconcile(s.db)
    event = s.run.lifecycle[recovery.KEY][s.ticket.id]
    infra = await s.db.get(Ticket, event['infra_ticket_id'])
    infra.status = 'succeeded'
    path = tmp_path / 'replacement.json'
    path.write_text(info_for('cloud').model_copy(update={'ticket_id': infra.id, 'instance_id': 'new-host'}).model_dump_json())
    s.db.add(WorkProduct(id='new-product', ticket_id=infra.id, role='device_info', path=str(path)))
    s.db.add(InfraInstance(id='new-owner', run_id=s.run.id, ticket_id=infra.id, provider='cloud', instance_id='new-host', status='ready'))
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 1
    assert s.ticket.status == 'repairing' and s.ticket.repair_route == 'self'
    assert s.ticket.iteration == 2 and s.ticket.lane == lane and s.ticket.repair_attempts == 4
    assert s.ticket.inputs['device_info']['work_product_id'] == 'new-product'
    assert s.job.meta['execution_repair']['action'] == 'resubmit'
    wakes = (await s.db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.ticket_id == s.ticket.id))).all()
    assert len(wakes) == 1
    assert await recovery.reconcile(s.db) == 0
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 2
    assert not (await s.db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.agent_id == 'orchestrator'))).all()


@pytest.mark.asyncio
async def test_pause_and_live_activation_prevent_replacement(setup):
    s = setup
    s.provider.state = 'terminated'
    await tick(s)
    gate = RunInstruction(run_id=s.run.id, status='needs_input', body='Pause')
    s.db.add(gate)
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    await s.db.delete(gate)
    live = HeartbeatRun(ticket_id=s.ticket.id, agent_id='inference', driver='mock', model='test')
    s.db.add(live)
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    live.finished_at = datetime.now(timezone.utc)
    await s.db.commit()
    await recovery.reconcile(s.db)
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
async def test_missing_checkpoint_does_not_resume_or_rent_twice(setup, tmp_path):
    s = setup
    s.ticket.agent_id = 'train'
    s.provider.state = 'terminated'
    await tick(s)
    await recovery.reconcile(s.db)
    infra = await s.db.get(Ticket, s.run.lifecycle[recovery.KEY][s.ticket.id]['infra_ticket_id'])
    infra.status = 'succeeded'
    path = tmp_path / 'replacement.json'
    path.write_text(info_for('cloud').model_copy(update={'ticket_id': infra.id, 'instance_id': 'new-host'}).model_dump_json())
    s.db.add(WorkProduct(id='new-product', ticket_id=infra.id, role='device_info', path=str(path)))
    s.db.add(InfraInstance(id='new-owner', run_id=s.run.id, provider='cloud', instance_id='new-host', status='ready'))
    s.db.add(ExecutionEvent(ticket_id=s.ticket.id, heartbeat_id='submit-hb', attempt_id='step', event_type='progress', phase='training', current_step=735, total_steps=2447))
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    await s.db.refresh(s.ticket)
    assert s.ticket.status == 'awaiting_input' and 'checkpoint' in s.ticket.summary
    assert s.ticket.inputs['device_info']['path'] != str(path)
    assert not (await s.db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.ticket_id == s.ticket.id))).all()
    # Failed state verification must not create another allocation on restart.
    await recovery.reconcile(s.db)
    assert len((await s.db.scalars(select(Ticket).where(Ticket.agent_id == 'infrastructure'))).all()) == 1


@pytest.mark.asyncio
async def test_release_must_be_confirmed_before_new_infrastructure(setup, monkeypatch):
    from zevo.engine.run import remote_jobs, resource_cleanup
    s = setup
    s.ticket.repair_attempts = 1
    s.job.meta = {**s.job.meta, 'monitor_terminal': True, 'scheduler_state': 'FAILED'}
    await recovery.request_replacement(s.db, s.ticket, s.job, reason='Verified unusable device')
    await s.db.commit()
    monkeypatch.setattr(remote_jobs, 'cancel_ticket_remote_job', AsyncMock(return_value={'ok': True, 'attempted': True}))
    release = AsyncMock(return_value=False)
    monkeypatch.setattr(resource_cleanup, 'release_cloud', release)
    await recovery.reconcile(s.db)
    assert s.owner.released_at is None
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    release.return_value = True
    await recovery.reconcile(s.db)
    assert s.owner.released_at is not None
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1
    assert release.call_args.kwargs['session'] is s.db


@pytest.mark.asyncio
@pytest.mark.parametrize('state,code,allowed', [('TIMEOUT','0:0',True),('PREEMPTED','0:0',True),('NODE_FAIL','0:0',True),('FAILED','75:0',True),('FAILED','1:0',False),('CANCELLED','0:15',False)])
async def test_heldout_deterministic_evaluation_retries_only_verified_interruptions(session, state, code, allowed):
    session.add(Run(id='r', metric='score', status='running'))
    ticket = Ticket(id='evaluation', run_id='r', agent_id='evaluation', lane='held_out_test', status='queued')
    session.add_all([ticket, HeartbeatRun(id='hb',ticket_id=ticket.id,agent_id='evaluation',driver='evaluation_runner',model='system',finished_at=datetime.now(timezone.utc)),
        InfraInstance(id='job',run_id='r',ticket_id=ticket.id,provider='cluster',instance_id='123',status='released',meta={'resource_request':True,'scheduler_state':state,'scheduler_exit_code':code,'execution_attempt':1})])
    await session.commit()
    normal_expiry = state == 'TIMEOUT' or code == '75:0'
    assert await _activate_failed_workload_repair(session,ticket) is (allowed and not normal_expiry)
    assert ticket.repair_attempts == int(allowed and not normal_expiry)
    assert ticket.repair_route == ('' if normal_expiry else 'self' if allowed else 'terminal')
    assert await _activate_failed_workload_repair(session,ticket) is (allowed and not normal_expiry)
    assert ticket.repair_attempts == int(allowed and not normal_expiry)


@pytest.mark.asyncio
async def test_recovered_binding_propagates_without_rewriting_historical_ticket(setup, tmp_path):
    s = setup
    historical = dict(s.ticket.inputs)
    old = info_for('cloud')
    path = tmp_path / 'new-device.json'
    path.write_text(old.model_copy(update={'ticket_id':'replacement-infra','instance_id':'new-host'}).model_dump_json())
    s.db.add_all([Ticket(id='replacement-infra',run_id=s.run.id,agent_id='infrastructure',status='succeeded'),
        WorkProduct(id='new-route',ticket_id='replacement-infra',role='device_info',path=str(path))])
    s.run.lifecycle={'device_replacements':{old.ticket_id:'new-route'}}
    await s.db.commit()
    effective = await recovery.current_device_binding(s.db,s.run,s.ticket.inputs)
    assert effective['device_info']['work_product_id']=='new-route'
    assert s.ticket.inputs==historical


@pytest.mark.asyncio
async def test_shared_allocation_waits_for_other_live_workload(setup, monkeypatch):
    from zevo.engine.run import remote_jobs, resource_cleanup
    s = setup
    s.ticket.repair_attempts=1
    s.db.add_all([Ticket(id='other',run_id=s.run.id,agent_id='evaluation',status='running'),
        InfraInstance(id='other-job',run_id=s.run.id,ticket_id='other',provider='cloud',instance_id='other-process',status='running',meta={'allocation_owner_row_id':s.owner.id,'scheduler_state':'RUNNING'})])
    await recovery.request_replacement(s.db,s.ticket,s.job,reason='Verified defective compute')
    await s.db.commit()
    stop=AsyncMock()
    release=AsyncMock()
    monkeypatch.setattr(remote_jobs,'cancel_ticket_remote_job',stop)
    monkeypatch.setattr(resource_cleanup,'release_cloud',release)
    await recovery.reconcile(s.db)
    stop.assert_not_awaited()
    release.assert_not_awaited()
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
async def test_instance_without_configured_replacement_waits_without_renting(setup, tmp_path, monkeypatch):
    from zevo.engine.run import remote_jobs
    s=setup
    s.run.gpu_provider='instance'
    s.owner.provider='instance'
    device=tmp_path/'fixed-instance.json'
    device.write_text(info_for('instance').model_dump_json())
    s.ticket.inputs={'device_info':{'path':str(device),'artifact_role':'device_info'}}
    s.ticket.repair_attempts=1
    await recovery.request_replacement(s.db,s.ticket,s.job,reason='Verified unusable fixed device')
    await s.db.commit()
    monkeypatch.setattr(remote_jobs,'cancel_ticket_remote_job',AsyncMock(return_value={'attempted':True,'ok':True}))
    await recovery.reconcile(s.db)
    assert s.ticket.status=='awaiting_input'
    assert 'configured' in s.ticket.summary
    assert s.owner.released_at is None
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
async def test_cluster_release_is_confirmed_before_retry_and_keeps_ticket(setup, tmp_path, monkeypatch):
    from tests.test_ssh_provider_contract import _device
    from zevo.contracts.infrastructure import InfrastructureDeviceInfo
    from zevo.engine.run import remote_jobs, resource_cleanup, gpu_controller
    s=setup
    s.run.gpu_provider='cluster'
    s.owner.provider='cluster'
    s.owner.ticket_id=s.ticket.id
    device=tmp_path/'cluster.json'
    device.write_text(InfrastructureDeviceInfo.model_validate(_device('cluster')).model_dump_json())
    s.ticket.inputs={'device_info':{'path':str(device),'artifact_role':'device_info'}}
    s.ticket.repair_attempts=2
    s.job.provider='cluster'
    s.job.meta={**s.job.meta,'monitor_terminal':True,'scheduler_state':'FAILED'}
    await recovery.request_replacement(s.db,s.ticket,s.job,reason='Replace failed GPU')
    await s.db.commit()
    monkeypatch.setattr(remote_jobs,'cancel_ticket_remote_job',AsyncMock(return_value={'attempted':False}))
    state=AsyncMock(return_value='RUNNING')
    release=AsyncMock(return_value=False)
    monkeypatch.setattr(gpu_controller,'_retained_allocation_state',state)
    monkeypatch.setattr(resource_cleanup,'release_cluster',release)
    assert await recovery.reconcile(s.db)==0
    assert s.owner.released_at is None
    assert not (await s.db.scalars(select(AgentWakeupRequest))).all()
    state.return_value='CANCELLED'
    assert await recovery.reconcile(s.db)==1
    assert s.owner.released_at is not None and s.ticket.status=='repairing'
    assert s.ticket.repair_attempts==2
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all())==1


@pytest.mark.asyncio
async def test_replacement_cannot_reduce_resource_requirements(setup, tmp_path):
    s = setup
    s.provider.state = 'terminated'
    await tick(s)
    await recovery.reconcile(s.db)
    infra = await s.db.get(Ticket, s.run.lifecycle[recovery.KEY][s.ticket.id]['infra_ticket_id'])
    infra.status = 'succeeded'
    replacement = info_for('cloud').model_copy(update={'ticket_id': infra.id, 'instance_id': 'new-host'})
    replacement.resource_plan.min_cpus -= 1
    path = tmp_path / 'replacement.json'
    path.write_text(replacement.model_dump_json())
    s.db.add(WorkProduct(id='new-product', ticket_id=infra.id, role='device_info', path=str(path)))
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    await s.db.refresh(s.ticket)
    assert s.ticket.status == 'awaiting_input'
    assert 'resource requirements' in s.ticket.summary
    assert s.ticket.inputs['device_info']['path'] != str(path)
    assert not (await s.db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.ticket_id == s.ticket.id))).all()


@pytest.mark.asyncio
async def test_failed_replacement_preserves_accumulated_repair_budget(setup):
    s = setup
    s.provider.state = 'terminated'
    await tick(s)
    await recovery.reconcile(s.db)
    infra = await s.db.get(Ticket, s.run.lifecycle[recovery.KEY][s.ticket.id]['infra_ticket_id'])
    infra.status = 'failed'
    infra.repair_attempts = 20
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 0
    await s.db.refresh(s.ticket)
    assert s.ticket.status == 'awaiting_input' and s.ticket.repair_attempts == 20
    assert not (await s.db.scalars(select(AgentWakeupRequest).where(AgentWakeupRequest.ticket_id == s.ticket.id))).all()


@pytest.mark.asyncio
async def test_shared_replacement_inherits_highest_consumer_repair_count(setup):
    s = setup
    s.provider.state = 'terminated'
    await tick(s)
    other = Ticket(id='other-consumer', run_id=s.run.id, agent_id='evaluation', status='waiting_external',
        repair_route='self', repair_attempts=20, inputs=dict(s.ticket.inputs))
    s.db.add(other)
    await recovery.request_replacement(s.db, other, owner=s.owner, reason='Same confirmed lost allocation')
    await s.db.commit()
    await recovery.reconcile(s.db)
    infra_ids = {state['infra_ticket_id'] for state in s.run.lifecycle[recovery.KEY].values()}
    assert len(infra_ids) == 1
    infra = await s.db.get(Ticket, infra_ids.pop())
    assert infra.repair_attempts == 20
    assert len((await s.db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
async def test_instance_rebind_requires_live_leases_on_replacement_host(setup, tmp_path, monkeypatch):
    from zevo.db import GpuLease
    from zevo.engine.run import remote_jobs
    s = setup
    old = info_for('instance')
    s.run.gpu_provider = s.owner.provider = 'instance'
    device = tmp_path / 'fixed-instance.json'
    device.write_text(old.model_dump_json())
    s.ticket.inputs = {'device_info': {'path': str(device), 'artifact_role': 'device_info'}}
    replacement = old.model_copy(update={'ticket_id': 'new-infra', 'instance_id': 'new-host',
        'host': 'new.example.com', 'ssh': old.ssh.model_copy(update={'host': 'new.example.com'})})
    path = tmp_path / 'new-instance.json'
    path.write_text(replacement.model_dump_json())
    old_lease = GpuLease(jobid=old.instance_id, gpu_index=0, run_id=s.run.id, ticket_id=old.ticket_id)
    new_lease = GpuLease(jobid='wrong-host', gpu_index=0, run_id=s.run.id, ticket_id='new-infra')
    s.db.add_all([Ticket(id=old.ticket_id, run_id=s.run.id, agent_id='infrastructure', status='succeeded'),
        Ticket(id='new-infra', run_id=s.run.id, agent_id='infrastructure', status='succeeded'),
        WorkProduct(id='new-product', ticket_id='new-infra', role='device_info', path=str(path)), old_lease, new_lease])
    await recovery.request_replacement(s.db, s.ticket, s.job, reason='Verified unusable instance')
    await s.db.commit()
    monkeypatch.setattr(remote_jobs, 'cancel_ticket_remote_job', AsyncMock(return_value={'attempted': True, 'ok': True}))
    assert await recovery.reconcile(s.db) == 0
    await s.db.refresh(old_lease)
    assert old_lease.released_at is None
    await s.db.refresh(new_lease)
    new_lease.jobid = replacement.instance_id
    event = dict(s.run.lifecycle[recovery.KEY][s.ticket.id])
    event.pop('next_check_at', None)
    recovery.save(s.run, s.ticket.id, event)
    await s.db.commit()
    assert await recovery.reconcile(s.db) == 1
    assert old_lease.released_at is not None and new_lease.released_at is None
    assert s.ticket.inputs['device_info']['work_product_id'] == 'new-product'
    assert s.owner.released_at is not None
