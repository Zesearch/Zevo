from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import hashlib

import pytest
import pytest_asyncio
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts._base import AgentResult
from zevo.db import Base, Run, Ticket, HeartbeatRun, InfraInstance, RunInstruction, AgentWakeupRequest, WorkProduct
from zevo.engine.run.execution_repair import RepairTicketRequest, repair_ticket, release_reviewed_repairs
from zevo.engine.run.repair_progress import execution_retry_authorized, verify_repair_training_config
from zevo.api.routers.ui.heartbeats import _repair_sources


@pytest_asyncio.fixture
async def repair_db(monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    run = Run(id='r', metric='accuracy', status='running', supervisor_ticket_id='orch')
    supervisor = Ticket(id='orch', run_id='r', agent_id='orchestrator', status='running')
    ticket = Ticket(id='train', run_id='r', agent_id='train', iteration=2, status='failed',
                    repair_attempts=2, repair_route='orchestrator',
                    payload={'fixed': 'plan'}, inputs={'parent': 'original'})
    hb = HeartbeatRun(id='decision', ticket_id='orch', agent_id='orchestrator', driver='mock', model='test')
    execution = InfraInstance(id='job', run_id='r', ticket_id='train', provider='cluster',
        instance_id='123', status='released', meta={'resource_request': True,
            'execution_attempt': 2, 'scheduler_state': 'COMPLETED', 'scheduler_exit_code': '0:0'})
    product = WorkProduct(id='product', ticket_id='train', role='checkpoint', path='/durable/checkpoint-735')
    session.add_all([run, supervisor, ticket, hb, execution, product])
    await session.commit()
    from zevo.engine.cost import budget
    monkeypatch.setattr(budget, 'snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=False, over_time_limit=False)))
    yield session, ticket, execution
    await session.close()
    await engine.dispose()


def request(action='resubmit'):
    return RepairTicketRequest(action=action, execution_attempt=2,
        reason='Launcher corrected; the prior process completed without training',
        evidence=['attempt-2/workload.out: declined before first step; checkpoint-735 intact'])


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['collect', 'resubmit'])
async def test_review_repairs_same_ticket_preserving_state_and_budget(repair_db, action):
    db, ticket, execution = repair_db
    result = await repair_ticket(db, ticket, request(action))
    assert result['status'] == 'repairing' and ticket.repair_attempts == 3
    assert ticket.iteration == 2 and ticket.payload == {'fixed': 'plan'}
    assert ticket.inputs == {'parent': 'original'}
    assert (await db.get(WorkProduct, 'product')).path == '/durable/checkpoint-735'
    assert execution_retry_authorized(execution.meta) is (action == 'resubmit')
    assert not execution_retry_authorized({**execution.meta, 'execution_attempt': 3})
    assert not (await db.scalars(select(AgentWakeupRequest))).all()
    await complete_review(db)
    wakes = (await db.scalars(select(AgentWakeupRequest))).all()
    assert len(wakes) == 1 and wakes[0].ticket_id == ticket.id
    assert wakes[0].payload['repair_attempt'] == 3
    with pytest.raises(HTTPException):
        await repair_ticket(db, ticket, request(action))
    assert ticket.repair_attempts == 3  # repeated HTTP request cannot spend twice


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['limit', 'pause', 'cancel', 'terminal', 'held_out', 'active', 'stale', 'no_supervisor', 'budget'])
async def test_review_cannot_override_execution_boundaries(repair_db, monkeypatch, boundary):
    db, ticket, execution = repair_db
    run = await db.get(Run, 'r')
    if boundary == 'limit': ticket.repair_attempts = 20
    if boundary == 'pause': db.add(RunInstruction(run_id='r', body='Pause until explicitly resumed', status='needs_input'))
    if boundary == 'cancel': run.cancel_requested_at = datetime.now(timezone.utc)
    if boundary == 'terminal': ticket.repair_route = 'terminal'
    if boundary == 'held_out': ticket.lane = 'held_out_test'
    if boundary == 'active': execution.meta = {**execution.meta, 'scheduler_state': 'RUNNING'}
    if boundary == 'stale': execution.meta = {**execution.meta, 'execution_attempt': 3}
    if boundary == 'no_supervisor': (await db.get(HeartbeatRun, 'decision')).finished_at = datetime.now(timezone.utc)
    if boundary == 'budget':
        from zevo.engine.cost import budget
        monkeypatch.setattr(budget, 'snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=True, over_time_limit=False)))
    await db.commit()
    count = ticket.repair_attempts
    with pytest.raises(HTTPException):
        await repair_ticket(db, ticket, request())
    assert ticket.repair_attempts == count and ticket.status == 'failed'
    assert not (await db.scalars(select(AgentWakeupRequest))).all()


def test_execution_retry_requires_failure_and_same_work_order():
    args = dict(ticket_id='train', status='failed', error_message='no work', notes='', execution_retry_reason='run corrected launcher')
    assert AgentResult(**args).execution_retry_reason
    with pytest.raises(ValidationError):
        AgentResult(**{**args, 'status': 'succeeded', 'error_message': ''})
    with pytest.raises(ValidationError):
        AgentResult(**args, failure_recovery=dict(scope='resource_plan', reason='need new host',
                    evidence=['host lost'], next_action='replace allocation'))


def test_training_repair_cannot_change_or_lose_the_registered_config(tmp_path):
    config = tmp_path / 'train_config.yaml'
    config.write_text('frozen plan')
    previous = SimpleNamespace(meta={'execution_attempt': 2,
        'execution_repair': {'execution_attempt': 2, 'action': 'resubmit', 'reason': 'launcher fix'},
        'training_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()})
    verify_repair_training_config(previous, config)
    config.write_text('changed plan')
    with pytest.raises(ValueError, match='exact registered training configuration'):
        verify_repair_training_config(previous, config)
    previous.meta.pop('training_config_sha256')
    with pytest.raises(ValueError):
        verify_repair_training_config(previous, config)


@pytest.mark.asyncio
async def test_timeline_uses_historical_trigger_not_mutable_ticket_status(repair_db):
    db, ticket, _ = repair_db
    hb = await db.get(HeartbeatRun, 'decision')
    hb.resolved_config = {'orchestration_trigger': {'ticket_id': ticket.id, 'status': 'failed'}}
    ticket.status = 'succeeded'
    await db.commit()
    assert (await _repair_sources(db, [hb]))[hb.id] == {'ticket_id': ticket.id, 'agent_id': 'train'}
    hb.resolved_config = {'orchestration_trigger': {'ticket_id': ticket.id, 'status': 'succeeded'}}
    ticket.status = 'failed'
    await db.commit()
    assert await _repair_sources(db, [hb]) == {}
    hb.resolved_config = {}
    db.add(AgentWakeupRequest(agent_id='orchestrator', ticket_id='orch', heartbeat_run_id=hb.id,
                             source='handoff', status='completed',
                             payload={'optimization_completion_key': 'train:failed-hb:failed'}))
    await db.commit()
    assert (await _repair_sources(db, [hb]))[hb.id]['ticket_id'] == 'train'
    ticket.lane = 'held_out_test'
    await db.commit()
    assert await _repair_sources(db, [hb]) == {}


@pytest.mark.asyncio
async def test_preparation_repair_resolves_corrected_upstream_without_execution(repair_db, tmp_path):
    import json
    from zevo.db import TicketNotice
    from zevo.engine.run.input_validation import input_signature, NOTICE_CODE
    db, ticket, execution = repair_db
    await db.delete(execution)
    ticket.input_format = 'freeform'
    ticket.payload = {'request': 'Prepare the existing experiment', 'attachments': []}
    upstream = Ticket(id='upstream', run_id='r', agent_id='infrastructure', status='succeeded')
    path = tmp_path / 'device.json'
    path.write_text('{}')
    db.add_all([upstream, WorkProduct(id='corrected', ticket_id='upstream', role='device_info', path=str(path))])
    ticket.inputs = {'device_info': {'source_ticket_id':'upstream', 'artifact_role':'device_info', 'work_product_id':'missing'}}
    signature = input_signature(ticket)
    db.add(TicketNotice(ticket_id=ticket.id,code=NOTICE_CODE,severity='error',body=json.dumps({'input_signature':signature})))
    await db.commit()
    req = RepairTicketRequest(action='prepare',execution_attempt=0,input_signature=signature,
        input_products={'device_info':'corrected'},reason='Upstream republished the valid device artifact',evidence=['registered corrected product'])
    with pytest.raises(HTTPException):
        await repair_ticket(db,ticket,req.model_copy(update={'input_signature':'stale'}))
    assert ticket.repair_attempts == 2
    result = await repair_ticket(db,ticket,req)
    assert result['repair_attempts'] == 3 and ticket.iteration == 2
    assert ticket.inputs['device_info']['work_product_id'] == 'corrected'
    assert (await db.get(HeartbeatRun,'decision')).resolved_config['repair_target_ticket_ids'] == ['train']
    assert not (await db.scalars(select(InfraInstance))).all()


@pytest.mark.asyncio
async def test_replacement_review_defers_dispatch_until_resources_are_ready(repair_db):
    from zevo.engine.run.resource_recovery import KEY
    db,ticket,execution = repair_db
    result = await repair_ticket(db,ticket,request('replace_resource'))
    assert result['status'] == 'repairing'
    assert not (await db.get(Run,'r')).lifecycle.get(KEY)
    await complete_review(db)
    assert ticket.status == 'waiting_external'
    assert ticket.repair_attempts == 3 and execution_retry_authorized(execution.meta)
    assert (await db.get(Run,'r')).lifecycle[KEY][ticket.id]['phase'] == 'stopping'
    assert not (await db.scalars(select(AgentWakeupRequest))).all()


def test_replacement_proposal_requires_explicit_actionable_evidence():
    args=dict(ticket_id='train',status='failed',error_message='bad device',notes='',execution_retry_action='replace_resource')
    with pytest.raises(ValidationError): AgentResult(**args)
    with pytest.raises(ValidationError): AgentResult(**args,execution_retry_reason='need fresh allocation')
    result=AgentResult(**args,execution_retry_reason='need fresh allocation',failure_recovery={
        'scope':'current_task','reason':'registered GPU health check failed','evidence':['gpu-health.json: defective'],
        'next_action':'replace compute within the unchanged plan'})
    assert result.execution_retry_action == 'replace_resource'


@pytest.mark.asyncio
async def test_pure_repair_is_nested_but_mixed_decision_stays_on_timeline(repair_db):
    from zevo.api.routers.ui.heartbeats import _dto
    db, ticket, _ = repair_db
    await repair_ticket(db,ticket,request())
    h=await db.get(HeartbeatRun,'decision')
    assert not _dto(h,{'action':'wait'}).repair_only  # live decisions may still do more
    h.finished_at=datetime.now(timezone.utc)
    dto=_dto(h,{'action':'wait','summary':'Resubmit the existing stage'})
    assert dto.repair_only and dto.repair_target_ticket_ids == [ticket.id]
    assert not _dto(h,{'action':'continue','child_ticket_id':'next'}).repair_only
    h.resolved_config={**h.resolved_config,'orchestration_child_ticket_ids':['new-stage']}
    assert not _dto(h,{'action':'wait'}).repair_only
    h.resolved_config={'orchestration_trigger':{'ticket_id':ticket.id,'status':'failed'}}
    assert not _dto(h,{'action':'wait'}).repair_only  # no guess based on error prose/trigger


@pytest.mark.asyncio
async def test_stage_history_includes_explicit_reviews_only_within_run(repair_db, monkeypatch):
    from zevo.api.routers.ui import heartbeats
    db, ticket, execution = repair_db
    decision = await db.get(HeartbeatRun, 'decision')
    decision.resolved_config = {'repair_target_ticket_ids': [ticket.id]}
    db.add_all([Run(id='other-run', metric='accuracy', status='running'),
        Ticket(id='other-orch', run_id='other-run', agent_id='orchestrator', status='running'),
        HeartbeatRun(id='foreign-review', ticket_id='other-orch', agent_id='orchestrator', driver='mock', model='test',
            resolved_config={'repair_target_ticket_ids': [ticket.id]})])
    await db.commit()
    monkeypatch.setattr(heartbeats, 'is_trusted_ui_request', lambda request: False)
    async def history(**kwargs):
        return await heartbeats.list_heartbeats(SimpleNamespace(), ticket_id=ticket.id,
            include_repairs=True, limit=500, db=db, **kwargs)
    assert [h.id for h in await history()] == ['decision']
    assert await history(run_id='other-run') == []
    ticket.lane = 'held_out_test'
    await db.commit()
    assert await history() == []


async def complete_review(db):
    hb = await db.get(HeartbeatRun, 'decision')
    hb.finished_at = datetime.now(timezone.utc)
    supervisor = await db.get(Ticket, 'orch')
    supervisor.status = 'succeeded'
    await db.commit()
    await release_reviewed_repairs(db, supervisor, hb.id)


@pytest.mark.asyncio
async def test_repair_cannot_run_from_message_before_valid_review(repair_db):
    from zevo.engine.run.runner import run_ticket, _queue_ticket_repair
    from zevo.engine.run.wakeup import WakeupDeferred
    from zevo.engine.run.supervisor_handoff import waiting
    db, ticket, _ = repair_db
    await repair_ticket(db, ticket, request())
    await _queue_ticket_repair(db, ticket)
    assert not (await db.scalars(select(AgentWakeupRequest))).all()
    with pytest.raises(WakeupDeferred, match='supervisor_decision_pending'):
        await run_ticket(db, ticket_id=ticket.id, activation_source='on_demand')
    hb = await db.get(HeartbeatRun, 'decision')
    hb.finished_at = datetime.now(timezone.utc)
    supervisor = await db.get(Ticket, 'orch')
    supervisor.status = 'failed'
    await db.commit()
    await release_reviewed_repairs(db, supervisor, hb.id)
    assert await waiting(db, ticket)
    assert not (await db.scalars(select(AgentWakeupRequest))).all()
    await complete_review(db)
    assert not await waiting(db, ticket)
    await complete_review(db)
    assert len((await db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('blocker', ['', 'unknown', 'live', 'execution'])
async def test_infrastructure_provision_repairs_without_execution(repair_db, blocker):
    import json
    from zevo.db import TicketNotice
    db, ticket, execution = repair_db
    ticket.agent_id = 'infrastructure'
    if blocker != 'execution':
        await db.delete(execution)
    db.add(TicketNotice(ticket_id=ticket.id, code='cloud.acquisition_failed', severity='warning',
        body=json.dumps({'confirmed_not_created': blocker != 'unknown'})))
    if blocker == 'live':
        db.add(InfraInstance(id='live', run_id='r', ticket_id=ticket.id, provider='cloud', instance_id='i-1'))
    await db.commit()
    req = RepairTicketRequest(action='provision', execution_attempt=0,
        reason='Retry confirmed rejected provisioning under unchanged plan', evidence=['acquisition journal'])
    if blocker:
        with pytest.raises(HTTPException):
            await repair_ticket(db, ticket, req)
    else:
        result = await repair_ticket(db, ticket, req)
        assert result['status'] == 'repairing' and ticket.repair_attempts == 3
        assert not (await db.scalars(select(AgentWakeupRequest))).all()
        await complete_review(db)
        assert len((await db.scalars(select(AgentWakeupRequest))).all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['capacity', 'unknown', 'stale', 'exhausted', 'budget', 'live', 'short_wait'])
async def test_cloud_capacity_wait_is_bounded_and_safe(repair_db, tmp_path, monkeypatch, case):
    import json
    from datetime import timedelta
    from zevo.engine.run.cloud_capacity import handle_failure, KEY, next_attempt
    from zevo.engine.run.runner import _queue_ticket_repair, run_ticket
    from zevo.engine.run.wakeup import WakeupDeferred
    db, ticket, execution = repair_db
    await db.delete(execution)
    ticket.agent_id = 'infrastructure'
    hb = HeartbeatRun(id='infra-hb', ticket_id=ticket.id, agent_id='infrastructure', driver='mock', model='test')
    db.add(hb)
    run = await db.get(Run, 'r')
    now = datetime.now(timezone.utc)
    if case == 'exhausted':
        run.lifecycle = {KEY: {ticket.id: {'attempts': 5, 'first_at': now.isoformat(), 'next_at': now.isoformat()}}}
    if case == 'budget':
        monkeypatch.setattr('zevo.engine.cost.budget.snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=True, over_time_limit=False)))
    if case == 'live':
        db.add(InfraInstance(id='live', run_id='r', ticket_id=ticket.id, provider='cloud', instance_id='i-1'))
    if case == 'short_wait':
        run.max_queue_wait_hours = 0.001
    await db.commit()
    state = {'run_id': 'r', 'ticket_id': ticket.id, 'phase': 'create_failed',
        'failure_kind': 'unknown' if case == 'unknown' else 'capacity',
        'confirmed_not_created': case != 'unknown',
        'attempted_at': (datetime.now(timezone.utc) - timedelta(days=1) if case == 'stale' else datetime.now(timezone.utc)).isoformat()}
    (tmp_path / 'acquire_state.json').write_text(json.dumps(state))
    result = await handle_failure(db, ticket, SimpleNamespace(work_dir=str(tmp_path)), hb.id)
    assert result is (True if case == 'capacity' else None if case == 'stale' else False)
    if case == 'capacity':
        due = next_attempt(run, ticket.id)
        assert due > datetime.now(timezone.utc)
        await db.commit()
        await _queue_ticket_repair(db, ticket)
        wake = (await db.scalars(select(AgentWakeupRequest))).one()
        assert wake.scheduled_for.replace(tzinfo=timezone.utc) == due
        with pytest.raises(WakeupDeferred, match='cloud_capacity_backoff'):
            await run_ticket(db, ticket_id=ticket.id, activation_source='on_demand')
        assert ticket.repair_attempts == 3 and ticket.iteration == 2
    else:
        assert not (await db.scalars(select(AgentWakeupRequest))).all()
