from datetime import datetime, timezone
from pathlib import Path
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
async def repair_db(monkeypatch, tmp_path):
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
    config = tmp_path / 'train_config.yaml'
    config.write_text('training:\n  learning_rate: 0.00001\n')
    execution.meta = {**execution.meta, 'training_config_local_path': str(config),
        'training_config_snapshot': config.read_text(),
        'training_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
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
    with pytest.raises(ValueError, match='snapshot unavailable'):
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
    from zevo.db import HeartbeatResult
    if not await db.scalar(select(HeartbeatResult.id).where(HeartbeatResult.heartbeat_id == hb.id)):
        db.add(HeartbeatResult(ticket_id=supervisor.id, heartbeat_id=hb.id,
            agent_id='orchestrator', status='succeeded',
            output={'ticket_id': supervisor.id, 'action': 'wait'}))
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


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['data', 'train', 'inference', 'evaluation', 'infrastructure', 'registry'])
async def test_upstream_failure_waits_for_review_without_self_retry(repair_db, agent):
    from zevo.contracts.failure import FailureRecovery
    from zevo.engine.run.runner import _apply_ticket_failure_policy
    from zevo.engine.run.reviewed_recovery import latest_request
    db, ticket, execution = repair_db
    ticket.agent_id = agent
    result = AgentResult(status='failed', ticket_id=ticket.id, error_message='missing gold', notes='',
        failure_recovery=FailureRecovery(scope='upstream_input', reason='upstream artifact is incomplete',
            evidence=['dataset schema has no gold'], next_action='prepare a corrected Data version'))
    assert not await _apply_ticket_failure_policy(db, ticket, error_message=result.error_message,
        result=result, heartbeat_id='failed-hb')
    await db.flush()
    assert ticket.status == 'awaiting_input' and ticket.repair_route == 'orchestrator'
    assert ticket.repair_attempts == 2
    assert (await latest_request(db, ticket))['request']['scope'] == 'upstream_input'


@pytest.mark.asyncio
@pytest.mark.parametrize('lane,attempts', [('held_out_test', 2), ('optimization', 20)])
async def test_review_wait_does_not_bypass_isolation_or_repair_limit(repair_db, lane, attempts):
    from zevo.contracts.failure import FailureRecovery
    from zevo.engine.run.runner import _apply_ticket_failure_policy
    db, ticket, _ = repair_db
    ticket.lane, ticket.repair_attempts = lane, attempts
    result = AgentResult(status='failed', ticket_id=ticket.id, error_message='upstream', notes='',
        failure_recovery=FailureRecovery(scope='upstream_input', reason='missing input',
            evidence=['missing file'], next_action='repair data'))
    await _apply_ticket_failure_policy(db, ticket, error_message=result.error_message, result=result)
    assert ticket.status == 'failed'
    if lane == 'held_out_test': assert ticket.repair_route == 'terminal'


async def configuration_proposal(db, ticket, execution, tmp_path, monkeypatch):
    import json
    from zevo.contracts.failure import TrainingConfigurationChange
    from zevo.db import TicketNotice
    from zevo.engine.run.reviewed_recovery import WAIT_CODE
    from zevo.contracts import configuration
    original = {'parent_model': 'parent', 'data_signature': 'data',
                'training': {'learning_rate': 0.01, 'world_size': 2}}
    candidate = {**original, 'training': {**original['training'], 'learning_rate': 0.001}}
    paths = [tmp_path / 'original.yaml', tmp_path / 'candidate.yaml']
    for path, value in zip(paths, [original, candidate]): path.write_text(json.dumps(value))
    # The production schema loader is independently tested; exercise admission,
    # byte binding, immutable identity and release against a small configuration.
    monkeypatch.setattr(configuration, 'load_train_config',
        lambda path: SimpleNamespace(model_dump=lambda: json.loads(Path(path).read_text())))
    proposal = TrainingConfigurationChange(original_config_path=str(paths[0]), proposed_config_path=str(paths[1]),
        original_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        proposed_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(), changes=['training.learning_rate: 0.01 -> 0.001'])
    execution.meta = {**execution.meta, 'training_config_sha256': proposal.original_sha256}
    ticket.status = 'awaiting_input'
    db.add(TicketNotice(ticket_id=ticket.id, code=WAIT_CODE, severity='info', body=json.dumps({
        'request': {'scope':'fixed_configuration', 'training_configuration_change':proposal.model_dump()}})))
    await db.commit()
    return proposal, paths


@pytest.mark.asyncio
async def test_configuration_change_only_runs_after_review_and_exact_hash(repair_db, tmp_path, monkeypatch):
    db, ticket, execution = repair_db
    proposal, paths = await configuration_proposal(db, ticket, execution, tmp_path, monkeypatch)
    req = RepairTicketRequest(action='reconfigure', execution_attempt=2, reason='numerical repair',
        evidence=['first nonfinite gradient at step 7'], training_configuration_change=proposal)
    await repair_ticket(db, ticket, req)
    assert ticket.repair_attempts == 3 and ticket.iteration == 2
    with pytest.raises(ValueError): verify_repair_training_config(execution, paths[1])
    verify_repair_training_config(execution, paths[0])
    await complete_review(db)
    verify_repair_training_config(execution, paths[1])
    with pytest.raises(ValueError): verify_repair_training_config(execution, paths[0])
    paths[1].write_text('changed after approval')
    with pytest.raises(ValueError): verify_repair_training_config(execution, paths[1])
    assert (await db.get(WorkProduct, 'product')).path == '/durable/checkpoint-735'
    approval = execution.meta['training_configuration_approval']
    assert approval['restart_mode'] == 'restart_parent'
    assert approval['configuration_diff']['training.learning_rate']['before'] == 0.01
    assert approval['configuration_diff']['training.learning_rate']['after'] == 0.001


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['pin', 'changed_bytes', 'stale_hash', 'wrong_proposal', 'not_train'])
async def test_configuration_review_rejects_invalid_changes(repair_db, tmp_path, monkeypatch, fault):
    db, ticket, execution = repair_db
    proposal, paths = await configuration_proposal(db, ticket, execution, tmp_path, monkeypatch)
    if fault == 'pin': ticket.payload = {'configuration_pins': {'learning_rate': 0.01}}
    if fault == 'changed_bytes': paths[1].write_text('unapproved')
    if fault == 'stale_hash': execution.meta = {**execution.meta, 'training_config_sha256': 'f'*64}
    if fault == 'wrong_proposal': proposal = proposal.model_copy(update={'changes':['different request']})
    if fault == 'not_train': ticket.agent_id = 'inference'
    await db.commit()
    with pytest.raises(HTTPException):
        await repair_ticket(db, ticket, RepairTicketRequest(action='reconfigure', execution_attempt=2,
            reason='repair', evidence=['diagnostics'], training_configuration_change=proposal))
    assert ticket.status == 'awaiting_input' and ticket.repair_attempts == 2
    assert not execution.meta.get('training_configuration_approval')


@pytest.mark.asyncio
async def test_upstream_dependency_requires_waiting_target_and_supervisor(repair_db):
    import json
    from zevo.db import TicketNotice
    from zevo.engine.run.reviewed_recovery import validate_dependency, WAIT_CODE
    db, ticket, _ = repair_db
    run = await db.get(Run, 'r')
    ticket.status = 'awaiting_input'
    db.add(TicketNotice(ticket_id=ticket.id, code=WAIT_CODE, severity='info',
        body=json.dumps({'request':{'scope':'upstream_input'}})))
    await db.commit()
    assert await validate_dependency(db, run, ticket.id, child_agent='data', iteration=2) is ticket
    with pytest.raises(HTTPException):
        await validate_dependency(db, run, ticket.id, child_agent='data', iteration=3)
    (await db.get(HeartbeatRun, 'decision')).finished_at = datetime.now(timezone.utc)
    await db.commit()
    with pytest.raises(HTTPException):
        await validate_dependency(db, run, ticket.id, child_agent='data', iteration=2)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['', 'pending_source', 'unlinked', 'executed_train'])
async def test_resume_original_train_with_reviewed_new_data(repair_db, tmp_path, fault):
    import json
    from zevo.db import HeartbeatResult, TicketNotice
    from zevo.engine.run.reviewed_recovery import WAIT_CODE, DEPENDENCY_CODE
    db, ticket, execution = repair_db
    if fault != 'executed_train': await db.delete(execution)
    await db.delete(await db.get(WorkProduct, 'product'))
    ticket.status = 'awaiting_input'
    ticket.payload = {'operation':'train', 'base_model':'org/model', 'model_source':'base_model',
                      'parent_selection_rationale':'first training', 'data_signature':'a'*64}
    infra = Ticket(id='infra', run_id='r', agent_id='infrastructure', status='succeeded', payload={'purpose':'train'})
    baseline = Ticket(id='baseline', run_id='r', agent_id='inference', status='succeeded',
        payload={'model_source':'base_model','base_model':'org/model'})
    data = Ticket(id='data-fixed', run_id='r', agent_id='data', iteration=2,
        status='running' if fault == 'pending_source' else 'succeeded',
        inputs={'device_info':{'source_ticket_id':'infra'}})
    old = Ticket(id='data-old', run_id='r', agent_id='data', status='succeeded')
    db.add_all([infra, baseline, data, old])
    for tid, role, pid in [('infra','device_info','device'),('baseline','inference_config','baseline-config'),
                           ('data-fixed','training_dataset','new-data'),('data-old','training_dataset','old-data')]:
        path = tmp_path / pid; path.write_text('{}')
        db.add(WorkProduct(id=pid,ticket_id=tid,role=role,path=str(path),meta={'data_signature':'b'*64}))
    ticket.inputs = {'training_dataset':{'source_ticket_id':'data-old','artifact_role':'training_dataset','work_product_id':'old-data'},
                     'device_info':{'source_ticket_id':'infra','artifact_role':'device_info','work_product_id':'device'},
                     'inference_config':{'source_ticket_id':'baseline','artifact_role':'inference_config','work_product_id':'baseline-config'}}
    db.add(HeartbeatResult(ticket_id=ticket.id,agent_id='train',status='failed',
        output={'status':'failed','failure_recovery':{'scope':'upstream_input'}}))
    db.add(TicketNotice(ticket_id=ticket.id,code=WAIT_CODE,severity='info',body=json.dumps({'request':{'scope':'upstream_input'}})))
    if fault != 'unlinked': db.add(TicketNotice(ticket_id=data.id,code=DEPENDENCY_CODE,severity='info',body=ticket.id))
    await db.commit()
    req = RepairTicketRequest(action='resume_inputs',execution_attempt=2 if fault == 'executed_train' else 0,
        reason='Verified gold repaired',evidence=['all rows have reliable gold'],input_products={'training_dataset':'new-data'})
    if fault:
        with pytest.raises(HTTPException): await repair_ticket(db,ticket,req)
        assert ticket.status == 'awaiting_input' and ticket.inputs['training_dataset']['work_product_id']=='old-data'
        return
    await repair_ticket(db,ticket,req)
    assert ticket.status == 'repairing' and ticket.repair_attempts==3 and ticket.iteration==2
    assert ticket.inputs['training_dataset']['work_product_id']=='new-data'
    assert ticket.payload['data_signature']=='b'*64
    assert not (await db.scalars(select(AgentWakeupRequest))).all()
    await complete_review(db)
    assert len((await db.scalars(select(AgentWakeupRequest))).all())==1
    assert (await db.get(WorkProduct,'old-data')) is not None


@pytest.mark.asyncio
async def test_create_linked_data_while_original_train_waits(repair_db):
    import json
    from zevo.api.routers.shared.tickets import create_ticket
    from zevo.contracts.tickets import CreateTicketBody
    from zevo.db import Agent, HeartbeatResult, TicketNotice
    from zevo.engine.run.reviewed_recovery import WAIT_CODE, DEPENDENCY_CODE
    db, ticket, execution = repair_db
    await db.delete(execution)
    await db.delete(await db.get(WorkProduct, 'product'))
    ticket.status = 'awaiting_input'
    from zevo.contracts.tickets import DataPayload
    from zevo.api.routers.shared.tickets import _resolve_data_source_selection
    db.add(Ticket(id='initial-data',run_id='r',agent_id='data',iteration=0,status='succeeded',
        payload=_resolve_data_source_selection(DataPayload(operation='prepare_run_data',dataset='org/data',training_method='sft').model_dump())))
    db.add(Agent(id='data',name='data',title='data',default_driver='stub',default_model='stub',
        output_schema='',tools=[],identity_path=''))
    db.add(Ticket(id='infra',run_id='r',agent_id='infrastructure',status='succeeded',payload={'purpose':'train'}))
    db.add(WorkProduct(id='device',ticket_id='infra',role='device_info',path='/remote/device.json'))
    db.add(HeartbeatResult(ticket_id=ticket.id,agent_id='train',status='failed',
        output={'status':'failed','failure_recovery':{'scope':'upstream_input'}}))
    db.add(TicketNotice(ticket_id=ticket.id,code=WAIT_CODE,severity='info',body=json.dumps({'request':{'scope':'upstream_input'}})))
    await db.commit()
    body = CreateTicketBody(agent_id='data',run_id='r',iteration=2,
        inputs={'device_info':{'artifact_role':'device_info','source_ticket_id':'infra'}},
        payload={'operation':'prepare_run_data','dataset':'org/data','training_method':'sft',
                 'recipe_intent':{'direction':'add verified gold'}})
    with pytest.raises(HTTPException): await create_ticket(body,db=db)
    created = await create_ticket(body.model_copy(update={'repairs_ticket_id':ticket.id}),db=db)
    assert ticket.status=='awaiting_input' and ticket.repair_attempts==3
    assert (await db.scalar(select(TicketNotice).where(TicketNotice.ticket_id==created.id,
        TicketNotice.code==DEPENDENCY_CODE))).body==ticket.id
    assert not (await db.scalars(select(AgentWakeupRequest))).all()


@pytest.mark.asyncio
async def test_waiting_repair_cannot_be_started_by_old_wakeup(repair_db):
    from zevo.engine.run.runner import run_ticket
    db, ticket, _ = repair_db
    ticket.status = 'awaiting_input'
    await db.commit()
    assert await run_ticket(db, ticket_id=ticket.id, activation_source='retry') is ticket
    assert ticket.status == 'awaiting_input'
    assert not (await db.scalars(select(HeartbeatRun).where(HeartbeatRun.ticket_id==ticket.id))).all()


@pytest.mark.asyncio
async def test_repair_chain_allows_waiting_ancestors_but_not_unrelated_work(repair_db):
    from zevo.db import TicketNotice
    from zevo.engine.run.reviewed_recovery import waiting_ancestors, DEPENDENCY_CODE
    db, ticket, _ = repair_db
    parent = Ticket(id='waiting-parent', run_id='r', agent_id='registry', status='awaiting_input', repair_route='orchestrator')
    unrelated = Ticket(id='unrelated', run_id='r', agent_id='data', status='awaiting_input', repair_route='orchestrator')
    db.add_all([parent, unrelated, TicketNotice(ticket_id=ticket.id,code=DEPENDENCY_CODE,severity='info',body=parent.id)])
    await db.commit()
    assert await waiting_ancestors(db,ticket) == {ticket.id,parent.id}
    parent.status='cancelled'
    await db.commit()
    assert await waiting_ancestors(db,ticket) == {ticket.id}


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['parent_model','world_size'])
async def test_config_review_cannot_change_experiment_or_pinned_parallelism(repair_db,tmp_path,monkeypatch,field):
    import json
    from zevo.db import TicketNotice
    from zevo.engine.run.reviewed_recovery import WAIT_CODE
    db,ticket,execution=repair_db
    proposal,paths=await configuration_proposal(db,ticket,execution,tmp_path,monkeypatch)
    config=json.loads(paths[1].read_text())
    if field=='parent_model': config[field]='different-model'
    else:
        ticket.payload = {**(ticket.payload or {}), 'configuration_pins': {'world_size': 1}}
        config['training'][field]=8
    paths[1].write_text(json.dumps(config))
    proposal=proposal.model_copy(update={'proposed_sha256':hashlib.sha256(paths[1].read_bytes()).hexdigest()})
    notice=await db.scalar(select(TicketNotice).where(TicketNotice.ticket_id==ticket.id,TicketNotice.code==WAIT_CODE))
    notice.body=json.dumps({'request':{'scope':'fixed_configuration','training_configuration_change':proposal.model_dump()}})
    await db.commit()
    with pytest.raises(HTTPException):
        await repair_ticket(db,ticket,RepairTicketRequest(action='reconfigure',execution_attempt=2,
            reason='repair',evidence=['diagnostics'],training_configuration_change=proposal))
    assert not execution.meta.get('training_configuration_approval')


def test_external_failure_without_retry_request_still_freezes_training_config(tmp_path):
    config = tmp_path / 'train_config.yaml'
    config.write_text('registered')
    previous = SimpleNamespace(meta={'scheduler_state':'FAILED', 'execution_attempt':2,
        'training_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()})
    assert not execution_retry_authorized(previous.meta)
    verify_repair_training_config(previous,config)
    config.write_text('unapproved repair')
    with pytest.raises(ValueError,match='fixed_configuration'):
        verify_repair_training_config(previous,config)


@pytest.mark.asyncio
async def test_release_observes_separate_approval_transaction(repair_db):
    from sqlalchemy.ext.asyncio import AsyncSession
    db, ticket, _ = repair_db
    cached = await db.get(HeartbeatRun, 'decision')
    assert not (cached.resolved_config or {}).get('repair_target_ticket_ids')
    async with AsyncSession(bind=db.bind, expire_on_commit=False) as api:
        await repair_ticket(api, await api.get(Ticket, ticket.id), request())
    # Runner retains the heartbeat it loaded before the API committed approval.
    assert not (cached.resolved_config or {}).get('repair_target_ticket_ids')
    await complete_review(db)
    wakes = (await db.scalars(select(AgentWakeupRequest))).all()
    assert len(wakes) == 1 and wakes[0].ticket_id == ticket.id


@pytest.mark.asyncio
@pytest.mark.parametrize('blocker', ['', 'unfinished', 'no_result', 'failed_result', 'cancel', 'stale_attempt', 'pause', 'budget'])
async def test_reconcile_reviewed_gate_without_heartbeat_target_list(repair_db, blocker, monkeypatch):
    from zevo.db import HeartbeatResult
    from zevo.engine.run.execution_repair import reconcile_reviewed_repairs
    from zevo.engine.run.supervisor_handoff import waiting
    db, ticket, _ = repair_db
    await repair_ticket(db, ticket, request())
    hb = await db.get(HeartbeatRun, 'decision')
    hb.resolved_config = {}  # overwritten/cached config must not lose durable authorization
    if blocker != 'unfinished': hb.finished_at = datetime.now(timezone.utc)
    (await db.get(Ticket, 'orch')).status = 'succeeded'
    if blocker != 'no_result':
        db.add(HeartbeatResult(ticket_id='orch', heartbeat_id='decision', agent_id='orchestrator',
            status='failed' if blocker == 'failed_result' else 'succeeded',
            output={'ticket_id': 'orch', 'action': 'wait'}))
    if blocker == 'cancel': (await db.get(Run, 'r')).cancel_requested_at = datetime.now(timezone.utc)
    if blocker == 'stale_attempt': ticket.repair_attempts += 1
    if blocker == 'pause': db.add(RunInstruction(run_id='r', body='pause', status='needs_input'))
    if blocker == 'budget':
        from zevo.engine.cost import budget
        monkeypatch.setattr(budget, 'snapshot_for_run', AsyncMock(return_value=SimpleNamespace(over_budget=True, over_time_limit=False)))
    await db.commit()
    await reconcile_reviewed_repairs(db)
    await reconcile_reviewed_repairs(db)
    wakes = (await db.scalars(select(AgentWakeupRequest))).all()
    assert len(wakes) == (0 if blocker else 1)
    assert await waiting(db, ticket) is bool(blocker)


@pytest.mark.asyncio
async def test_approval_and_submission_share_semantic_comparison(repair_db):
    import yaml
    db, ticket, execution = repair_db
    path = Path(execution.meta['training_config_local_path'])
    value = yaml.safe_load(path.read_text())
    value['training']['learning_rate'] = 2e-5
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(HTTPException, match='learning_rate'):
        await repair_ticket(db, ticket, request())
    assert ticket.status == 'failed' and ticket.repair_attempts == 2
    value['training']['learning_rate'] = 1e-5
    path.write_text('# execution-only formatting update\n' + yaml.safe_dump(value))
    await repair_ticket(db, ticket, request())
    assert execution.meta['training_config_retry_audit']['changes'] == []
    # Subsequent unapproved critical edits are still rejected at submission.
    value['training']['learning_rate'] = 3e-5
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='learning_rate'):
        verify_repair_training_config(execution, path)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["safe", "unmarked", "no_history", "active", "unknown_creation", "released_resource"])
async def test_infra_pre_dispatch_repair_requires_positive_evidence(repair_db, case):
    import json
    from zevo.db import TicketNotice
    db, ticket, execution = repair_db
    await db.delete(execution)
    ticket.agent_id = "infrastructure"
    ticket.payload = {"operation": "provision", "purpose": "train"}
    if case != "no_history":
        db.add(HeartbeatRun(id="failed-infra", ticket_id=ticket.id,
            agent_id="infrastructure", driver="mock", model="test",
            finished_at=None if case == "active" else datetime.now(timezone.utc),
            resolved_config={"submission_contract_failed_before_dispatch": True}))
    if case == "unmarked":
        db.add(HeartbeatRun(id="older-infra", ticket_id=ticket.id,
            agent_id="infrastructure", driver="mock", model="test",
            finished_at=datetime.now(timezone.utc)))
    if case == "unknown_creation":
        db.add(TicketNotice(ticket_id=ticket.id, code="cloud.acquisition_failed",
            body=json.dumps({"confirmed_not_created": False})))
    if case == "released_resource":
        db.add(InfraInstance(id="old-resource", run_id="r", ticket_id=ticket.id,
            provider="cloud", instance_id="i-1", released_at=datetime.now(timezone.utc)))
    await db.commit()
    req = RepairTicketRequest(action="provision", execution_attempt=0,
        reason="Fixed schema binding before Agent dispatch", evidence=["engine pre-dispatch marker"])
    if case != "safe":
        with pytest.raises(HTTPException):
            await repair_ticket(db, ticket, req)
        assert ticket.status == "failed" and ticket.repair_attempts == 2
    else:
        await repair_ticket(db, ticket, req)
        assert ticket.status == "repairing" and ticket.repair_attempts == 3
        assert ticket.payload == {"operation": "provision", "purpose": "train"}
        assert not (await db.scalars(select(AgentWakeupRequest))).all()
        await complete_review(db)
        assert len((await db.scalars(select(AgentWakeupRequest))).all()) == 1


def member_configs(tmp_path):
    import yaml
    from tests.test_sequential_yaml_workflow import _write_inference_config
    old=_write_inference_config(tmp_path)
    body=yaml.safe_load(old.read_text())
    body['measurement']['inference_config']['answer_regex']=r'Final Answer: ([A-D])'
    old.write_text(yaml.safe_dump(body))
    new=tmp_path/'member.yaml'
    body['measurement']['inference_config']['answer_regex']=r'Final Answer: ([A-J])'
    new.write_text(yaml.safe_dump(body))
    return old,new


@pytest.mark.parametrize('fault',['','decoding','template','narrow','syntax','pin','missing'])
def test_frozen_member_compatibility(tmp_path,fault):
    import yaml
    from zevo.engine.run.reviewed_recovery import compatible_training_member
    old,new=member_configs(tmp_path)
    body=yaml.safe_load(new.read_text())
    if fault=='decoding': body['measurement']['max_new_tokens']=2048; body['stop_token_ids']=[3]
    if fault=='template': body['chat_template_hash']='b'*64
    if fault=='narrow': body['measurement']['inference_config']['answer_regex']=r'Final Answer: ([A-C])'
    if fault=='syntax': body['measurement']['inference_config']['answer_regex']=r'Answer: ([A-J])'
    new.write_text(yaml.safe_dump(body))
    if fault=='missing': new.unlink()
    pins=[{'inference_config':{'answer_regex':r'Final Answer: ([A-D])'}}] if fault=='pin' else []
    if fault:
        with pytest.raises(HTTPException): compatible_training_member(old,new,pins=pins)
    else:
        result=compatible_training_member(old,new,pins=pins)
        assert list(result['changes'])==['measurement.inference_config.answer_regex']
        assert result['sha256']!=result['previous_sha256']


@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['','executed','other_baseline','new_product','heldout'])
async def test_reviewed_train_member_rebinding(repair_db,tmp_path,monkeypatch,fault):
    import json
    from datetime import timedelta
    from zevo.db import TicketNotice
    from zevo.engine.run.reviewed_recovery import restore_inputs, WAIT_CODE
    from zevo.api.routers.shared import tickets as api
    db,ticket,execution=repair_db
    if fault!='executed': await db.delete(execution)
    await db.delete(await db.get(WorkProduct,'product'))
    ticket.status='awaiting_input'
    ticket.payload={'operation':'train','base_model':'Qwen/Qwen3-0.6B-Base',
        'model_source':'base_model','parent_selection_rationale':'initial'}
    old,new=member_configs(tmp_path)
    earlier=ticket.created_at-timedelta(seconds=10)
    db.add(Ticket(id='baseline',run_id='r',agent_id='inference',status='succeeded',
        lane='held_out_test' if fault=='heldout' else 'optimization',
        payload={'model_source':'base_model','base_model':'Qwen/Qwen3-0.6B-Base'}))
    db.add(Ticket(id='other',run_id='r',agent_id='inference',status='succeeded'))
    db.add_all([WorkProduct(id='old-config',ticket_id='baseline',role='inference_config',path=str(old),created_at=earlier),
        WorkProduct(id='new-config',ticket_id='other' if fault=='other_baseline' else 'baseline',role='inference_config',path=str(new),
        created_at=ticket.created_at+timedelta(seconds=10) if fault=='new_product' else earlier)])
    ticket.inputs={'inference_config':{'source_ticket_id':'baseline','artifact_role':'inference_config','work_product_id':'old-config','path':str(old)}}
    db.add(TicketNotice(ticket_id=ticket.id,code=WAIT_CODE,body=json.dumps({'request':{'scope':'upstream_input'}})))
    await db.commit()
    async def validate(*args,**kwargs): return kwargs['payload'],kwargs['inputs']
    monkeypatch.setattr(api,'validate_pipeline_bindings',validate)
    req=RepairTicketRequest(action='resume_inputs',execution_attempt=0,reason='Select compatible member',
        evidence=['same frozen baseline'],input_products={'inference_config':'new-config'})
    if fault:
        with pytest.raises(HTTPException): await restore_inputs(db,await db.get(Run,'r'),ticket,req,None)
        assert ticket.inputs['inference_config']['work_product_id']=='old-config'
    else:
        await restore_inputs(db,await db.get(Run,'r'),ticket,req,None)
        assert ticket.inputs['inference_config']['work_product_id']=='new-config'
        await db.flush()
        notice=await db.scalar(select(TicketNotice).where(TicketNotice.code=='repair.inputs_rebound'))
        assert json.loads(notice.body)['inference_member_change']['changes']
        assert (await db.get(WorkProduct,'old-config')).path==str(old)
