"""A checkpoint startup is distinct from the supervisor's previous success."""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Agent, AgentWakeupRequest, Base, HeartbeatRun, Run, Ticket, TicketNotice
from zevo.engine.run.adaptive_training import checkpoint_orchestration_history, wake_checkpoint_orchestrator
from zevo.engine.run.input_validation import report_input_failure
from zevo.engine.run.runner import _build_input
from zevo.engine.run.supervisor import supervisor_input_signature


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add(Agent(id='orchestrator', name='Orchestrator', title='Orchestrator', identity_path=''))
        session.add(Run(id='r', metric='accuracy', status='running', gpu_provider='cluster', num_gpus=4,
            generation_backend='vllm', gpu_allocation_mode='per_run', supervisor_ticket_id='orch', lifecycle={
                'adaptive_checkpoints': {'new': {'request': {'ticket_id': 'train', 'step': 1468, 'planned_steps': 2447}}},
                'adaptive_training_requests': {'new': {'ticket_id': 'train', 'step': 1468, 'status': 'pending'}},
            }))
        session.add(Ticket(id='train', run_id='r', agent_id='train', status='waiting_external'))
        session.add(Ticket(id='orch', run_id='r', agent_id='orchestrator', status='succeeded', payload={
            'mode': 'full_pipeline', 'task_objective': 'Improve accuracy', 'agent_objective': 'Improve accuracy',
            'user_request': {'task_objective': 'Improve accuracy', 'metric': 'accuracy', 'metric_direction': 'max',
                'training_method': '', 'dataset': '', 'base_model': '', 'test_set': '', 'test_sample_submission': '', 'constraints': []},
            'history': [], 'budget': {'max_cost_usd': 0, 'spent_usd': 0, 'remaining_usd': 0},
        }))
        await session.commit()
        yield session
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize('cached', [None, {'gpu_provider': 'cloud', 'num_gpus': 1, 'generation_backend': 'hf'}])
async def test_invocation_rebuilds_current_configuration_and_checkpoint_trigger(db, cached):
    run, ticket = await db.get(Run, 'r'), await db.get(Ticket, 'orch')
    payload = {**ticket.payload, 'trigger': {'type': 'ticket_completed', 'ticket_id': 'old'}}
    if cached:
        payload['runtime'] = cached
    inp = await _build_input(agent_id='orchestrator', ticket=ticket, payload=payload, inputs={}, work_dir='',
        generation_backend=run.generation_backend, run=run, session=db,
        activation_payload={'adaptive_checkpoint_request_id': 'new', 'adaptive_checkpoint_phase': 'inference'})
    assert (inp.runtime.gpu_provider, inp.runtime.num_gpus, inp.runtime.generation_backend,
            inp.runtime.gpu_allocation_mode) == ('cluster', 4, 'vllm', 'per_run')
    assert inp.trigger.type == 'checkpoint_ready' and inp.trigger.ticket_id == 'train'
    assert inp.runtime.adaptive_checkpoints[0]['step'] == 1468
    assert inp.user_request.task_objective == payload['user_request']['task_objective']
    assert 'runtime' not in ticket.payload  # Invocation does not rewrite historical inputs.


@pytest.mark.asyncio
@pytest.mark.parametrize('source,activation,expected', [
    ('assignment', {}, 'run_created'),
    ('handoff', {'orchestration_trigger': {'type': 'ticket_completed', 'ticket_id': 'current'}}, 'ticket_completed'),
    ('on_demand', {}, 'on_demand'),
])
async def test_non_checkpoint_wakes_do_not_reuse_an_old_completion(db, source, activation, expected):
    run, ticket = await db.get(Run, 'r'), await db.get(Ticket, 'orch')
    inp = await _build_input(agent_id='orchestrator', ticket=ticket,
        payload={**ticket.payload, 'trigger': {'type': 'ticket_completed', 'ticket_id': 'stale'}}, inputs={},
        work_dir='', generation_backend='vllm', run=run, session=db,
        activation_source=source, activation_payload=activation)
    assert inp.trigger.type == expected and inp.trigger.ticket_id != 'stale'


async def wakes(db):
    return list((await db.scalars(select(AgentWakeupRequest).order_by(AgentWakeupRequest.created_at))).all())


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['inference', 'evaluation', 'decision'])
async def test_failed_start_is_visible_and_only_corrected_input_can_retry(db, phase):
    run, ticket = await db.get(Run, 'r'), await db.get(Ticket, 'orch')
    await wake_checkpoint_orchestrator(db, run, 'new', phase)
    first = (await wakes(db))[0]
    await report_input_failure(db, ticket, run, 'Invalid input before agent startup')
    first.status = 'failed'
    await db.commit()
    assert ticket.status == 'failed'  # Previous succeeded cannot hide a new failure.
    assert len((await db.scalars(select(TicketNotice))).all()) == 1
    await wake_checkpoint_orchestrator(db, run, 'new', phase)
    assert len(await wakes(db)) == 1
    run.num_gpus = 2
    await db.commit()
    await wake_checkpoint_orchestrator(db, run, 'new', phase)
    second = (await wakes(db))[-1]
    assert second.id != first.id and second.status == 'queued'
    assert second.payload['supervisor_input_signature'] != first.payload['supervisor_input_signature']
    for state in ['queued', 'running', 'completed']:
        second.status = state
        await db.commit()
        await wake_checkpoint_orchestrator(db, run, 'new', phase)
        assert len(await wakes(db)) == 2
    assert ticket.repair_attempts == 0  # These failures never ran an Agent repair.


@pytest.mark.asyncio
async def test_repeated_corrections_cannot_bypass_twenty_failure_ceiling(db):
    run, ticket = await db.get(Run, 'r'), await db.get(Ticket, 'orch')
    for attempt in range(20):
        run.num_gpus = attempt + 1
        await db.commit()
        await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
        rows = await wakes(db)
        assert len(rows) == attempt + 1
        rows[-1].status = 'failed'
        await db.commit()
    run.num_gpus = 30
    await db.commit()
    await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
    assert len(await wakes(db)) == 20


@pytest.mark.asyncio
async def test_changing_a_cached_runtime_does_not_bypass_unchanged_input_guard(db):
    run, ticket = await db.get(Run, 'r'), await db.get(Ticket, 'orch')
    signature = supervisor_input_signature(run, ticket, {})
    ticket.payload = {**ticket.payload, 'runtime': {'num_gpus': 99}, 'trigger': {'type': 'run_created'}}
    assert supervisor_input_signature(run, ticket, {}) == signature


@pytest.mark.asyncio
async def test_startup_history_keeps_failures_separate_from_previous_and_later_success(db):
    run, train = await db.get(Run, 'r'), await db.get(Ticket, 'train')
    now = datetime.now(timezone.utc)
    previous = HeartbeatRun(id='old', ticket_id='orch', agent_id='orchestrator', driver='test', model='test',
        started_at=now - timedelta(minutes=10), finished_at=now - timedelta(minutes=9), exit_code=0,
        resolved_config={'mid_training': {'request_id': 'old', 'train_ticket_id': 'train', 'step': 735}})
    first = AgentWakeupRequest(id='failed', agent_id='orchestrator', ticket_id='orch', source='on_demand',
        status='failed', reason='Input validation failed', created_at=now,
        payload={'adaptive_checkpoint_request_id': 'new', 'adaptive_checkpoint_phase': 'inference'})
    db.add_all([previous, first])
    await db.commit()
    history = await checkpoint_orchestration_history(db, run, train)
    assert len(history) == 1 and history[0]['heartbeat_id'] == ''
    assert history[0]['error'] == 'Input validation failed' and history[0]['step'] == 1468
    db.add(AgentWakeupRequest(id='retry', agent_id='orchestrator', ticket_id='orch', source='on_demand',
        status='completed', heartbeat_run_id='new-hb', created_at=now + timedelta(minutes=1), payload=dict(first.payload)))
    db.add(HeartbeatRun(id='new-hb', ticket_id='orch', agent_id='orchestrator', driver='test', model='test',
        started_at=now + timedelta(minutes=2), finished_at=now + timedelta(minutes=3), exit_code=0,
        resolved_config={'mid_training': {'request_id': 'new', 'train_ticket_id': 'train', 'step': 1468, 'phase': 'inference'}}))
    await db.commit()
    history = await checkpoint_orchestration_history(db, run, train)
    assert [item['heartbeat_id'] for item in history] == ['', 'new-hb']
    assert history[0]['status'] == 'failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'running', 'completed', 'failed'])
async def test_legacy_decision_without_new_phase_token_is_not_duplicated(db, status):
    run = await db.get(Run, 'r')
    db.add(AgentWakeupRequest(agent_id='orchestrator', ticket_id='orch', source='on_demand', status=status,
        payload={'adaptive_training_request_id': 'new'}))
    await db.commit()
    await wake_checkpoint_orchestrator(db, run, 'new', 'decision')
    assert len(await wakes(db)) == 1


@pytest.mark.asyncio
async def test_decided_request_keeps_its_startup_history(db):
    run, train = await db.get(Run, 'r'), await db.get(Ticket, 'train')
    db.add(AgentWakeupRequest(agent_id='orchestrator', ticket_id='orch', source='on_demand', status='failed',
        payload={'adaptive_training_request_id': 'new'}, reason='Startup rejected'))
    run.lifecycle = {**run.lifecycle, 'adaptive_training_requests': {'new': {
        'ticket_id': 'train', 'step': 1468, 'status': 'decided'}}}
    await db.commit()
    history = await checkpoint_orchestration_history(db, run, train)
    assert len(history) == 1 and history[0]['phase'] == 'decision'


@pytest.mark.asyncio
async def test_measurement_gate_defers_and_releases_exactly_one_new_wake(db, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from zevo.engine.run.scheduler import wakeup_daemon as daemon
    from zevo.engine.run.runner import run_ticket
    monkeypatch.setattr(daemon, 'get_session_factory', lambda: async_sessionmaker(db.bind, expire_on_commit=False))
    monkeypatch.setattr(daemon, 'run_ticket', run_ticket)
    run = await db.get(Run, 'r')
    await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
    first = (await wakes(db))[0]
    run.lifecycle = {**run.lifecycle, 'token_budget_remeasurements': [{'lane': 'optimization', 'status': 'running'}]}
    await db.commit()
    await daemon._process_wakeup(db, first)
    await db.refresh(first)
    await db.refresh(run)
    assert first.status == 'deferred' and first.heartbeat_run_id is None
    assert first.payload['deferred_reason'] == 'measurement_revision_pending'
    await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
    assert len(await wakes(db)) == 1  # No queue churn while the gate remains closed.
    run.lifecycle = {**run.lifecycle, 'token_budget_remeasurements': []}
    await db.commit()
    await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
    await wake_checkpoint_orchestrator(db, run, 'new', 'inference')
    assert [w.status for w in await wakes(db)] == ['deferred', 'queued']
    assert (await db.get(Ticket, 'orch')).repair_attempts == 0


@pytest.mark.asyncio
async def test_exact_activation_link_survives_a_later_request_queued_before_start(db):
    from zevo.engine.run.wakeup import bind_wakeup_heartbeat
    run, train, orch = await db.get(Run, 'r'), await db.get(Ticket, 'train'), await db.get(Ticket, 'orch')
    now = datetime.now(timezone.utc)
    payload = {'adaptive_checkpoint_request_id': 'new', 'adaptive_checkpoint_phase': 'inference', 'activation_link_recorded': True}
    first = AgentWakeupRequest(id='first', agent_id='orchestrator', ticket_id='orch', source='on_demand',
        status='running', created_at=now, payload=payload)
    later = AgentWakeupRequest(id='later', agent_id='orchestrator', ticket_id='orch', source='on_demand',
        status='queued', created_at=now + timedelta(seconds=1), payload=payload)
    hb = HeartbeatRun(id='exact-hb', ticket_id='orch', agent_id='orchestrator', driver='test', model='test',
        started_at=now + timedelta(seconds=2), resolved_config={'mid_training': {
            'request_id': 'new', 'train_ticket_id': 'train', 'step': 1468, 'phase': 'inference'}})
    db.add_all([first, later, hb])
    await db.flush()
    await bind_wakeup_heartbeat(db, first.id, orch, hb)
    await db.commit()
    history = await checkpoint_orchestration_history(db, run, train)
    assert [(r['wakeup_id'], r['heartbeat_id']) for r in history] == [('first', 'exact-hb'), ('later', '')]
    first.heartbeat_run_id = None
    first.payload = {k: v for k, v in payload.items() if k != 'activation_link_recorded'}
    await db.commit()
    history = await checkpoint_orchestration_history(db, run, train)
    assert history[0]['heartbeat_id'] == '' and history[0]['activation_link_known'] is False
