"""A completed finite job can bank a partial plan without failing its ticket."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.train import TrainingContinuation, TrainResult
from zevo.engine.run.training_continuation import authorized, verify_files, queue_continuation
from zevo.db import Run, Ticket, InfraInstance, HeartbeatRun, AgentWakeupRequest
from zevo.db.models import Base


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'train-ticket'
    root.mkdir()
    config = root / 'train_config.yaml'
    config.write_text('unchanged 450 step configuration')
    checkpoint = root / 'state' / 'checkpoint-15'
    checkpoint.mkdir(parents=True)
    plan = root / 'state' / 'training_plan.json'
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    plan.write_text(json.dumps(dict(ticket_id='train-ticket', config_sha256=digest, planned_steps=450)))
    for name in ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'training_args.bin',
                 *[f'rng_state_{i}.pth' for i in range(4)]]:
        (checkpoint / name).write_bytes(b'full state test fixture')
    (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=15, max_steps=450)))
    request = TrainingContinuation(checkpoint_path=str(checkpoint), plan_path=str(plan),
        completed_steps=15, planned_steps=450, config_sha256=digest).model_dump()
    return dict(root=str(root), request=request, ticket_id='train-ticket', executed_config=str(config), world_size=4)


def test_partial_complete_job_has_valid_continuation(saved):
    proof = verify_files(**saved)
    assert proof['completed_steps'] == 15 and proof['planned_steps'] == 450
    assert proof['verified']
    result = TrainResult(status='deferred', ticket_id='train-ticket', error_message='', notes='', continuation=saved['request'])
    assert not result.checkpoint_path
    with pytest.raises(ValueError, match='continuation requires deferred'):
        TrainResult(status='succeeded', ticket_id='train-ticket', error_message='', notes='', continuation=saved['request'])


@pytest.mark.parametrize('damage', ['optimizer', 'rng', 'empty', 'step', 'plan', 'config', 'outside', 'symlink', 'weights'])
def test_incomplete_or_changed_state_cannot_continue(saved, damage, tmp_path):
    from pathlib import Path
    checkpoint = Path(saved['request']['checkpoint_path'])
    if damage == 'optimizer': (checkpoint / 'optimizer.pt').unlink()
    if damage == 'rng': (checkpoint / 'rng_state_3.pth').unlink()
    if damage == 'empty': (checkpoint / 'scheduler.pt').write_bytes(b'')
    if damage == 'weights': (checkpoint / 'model.safetensors').unlink()
    if damage == 'step': (checkpoint / 'trainer_state.json').write_text('{"global_step": 14, "max_steps":450}')
    if damage == 'plan': Path(saved['request']['plan_path']).write_text('{}')
    if damage == 'config': Path(saved['executed_config']).write_text('shortened plan')
    if damage == 'outside': saved['request']['checkpoint_path'] = str(tmp_path)
    if damage == 'symlink':
        outside = tmp_path / 'foreign-optimizer'; outside.write_text('other ticket')
        (checkpoint / 'optimizer.pt').unlink(); (checkpoint / 'optimizer.pt').symlink_to(outside)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_files(**saved)


@pytest.mark.asyncio
async def test_verified_exit_zero_queues_same_ticket_without_repair(saved):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        run = Run(id='run', task_name='task', metric='accuracy', status='running')
        ticket = Ticket(id='train-ticket', run_id='run', agent_id='train', status='running', repair_attempts=2)
        hb = HeartbeatRun(id='collect-hb', agent_id='train', ticket_id=ticket.id, driver='test', model='test')
        row = InfraInstance(id='old', instance_id='123', provider='cluster', run_id='run', ticket_id=ticket.id,
            status='released', meta={'scheduler_state': 'COMPLETED'})
        session.add_all([run, ticket, hb, row]); await session.commit()
        proof = {**verify_files(**saved), 'source_resource_id': row.id, 'source_job_id': row.instance_id}
        await queue_continuation(session, ticket, hb, row, proof)
        assert ticket.status == 'queued' and ticket.repair_attempts == 2
        assert hb.finished_at and hb.activation_phase == 'continue'
        assert authorized(ticket, row)
        assert not authorized(SimpleNamespace(id='different', agent_id='train'), row)
        # Redelivery coalesces; it must not produce two new submissions.
        await queue_continuation(session, ticket, hb, row, proof)
        queued = (await session.execute(select(AgentWakeupRequest).where(AgentWakeupRequest.status == 'queued'))).scalars().all()
        assert len(queued) == 1
        assert queued[0].ticket_id == ticket.id
        assert row.meta['training_continuation']['planned_steps'] == 450
        row.meta = {**row.meta, 'scheduler_state': 'RUNNING'}
        assert not authorized(ticket, row)
    await engine.dispose()
