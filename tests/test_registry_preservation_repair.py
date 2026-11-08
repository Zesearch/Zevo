"""A preservation failure happens after Registry runs and uses normal repair."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.db import Base, Run, Ticket, TicketMessage, HeartbeatResult
from zevo.contracts.model_registry import RegisterResult
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.run import runner, model_save, model_save_worker


@pytest.mark.asyncio
async def test_registry_agent_repairs_card_after_backend_failure(tmp_path, monkeypatch):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/run.db')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr('zevo.db.get_session_factory', lambda: Session)
    source = tmp_path / 'model'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    (source / 'model.safetensors').write_bytes(b'unchanged-weights')
    bad_card = '---\nbase_model: /local/cache/model\n---\nModel description\n'
    good_card = '---\nbase_model: owner/base\n---\nModel description\n'
    (source / 'README.md').write_text(bad_card)
    original = model_save_worker.manifest(source)
    metrics = tmp_path / 'metrics.json'
    metrics.write_text('{"score": 0.75}')
    train_config = tmp_path / 'train_config.yaml'
    train_config.write_text('training_method: sft\nparent_model: owner/base\n')
    events = []
    provenance = dict(base_model='owner/base', training_method='sft', dataset_source='dataset',
                      task_objective='task', metric='accuracy', metric_direction='max')
    policy = dict(weights='hf', hf_repo_id='owner/result', hf_private=True, remote_dir='')

    async def build(agent_id, ticket, payload, inputs, work_dir, **kwargs):
        inp = runner._build_registry_input(ticket, provenance, {
            'checkpoint': {'path': str(source)}, 'metrics': {'path': str(metrics)},
            'train_config': {'path': str(train_config)},
        }, work_dir)
        inp.model_save_policy = policy
        return inp

    class Driver:
        name = 'stub'
        async def run_agent(self, *, input_payload, workspace_dir, conversation, **kwargs):
            inp = input_payload
            events.append('agent')
            work = Path(workspace_dir)
            card = ''
            if events.count('agent') == 2:
                assert 'base_model' in json.dumps(conversation)
                corrected = work / 'README.md'
                corrected.write_text(good_card)
                card = str(corrected)
            entry = dict(run_id=inp.run_id, ticket_id=inp.ticket_id, iteration=1, **provenance,
                model_path=str(source), storage='hf', revision='', retention='retained',
                eval={'score': .75}, registered_at=datetime.now(timezone.utc).isoformat())
            manifest = work / 'registry.yaml'
            manifest.write_text(yaml.safe_dump({'models': {inp.expected_version_tag: entry}}))
            script = work / 'register.py'
            script.write_text('# reproducible preparation\n')
            return DriverRunResult(output=RegisterResult(status='succeeded', ticket_id=inp.ticket_id,
                notes='prepared', error_message='', registry_path=str(manifest), register_script_path=str(script), model_card_path=card))

    async def save(session, run, ticket, *, source, remote, policy, model_card=None):
        events.append('save')
        assert events[-2] == 'agent'
        if model_card is None:
            raise RuntimeError('Model preservation failed: base_model.0 /local/cache/model is not a model ID')
        assert model_card == good_card
        receipt = dict(storage='hf', path='https://huggingface.co/owner/result/tree/commit1', revision='commit1',
            identity=model_save.save_identity(run, source, policy), ticket_id=ticket.id)
        run.lifecycle = {**run.lifecycle, 'model_storage': receipt}
        await session.commit()
        return receipt

    monkeypatch.setattr(runner, '_build_input', build)
    monkeypatch.setattr(runner, 'get_driver', lambda _: Driver())
    monkeypatch.setattr(model_save, 'save_model', save)
    async with Session() as db:
        run = Run(id='run123', metric='accuracy', status='running', lifecycle={'model_save_policy': policy})
        ticket = Ticket(id='registry-run123', run_id=run.id, agent_id='registry', iteration=1,
                        status='queued', payload=provenance, inputs={})
        db.add_all([run, ticket])
        await db.commit()
        first = await runner.run_ticket(db, ticket_id=ticket.id, work_dir_root=str(tmp_path / 'runs'), driver_name='stub')
        assert events == ['agent', 'save'], (first.error_message, first.summary)
        assert first.status == 'repairing' and first.repair_attempts == 1
        assert not run.lifecycle.get('model_storage')
        messages = (await db.scalars(select(TicketMessage).where(TicketMessage.ticket_id == ticket.id))).all()
        assert any('base_model.0' in m.body for m in messages)
        second = await runner.run_ticket(db, ticket_id=ticket.id, work_dir_root=str(tmp_path / 'runs'),
                                         driver_name='stub', activation_source='retry')
        assert events == ['agent', 'save', 'agent', 'save']
        assert second.status == 'succeeded'
        results = (await db.scalars(select(HeartbeatResult).order_by(HeartbeatResult.created_at))).all()
        assert [r.status for r in results] == ['failed', 'succeeded']
        assert model_save_worker.manifest(source) == original
        final = yaml.safe_load(Path(results[-1].output['registry_path']).read_text())['models']['M-run123']
        assert final['revision'] == 'commit1'
        assert final['model_path'] == run.lifecycle['model_storage']['path']
        assert final['eval'] == {'score': .75}
    await engine.dispose()
