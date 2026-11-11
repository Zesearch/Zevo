from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from zevo.contracts.freeform import FreeformInput
from zevo.contracts.tickets import ArtifactBinding
from zevo.contracts.data import DataResult, DataTaskInput
from zevo.db.models import Base, Run, Ticket, TicketMessage
from zevo.engine.agent.loader import load_agent
from zevo.engine.run.freeform_resolution import FreeformResolution, authorize_paths, resolution_blueprint, resolve_work_order
from zevo.engine.run.runner import _build_input, _bind_system_validation_artifacts, _extract_summary_artifact_meta


def request(tmp_path, agent='data'):
    source = tmp_path/'source.jsonl'
    source.write_text('{"prompt":"question", "completion":"answer"}\n')
    return FreeformInput(ticket_id='task', agent_id=agent, request='Convert the attached dataset to SFT JSONL',
                         attachments=[str(source)], work_dir=str(tmp_path), run_id='run')


@pytest.mark.parametrize('agent', ['data', 'train', 'inference', 'registry', 'infrastructure'])
def test_every_specialist_resolves_before_execution(tmp_path, agent):
    inp = request(tmp_path, agent)
    original = load_agent(agent, input_format='freeform')
    blueprint = resolution_blueprint(original, inp, {})
    assert blueprint.output_schema is FreeformResolution
    assert 'preparation only' in blueprint.instructions
    assert original.output_schema is not FreeformResolution


def test_resolution_cannot_rebind_existing_input(tmp_path):
    inp = request(tmp_path)
    existing = {'device_info': {'artifact_role': 'device_info', 'path': '/bound/device.json'}}
    changed = FreeformResolution(ticket_id='task', explanation='', inputs={'device_info':
        ArtifactBinding(artifact_role='device_info', path=str(tmp_path/'different.json'))})
    with pytest.raises(ValueError, match='cannot replace'):
        authorize_paths(changed, inp, existing, standalone=True)


@pytest_asyncio.fixture
async def database():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        run = Run(id='run', mode='single_stage', status='running', metric='accuracy')
        ticket = Ticket(id='task', run_id='run', agent_id='data', input_format='freeform',
                        payload={'request': 'Transform the attachment'}, status='running', lane='optimization')
        session.add_all([run, ticket]); await session.commit()
        yield session, run, ticket
    await engine.dispose()


@pytest.mark.asyncio
async def test_freeform_data_builds_local_contract_and_accepts_result_without_scoring(database, tmp_path):
    session, run, ticket = database
    inp = request(tmp_path)
    resolution = FreeformResolution(ticket_id='task', explanation='Use attached source as requested',
        payload={'operation': 'prepare_run_data', 'dataset': inp.attachments[0], 'training_method': 'sft'})
    driver = SimpleNamespace(run_agent=AsyncMock(return_value=SimpleNamespace(output=resolution, exit_code=0)))
    async def build(payload, inputs):
        return await _build_input(agent_id='data', ticket=ticket, payload=payload, inputs=inputs,
            work_dir=str(tmp_path), generation_backend='vllm', run=run, session=session)
    resolved = await resolve_work_order(driver=driver, blueprint=load_agent('data', input_format='freeform'),
        inp=inp, ticket=ticket, run=run, session=session, bindings={}, driver_kwargs={}, build_input=build)
    assert isinstance(resolved, DataTaskInput)
    assert not resolved.slurm_job.enabled and not resolved.device_info_path
    assert ticket.input_format == 'typed'
    assert (tmp_path/'freeform_resolution.json').is_file()
    assert (await session.scalars(select(TicketMessage))).one().body == inp.request
    result = DataResult(ticket_id='task', status='succeeded', operation='prepare_run_data', error_message='', notes='',
        training_dataset_path=inp.attachments[0], data_recipe_path=str(tmp_path/'recipe.json'), n_rows_out=1)
    assert await _bind_system_validation_artifacts(result, run=run, ticket=ticket,
        work_dir=str(tmp_path), inp=resolved, session=session) is result
    status, *_ = _extract_summary_artifact_meta(result, inp=resolved, agent_id='data', work_dir=str(tmp_path), run_id=run.id)
    assert status == 'succeeded'


@pytest.mark.asyncio
async def test_unresolved_request_never_enters_specialist_execution(database, tmp_path):
    session, run, ticket = database
    inp = request(tmp_path)
    resolution = FreeformResolution(ticket_id='task', explanation='', missing_information='Which source?')
    driver = SimpleNamespace(run_agent=AsyncMock(return_value=SimpleNamespace(output=resolution, exit_code=0)))
    build = AsyncMock()
    with pytest.raises(ValueError, match='needs information'):
        await resolve_work_order(driver=driver, blueprint=load_agent('data', input_format='freeform'), inp=inp,
            ticket=ticket, run=run, session=session, bindings={}, driver_kwargs={}, build_input=build)
    build.assert_not_awaited()
    assert ticket.input_format == 'freeform'


@pytest.mark.asyncio
async def test_runner_completes_freeform_data_through_resolution_and_typed_validation(database, tmp_path, monkeypatch):
    from zevo.engine.run import runner
    from zevo.engine.agent.drivers.base import DriverRunResult
    from zevo.contracts.data import DataRecipe
    from zevo.db.models import WorkProduct
    session, run, ticket = database
    original = request(tmp_path)
    ticket.status = 'queued'
    ticket.payload = {'request': original.request, 'attachments': original.attachments}
    await session.commit()
    calls = []
    class Driver:
        name = 'stub'
        async def run_agent(self, *, input_payload, workspace_dir, **kwargs):
            calls.append(type(input_payload))
            if isinstance(input_payload, FreeformInput):
                return DriverRunResult(output=FreeformResolution(ticket_id=ticket.id, explanation='Use attachment',
                    payload={'operation': 'prepare_run_data', 'dataset': original.attachments[0], 'training_method': 'sft'}))
            inp = input_payload
            path = Path(workspace_dir)/'dataset.jsonl'
            path.write_text(Path(inp.dataset).read_text())
            recipe = DataRecipe(dataset_name='attached source', source_identity=inp.expected_source_identity,
                source_fingerprint=inp.expected_source_fingerprint, training_method='sft', method_format='prompt_completion',
                method_ids=['passthrough_jsonl'])
            recipe_path = Path(workspace_dir)/'data_recipe.json'
            recipe_path.write_text(recipe.model_dump_json())
            return DriverRunResult(output=DataResult(ticket_id=ticket.id, status='succeeded', error_message='', notes='prepared',
                operation='prepare_run_data', training_dataset_path=str(path), data_recipe_path=str(recipe_path), n_rows_in=1, n_rows_out=1))
    monkeypatch.setattr(runner, 'get_driver', lambda _: Driver())
    result = await runner.run_ticket(session, ticket_id=ticket.id, work_dir_root=str(tmp_path/'runs'), driver_name='stub')
    assert calls == [FreeformInput, DataTaskInput], result.error_message
    assert result.status == 'succeeded', result.error_message
    products = (await session.scalars(select(WorkProduct).where(WorkProduct.ticket_id == ticket.id))).all()
    assert any(p.role == 'training_dataset' and p.meta['training_rows'] == 1 for p in products)


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [None, 1, 4])
async def test_unknown_frozen_scoring_fingerprint_version_is_not_reinterpreted(database, tmp_path, version):
    session, run, ticket = database
    run.holdout = {'test_semantic_fingerprints': ['legacy-hash']}
    if version is not None:
        run.holdout = {**run.holdout, 'semantic_fingerprint_version': version}
    with pytest.raises(ValueError, match='legacy scoring fingerprints'):
        await _bind_system_validation_artifacts(SimpleNamespace(), run=run, ticket=ticket,
            work_dir=str(tmp_path), inp=SimpleNamespace(), session=session)
