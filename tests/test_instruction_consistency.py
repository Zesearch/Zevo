"""Effective instructions and narrative contracts match activation ownership."""
import pytest
from pydantic import ValidationError
from zevo.contracts._base import AgentTaskInput
from zevo.contracts.orchestrator import JournalNarrativeEntry, PatchRunRequest
from zevo.engine.agent.loader import load_agent
from zevo.engine.agent.drivers._prompt import _activation_block, _do_the_work_block


def test_pipeline_method_selection_has_one_owner_and_guides_are_accessible():
    instructions = load_agent('train').instructions
    assert 'execute the Orchestrator-selected method' in instructions
    assert 'select every unpinned `training_method` yourself' not in instructions
    assert 'Operational guide (read relevant sections' in instructions
    assert 'platform.md' in instructions
    assert len(instructions) < 55000


@pytest.mark.parametrize('mode,expected,absent', [
    ('per_run', 'same healthy retained allocation', 'fresh allocation'),
    ('per_stage', 'same healthy retained allocation', 'fresh allocation'),
    ('per_submission', 'fresh allocation', 'same healthy retained allocation'),
])
def test_only_applicable_gpu_lifetime_is_injected(mode, expected, absent):
    text = _activation_block(AgentTaskInput(ticket_id='t', activation_contract={'gpu_allocation_mode': mode}))
    assert expected in text and absent not in text


@pytest.mark.parametrize('plane', ['local', 'remote'])
def test_data_plane_is_explicit(plane):
    text = _activation_block(AgentTaskInput(ticket_id='t', activation_contract={'data_plane': plane}))
    assert ('locally in work_dir' in text) == (plane == 'local')
    assert ('assigned remote plane' in text) == (plane == 'remote')
    assert 'deferred' in _do_the_work_block()
    assert 'Never end your turn early' not in _do_the_work_block()


@pytest.mark.parametrize('result,analysis', [
    ('验证集得分提升到 0.75。', '未观察到幻觉。'),
    ('Measured gain of 0.05.', 'No hallucinations were observed.'),
    ('Stable performance.', 'Overfitting was ruled out by controlled experiments.'),
])
def test_narrative_language_is_not_a_keyword_gate(result, analysis):
    entry = JournalNarrativeEntry(iteration=1, source='trained', action='Refine',
        result=result, analysis=analysis, next='Retain the measured champion.')
    assert entry.measurement_split == 'validation'
    PatchRunRequest(summary=analysis, history_entry=entry)
    with pytest.raises(ValidationError):
        JournalNarrativeEntry.model_validate({**entry.model_dump(), 'measurement_split': 'test'})


@pytest.mark.parametrize('agent', ['train', 'orchestrator', 'registry'])
@pytest.mark.parametrize('layout', ['claude', 'codex'])
def test_operational_guides_travel_with_the_ticket_workspace(tmp_path, agent, layout):
    from zevo.engine.agent.skills import stage_skills
    from zevo.engine.agent.loader import AGENTS_DIR
    stage_skills(agent, str(tmp_path), layout=layout)
    staged = tmp_path / '.zevo-guides' / 'platform.md'
    assert staged.read_bytes() == (AGENTS_DIR / agent / 'platform.md').read_bytes()
    # Every activation refreshes a damaged/stale copy from the shipped guide.
    staged.write_text('stale procedure')
    stage_skills(agent, str(tmp_path), layout=layout)
    assert staged.read_bytes() == (AGENTS_DIR / agent / 'platform.md').read_bytes()
    assert '.zevo-guides/platform.md' in load_agent(agent).instructions


@pytest.mark.parametrize('provider', ['cluster', 'cloud', 'instance'])
def test_gpu_instructions_follow_provider(provider):
    text = _activation_block(AgentTaskInput(ticket_id='t', activation_contract={
        'gpu_allocation_mode': 'per_run', 'gpu_provider': provider,
    }))
    assert ('JOBID' in text) == (provider == 'cluster')
    assert ('parent allocation' in text) == (provider == 'cluster')
    assert 'same healthy retained allocation' in text
