import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('stage_resume', Path(__file__).parents[1] / 'playbook/runners/stage_resume.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_data_restart_reuses_empty_and_nonempty_batches_and_invalidates_identity(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    calls = []
    interrupted = True
    def process(rows, start):
        calls.append(start)
        if interrupted and start == 4:
            raise RuntimeError('worker lost')
        return {'accepted': [] if start == 0 else rows, 'processed': len(rows)}
    def run(identity):
        return list(resume.run_batches(list(range(6)), process, root=tmp_path,
                    ticket='data-r1-001', identity=identity, batch_size=2))
    with pytest.raises(RuntimeError, match='worker lost'):
        run({'recipe': 'a', 'source': 'sha'})
    interrupted = False
    result = run({'recipe': 'a', 'source': 'sha'})
    assert calls == [0, 2, 4, 4]
    assert result == [{'accepted': [], 'processed': 2}, {'accepted': [2, 3], 'processed': 2},
                      {'accepted': [4, 5], 'processed': 2}]
    run({'recipe': 'b', 'source': 'sha'})
    assert calls[-3:] == [0, 2, 4]


def test_corrupt_batch_is_not_reused(tmp_path):
    store = resume.BatchStore(tmp_path, ticket='data-r1-001', identity={'revision': 1})
    with store.batch('one') as key:
        store.commit(key, [1, 2])
        path = store.root / (key + '.json')
        envelope = json.loads(path.read_text())
        envelope['value'] = [9]
        path.write_text(json.dumps(envelope))
        with pytest.raises(ValueError, match='invalid saved batch'):
            store.load(key)


def test_actual_allocation_end_overrides_long_stage_deadline(monkeypatch, capsys):
    monkeypatch.setenv('SLURM_JOB_END_TIME', '1050')
    monkeypatch.setattr(resume.time, 'time', lambda: 1000)
    window = resume.AllocationWindow(deadline=50000, reserve_seconds=60)
    with pytest.raises(SystemExit) as exc:
        window.check('data-r1-001')
    assert exc.value.code == 75
    assert 'waiting_for_allocation' in capsys.readouterr().out


def test_deadline_preserves_preceding_batch(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    clock = [1000]
    monkeypatch.setattr(resume.time, 'time', lambda: clock[0])
    def process(rows, start):
        clock[0] = 1100
        return rows
    it = resume.run_batches([1, 2], process, root=tmp_path, ticket='data-r1-001',
                            identity={'recipe': 1}, batch_size=1,
                            window=resume.AllocationWindow(deadline=1150, reserve_seconds=60))
    assert next(it) == [1]
    with pytest.raises(SystemExit) as exc:
        next(it)
    assert exc.value.code == 75
    calls = []
    values = list(resume.run_batches([1, 2], lambda rows, start: calls.append(start) or rows,
                  root=tmp_path, ticket='data-r1-001', identity={'recipe': 1}, batch_size=1,
                  window=resume.AllocationWindow(deadline=10000)))
    assert values == [[1], [2]] and calls == [1]


@pytest.mark.parametrize('stage', ['data', 'inference'])
@pytest.mark.parametrize('state,code,allowed', [('TIMEOUT', '0:0', True), ('PREEMPTED', '0:0', True),
    ('FAILED', '75:0', True), ('FAILED', '1:0', False), ('COMPLETED', '75:0', False),
    ('CANCELLED', '0:15', False)])
def test_retry_only_after_allocation_interruption(stage, state, code, allowed):
    from zevo.engine.run.runner import resumable_stage_interruption
    assert resumable_stage_interruption(SimpleNamespace(agent_id=stage),
        SimpleNamespace(meta={'scheduler_state': state, 'scheduler_exit_code': code})) is allowed


@pytest.mark.parametrize('source,valid', [
    ('# zevo_stage_resume.run_batches(rows)\nprint("done")', False),
    ('from zevo_stage_resume import run_batches', False),
    ('from zevo_stage_resume import run_batches as batches\nlist(batches(rows, fn))', True),
    ('import zevo_stage_resume as resume\nlist(resume.run_batches(rows, fn))', True),
])
def test_generated_workload_requires_real_resume_helper_call(tmp_path, source, valid):
    from zevo.engine.run.runner import _python_stage_resume_problem
    path = tmp_path / 'prepare_data.py'
    path.write_text(source)
    assert (_python_stage_resume_problem(str(path)) == '') is valid


def test_missing_allocation_end_fails_closed(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURM_JOB_END_TIME', 'NaN')
    monkeypatch.setattr(resume.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='EndTime=Unknown'))
    with pytest.raises(ValueError, match='cannot determine allocation end'):
        resume.AllocationWindow()
