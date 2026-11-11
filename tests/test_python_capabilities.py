import pytest
from zevo.engine.run.python_capabilities import calls_capability, gpu_launch_problem
from zevo.engine.run.runner import _referenced_paths


@pytest.mark.parametrize('source, expected', [
    ('from zevo_stage_resume import run_batches\nf = run_batches\nlist(f(rows, fn))', True),
    ('import zevo_stage_resume as m\nf = m.run_batches\nf(rows, fn)', True),
    ('from zevo_stage_resume import run_batches\nif False:\n    run_batches()', False),
    ('import zevo_stage_resume\nwhile False:\n    zevo_stage_resume.run_batches()', False),
    ('# zevo_stage_resume.run_batches()\nprint("done")', False),
])
def test_call_aliases_and_dead_code(tmp_path, source, expected):
    p = tmp_path/'prepare.py'; p.write_text(source)
    assert calls_capability(p, {'zevo_stage_resume.run_batches'}) is expected


def test_local_helper_module_is_supported(tmp_path):
    (tmp_path/'worker.py').write_text('from zevo_stage_resume import run_batches\ndef execute():\n    return run_batches(rows, fn)\n')
    main = tmp_path/'prepare.py'; main.write_text('from worker import execute\nexecute()\n')
    assert calls_capability(main, {'zevo_stage_resume.run_batches'})


def test_gpu_runtime_python_api_and_cpu_preparation(tmp_path):
    main = tmp_path/'train.py'
    main.write_text('import zevo_gpu_runtime as g\nlaunch = g.run\nlaunch(["python", "worker.py"])\n')
    assert gpu_launch_problem('python train.py', main) == ''
    main.write_text('# g.run()\nprint("nothing")\n')
    assert gpu_launch_problem('python train.py', main)
    cpu = tmp_path/'prepare_data.py'; cpu.write_text('print("CPU curation")\n')
    assert gpu_launch_problem('python prepare_data.py', cpu) == ''


@pytest.mark.parametrize('command, hit', [
    ('# Do not read /private/run-xyz/eval.jsonl\nprint("safe")', False),
    ('echo "Do not read /private/run-xyz/eval.jsonl"', False),
    ('cat /private/run-xyz/eval.jsonl', True),
    ('python -c \'print("/private/run-xyz/eval.jsonl")\'', False),
    ('python -c \'open("/private/run-xyz/eval.jsonl").read()\'', True),
    ('python - <<\'PY\'\n# /private/run-xyz/eval.jsonl\nprint("safe")\nPY\n', False),
])
def test_path_mentions_are_not_file_access(command, hit):
    result = _referenced_paths(['/private/run-xyz/eval.jsonl'], [{'type': 'tool_call', 'payload': {'input': {'command': command}}}])
    assert bool(result) is hit


@pytest.mark.parametrize('command', [
    'rg "/private/run-xyz/eval.jsonl" /my/notes.txt',
    'grep -e "/private/run-xyz/eval.jsonl" /my/notes.txt',
    'sed -n "/private/run-xyz/eval.jsonl/p" /my/notes.txt',
])
def test_path_in_search_pattern_is_not_access(command):
    assert _referenced_paths(['/private/run-xyz/eval.jsonl'], [
        {'type': 'tool_call', 'payload': {'command': command, 'name': 'run_bash'}}]) == []


def test_codex_and_claude_read_events_are_checked():
    path = '/private/run-xyz/eval.jsonl'
    for payload in [{'name': 'run_bash', 'command': 'head ' + path},
                    {'name': 'Read', 'input': {'file_path': path}}]:
        assert _referenced_paths([path], [{'type': 'tool_call', 'payload': payload}]) == ['eval.jsonl']


def test_callback_class_passed_to_trainer_is_valid(tmp_path):
    path = tmp_path/'train.py'
    path.write_text('from zevo_train_telemetry import ZevoTrainerTelemetryCallback as C\nTrainer(callbacks=[C])\n')
    assert calls_capability(path, {'zevo_train_telemetry.ZevoTrainerTelemetryCallback'})
