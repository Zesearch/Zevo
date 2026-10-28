import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tests.test_parallel_inference import _suite, _write_csv, HELPER

ADAPTER = '''
import csv, json, os
from pathlib import Path

def record(kind, **kwargs):
    with open(os.environ['AUDIT'], 'a') as f:
        f.write(json.dumps(dict(kind=kind, pid=os.getpid(), **kwargs)) + '\\n')

def load_engine(model, config):
    record('load', model=model)
    return object()

def predict_member(engine, member, config, *, progress):
    calibration = 'resident-calibration' in member['output']
    record('calibrate' if calibration else 'generate', name=member['name'])
    rows = list(csv.DictReader(open(member['questions'])))
    fields = ['id', 'prediction', 'complete_prediction']
    Path(member['output']).parent.mkdir(parents=True, exist_ok=True)
    with open(member['output'], 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(id=r['id'], prediction='A', complete_prediction='The answer is A.') for r in rows)
    limited = config['measurement']['max_new_tokens'] < 32 or os.environ.get('FAIL_MEMBER') == member['name']
    records = [dict(row_index=i, request_index=i, turn=1, finish_reason='length' if limited else 'stop',
                    stop_reason=None, generated_tokens=16) for i in range(len(rows))]
    Path(member['diagnostics']).write_text(json.dumps({'schema_version':1,'records':records}))
    progress(len(rows), len(rows))
    return dict(name=member['name'], n_rows=len(rows), n_requests=len(rows), n_unparseable=0)

def close_engine(engine):
    record('close')
'''


def run_resident(tmp_path, counts, *, fail_member='', incompatible=False, calibrate=True):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    runner = runtime / 'zevo_parallel_inference.py'
    shutil.copyfile(HELPER, runner)
    shutil.copyfile(HELPER.with_name("gpu_runtime.py"), runtime / "zevo_gpu_runtime.py")
    shutil.copyfile(HELPER.parents[2] / 'src/zevo/contracts/length_calibration.py',
                    runtime / 'zevo_length_calibration.py')
    adapter = tmp_path / 'predict.py'
    adapter.write_text(ADAPTER)
    suite = _suite(tmp_path, counts)
    members = json.loads(suite.read_text())['members']
    for member in members:
        cfg = {'base_model': 'test', 'measurement': {'max_new_tokens': 16},
               'implementation_config': {'llm_kwargs': {'tensor_parallel_size': 1}}}
        Path(member['config']).write_text(json.dumps(cfg))
        member['calibration_configs'] = []
        for n in [16, 32]:
            cfg['measurement']['max_new_tokens'] = n
            if incompatible and n == 32:
                cfg['implementation_config']['llm_kwargs']['max_model_len'] = 999
            path = Path(member['config']).with_name(f'candidate-{n}.yaml')
            path.write_text(json.dumps(cfg))
            member['calibration_configs'].append(str(path))
        _write_csv(Path(member['sample_submission']), ['id','prediction','complete_prediction'], [])
    suite.write_text(json.dumps({'members': members}))
    audit = tmp_path / 'audit.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', TMPDIR='/tmp', AUDIT=str(audit), FAIL_MEMBER=fail_member)
    command = [sys.executable, str(runner), '--resident', '--predict', str(adapter), '--model', 'test',
               '--suite', str(suite), '--summary', str(tmp_path / 'summary.json'), '--ticket-id', 'test',
               '--allocated-gpus', '2', '--gpus-per-worker', '1']
    if calibrate:
        command.append('--calibrate')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
    return result, events, members


def test_calibration_and_generation_share_exact_loaded_instances(tmp_path):
    result, events, members = run_resident(tmp_path, [160, 80])
    assert result.returncode == 0, result.stdout + result.stderr
    loads = [e for e in events if e['kind'] == 'load']
    assert len(loads) == 2
    assert len({e['pid'] for e in loads}) == 2
    assert {e['pid'] for e in events if e['kind']=='generate'} <= {e['pid'] for e in loads}
    assert len([e for e in events if e['kind']=='calibrate']) == 4
    assert len([e for e in events if e['kind']=='close']) == 2
    first_generate = next(i for i,e in enumerate(events) if e['kind']=='generate')
    assert all(e['kind'] != 'calibrate' for e in events[first_generate:])
    for member, count in zip(members, [160, 80]):
        assert len(list(__import__('csv').DictReader(open(member['output'])))) == count
        assert json.loads(Path(member['config']).read_text())['measurement']['max_new_tokens'] == 32
        assert Path(member['config']).with_name('length_calibration.json').exists()


def test_one_failed_calibration_blocks_all_measured_generation(tmp_path):
    result, events, _ = run_resident(tmp_path, [80, 80], fail_member='benchmark 1')
    assert result.returncode != 0
    assert not any(e['kind']=='generate' for e in events)
    assert 'length calibration exhausted' in result.stderr


def test_incompatible_engine_configs_fail_before_loading(tmp_path):
    result, events, _ = run_resident(tmp_path, [80], incompatible=True)
    assert result.returncode != 0
    assert not events
    assert 'engine initialization' in result.stderr


def test_frozen_reuse_does_not_recalibrate(tmp_path):
    result, events, _ = run_resident(tmp_path, [80], calibrate=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(e['kind']=='generate' for e in events)
    assert not any(e['kind']=='calibrate' for e in events)
