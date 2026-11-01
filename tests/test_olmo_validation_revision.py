"""Exercise the frozen Validation revision and its remote transport contract."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from zevo.engine import remote_evaluation as remote
from tests.test_remote_evaluation import work

ROOT = Path(__file__).resolve().parents[1] / 'scripts/evaluators/olmo_validation_v2'
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
SCRIPT = ROOT / (MANIFEST['new_sha256'] + '.py')
SOURCE = SCRIPT.read_text()


def defs(source, names, env):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(tree, '<frozen-validation>', 'exec'), env)
    return env


def reference_source():
    return next(ast.literal_eval(n.value) for n in ast.parse(SOURCE).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'REFERENCE_SOURCE' for t in n.targets))


def test_dependency_is_embedded_and_hash_verified():
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == MANIFEST['new_sha256']
    assert hashlib.sha256(reference_source().encode()).hexdigest() == MANIFEST['embedded_reference_sha256']
    env = defs(SOURCE, {'load_reference'}, {'__file__': str(SCRIPT), 'REFERENCE_SOURCE': 'answer = 42'})
    assert env['load_reference']().answer == 42
    assignments = {t.id for n in ast.parse(SOURCE).body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    assert 'REFERENCE' not in assignments
    assert 'RUNTIME' in assignments


@pytest.mark.parametrize(('raw', 'allowed', 'expected'), [
    ('Answer: A. Correction: Answer: B.', 'ABCD', 'B'),
    ('Answer: A\nCorrection: Answer: B', 'ABCD', 'B'),
    ('Answer: 1. Correction: Answer: 3.', '1234', '3'),
    ('Answer: E', 'ABCDE', 'E'),
    ('Answer: D', 'ABC', ''),
    ('Answer: A or B', 'ABCD', ''),
    ('I considered (B), but cannot answer.', 'ABCD', ''),
    (r'Answer: A then \boxed{B}', 'ABCD', 'B'),
    (r'Answer: A then \boxed{', 'ABCD', ''),
])
def test_choice_extraction(raw, allowed, expected):
    env = defs(SOURCE, {'label', 'extract_final_choice'}, {'re': re})
    assert env['label'](raw, allowed) == expected


@pytest.mark.parametrize(('raw', 'expected'), [
    ('42', []), (r'\boxed{42x', []),
    (r'\boxed{42}', ['42']),
    (r'\boxed{\frac{1}{2}}', [r'\frac{1}{2}']),
    (r'\boxed{42} then \boxed{', []),
])
def test_embedded_math_requires_complete_box(raw, expected):
    reference = reference_source()
    embedded = next(n.value for n in ast.walk(ast.parse(reference)) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and 'class OfficialMath:' in n.value)
    env = defs(embedded, {'extract_boxed', 'OfficialMath'}, {'re': re, 'normalize_final_answer': lambda s: s})
    assert env['OfficialMath']().extract_answers([raw]) == expected


def test_actual_cli_from_relocated_validation_script(tmp_path):
    script = tmp_path / 'compute/scorer.py'
    script.parent.mkdir(); script.write_bytes(SCRIPT.read_bytes())
    prediction = tmp_path / 'prediction.csv'
    gold = tmp_path / 'gold.csv'
    with prediction.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['id', 'prediction', 'complete_prediction'])
        writer.writerows([['1', 'A', 'Answer: A. Correction: Answer: B.'],
                          ['2', '1', 'Answer: 1. Correction: Answer: 3.'],
                          ['3', 'B', 'I considered (B), but cannot answer.']])
    with gold.open('w', newline='') as handle:
        writer = csv.writer(handle);writer.writerow(['id', 'scoring_kind', 'allowed_labels', 'answer'])
        writer.writerows([['1', 'mc', 'ABCD', 'B'], ['2', 'mc', '1234', '3'], ['3', 'mc', 'ABCD', 'B']])
    output = tmp_path / 'metrics.json'
    # Simulate arrival inside the declared runtime; choice scoring itself needs
    # only stdlib. No original /app/data/evaluators directory is available here.
    command = ('import runpy,sys; sys.prefix=sys.argv.pop(1); '
               'path=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(path,run_name="__main__")')
    result = subprocess.run([sys.executable, '-c', command, MANIFEST['runtime'],
                             str(script), str(prediction), str(gold), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.read_text())
    assert metrics['score'] == 2 / 3
    assert metrics['protocol_revision'] == MANIFEST['protocol_revision']


def test_remote_bundle_preserves_script_without_rebuilding_environment(tmp_path):
    inp = work(tmp_path)
    inp.evaluation_script = str(SCRIPT);inp.evaluator_sha256 = MANIFEST['new_sha256']
    staging = tmp_path / 'staging';staging.mkdir()
    remote.make_bundle(inp, staging, '/compute/work')
    bundle = staging / 'bundle'
    assert not (bundle / 'runtimes.json').exists()
    assert not (bundle / 'compat').exists()
    copied = bundle / 'inputs/0/evaluation_script' / SCRIPT.name
    assert copied.read_bytes() == SCRIPT.read_bytes()
    inventory = json.loads((bundle / 'manifest.json').read_text())
    assert inventory[str(copied.relative_to(bundle))] == MANIFEST['new_sha256']


def test_all_twelve_contracts_update_without_query_changes():
    update = json.loads((ROOT / 'setting-update.json').read_text())
    changes = update['settings'][0]['validation_sets']
    assert len(changes) == 12
    for item in changes:
        assert item['after']['inference_query'] == item['before']['inference_query']
        assert item['after']['evaluator_sha256'] == MANIFEST['new_sha256']
        assert item['baseline_action'] == 'rescore_saved_predictions'
