"""Regression tests against the actual frozen replacement artifacts."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'scripts/evaluators/boxed_math_v1'
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
MATH = [r for r in MANIFEST if r['family'] != 'olmo-task-v5']


@pytest.fixture(params=MATH, ids=lambda r: r['family'])
def scorer(request):
    return runpy.run_path(str(ROOT / (request.param['new_sha256'] + '.py')))


@pytest.mark.parametrize(('prediction', 'gold', 'expected'), [
    (r'\boxed{42x', '42', False),
    (r'\boxed{1} then \boxed{2x', '2', False),
    (r'\boxed{42} then \boxed{', '42', False),
    (r'\boxed{42} then \boxed{}', '42', False),
    ('42', '42', False),
    ('', '42', False),
    (r'work \boxed{1} final \boxed{42}', '42', True),
    (r'\boxed{1000001}', '1000000', False),
    (r'\boxed{1000000.1}', '1000000', False),
    (r'\boxed{9007199254740993}', '9007199254740992', False),
    (r'\boxed{025}', '25', True),
    (r'\boxed{1,234}', '1234', True),
    (r'\boxed{\frac{1}{2}}', '0.5', True),
    (r'\boxed{0.5}', r'\frac{1}{2}', True),
])
def test_math_grading(scorer, prediction, gold, expected):
    assert scorer['is_correct']('gsm8k-0001', prediction, gold) is expected


def test_balanced_extraction(scorer):
    extract = scorer['extract_boxed']
    assert extract(r'\boxed{\frac{1}{\sqrt{2}}}') == r'\frac{1}{\sqrt{2}}'
    assert extract(r'\boxed{\{1,2\}}') == r'\{1,2\}'
    assert extract(r'\boxed{1} and \boxed{2}') == '2'


@pytest.mark.parametrize(('prediction', 'gold', 'expected'), [
    (r'\boxed{4/5}', '4', False),
    (r'\boxed{41.8+100}', '41.8', False),
    (r'\boxed{41.8\text{ kJ}}', '41.8', True),
    (r'\boxed{4.5\times10^{33}}', '4.5e33', True),
    (r'\boxed{9.8\text{ m/s^2}}', '9.8', True),
])
def test_minerva(prediction, gold, expected):
    record = next(r for r in MATH if r['family'] == 'math-full-v2')
    module = runpy.run_path(str(ROOT / (record['new_sha256'] + '.py')))
    assert module['is_correct']('minerva-0001', prediction, gold) is expected


def olmo_reference():
    record = next(r for r in MANIFEST if r['family'] == 'olmo-task-v5')
    source = (ROOT / (record['new_sha256'] + '.py')).read_text()
    embedded = next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and 'class OfficialOmega:' in n.value)
    tree = ast.parse(embedded)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in {'extract_boxed', 'OfficialOmega', 'OfficialMath'}]
    namespace = {'re': re, 'normalize_final_answer': lambda x: x}
    exec(compile(tree, '<reference>', 'exec'), namespace)
    return namespace


def test_omega_nested_and_missing_answer():
    module = olmo_reference()
    omega = module['OfficialOmega']()
    assert omega._extract_answer(r'Therefore, the final answer is \boxed{\frac{1}{2}}.')['answer'] == r'\frac{1}{2}'
    assert omega._extract_answer('The answer is 42')['answer'] == ''
    assert omega._extract_answer(r'\boxed{42x')['answer'] == ''
    assert omega._extract_answer(r'\boxed{1,234,567}')['answer'] == '1234567'
    math = module['OfficialMath']()
    assert math.extract_answers(['42']) == []
    assert math.extract_answers([r'\boxed{1} then \boxed{2}']) == ['2']
    assert math.process_results({'answer': '42'}, ['42'])['exact_match_flex'] == 0


def test_omega_long_missing_box_is_fast():
    omega = olmo_reference()['OfficialOmega']()
    start = time.monotonic()
    for _ in range(100):
        assert omega._extract_answer('reasoning without answer ' * 1000)['answer'] == ''
    # Original regex fallback took seconds for ONE similarly sized prediction.
    assert time.monotonic() - start < 3


@pytest.mark.parametrize('record', MANIFEST, ids=lambda r: r['family'])
def test_artifact_hash(record):
    raw = (ROOT / (record['new_sha256'] + '.py')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['new_sha256']
    compile(raw, '<evaluator>', 'exec')


@pytest.mark.parametrize('record', MATH, ids=lambda r: r['family'])
def test_real_cli(record, tmp_path):
    predictions = tmp_path / 'predictions.csv'
    with predictions.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        writer.writerows([['gsm8k-1', r'\boxed{42}'], ['gsm8k-2', '42'],
                         ['gsm8k-3', r'\boxed{1000001}'], ['gsm8k-4', r'\boxed{025}']])
    scoring = tmp_path / 'scoring.jsonl'
    scoring.write_text('\n'.join(json.dumps({'id': f'gsm8k-{i}', 'answer': gold})
                                 for i, gold in enumerate(['42', '42', '1000000', '25'], 1)))
    metrics = tmp_path / 'metrics.json'
    result = subprocess.run([sys.executable, str(ROOT / (record['new_sha256'] + '.py')),
                             str(predictions), str(scoring), str(metrics)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(metrics.read_text())['score'] == 0.5
