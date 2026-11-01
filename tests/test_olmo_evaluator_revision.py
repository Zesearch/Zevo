"""Behavioral tests of the 32B task's actual frozen v8 artifacts."""
import ast
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import types
import typing

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'scripts/evaluators/olmo_v8'
RECORDS = json.loads((ROOT / 'manifest.json').read_text())


def source(family):
    record = next(r for r in RECORDS if r['family'] == family)
    return (ROOT / (record['new_sha256'] + '.py')).read_text()


def embedded(family, marker):
    return next(n.value for n in ast.walk(ast.parse(source(family))) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and marker in n.value)


def definitions(code, names, env):
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
                 or isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in names for t in n.targets)]
    exec(compile(tree, '<frozen-evaluator>', 'exec'), env)
    return env


@pytest.fixture
def reference():
    return definitions(embedded('reasoning-knowledge-math', 'class OfficialOmega:'),
                       {'OfficialMMLU', 'OfficialGPQA', 'OfficialAGI', 'extract_final_choice'}, {'re': re})


@pytest.mark.parametrize('cls', ['OfficialMMLU', 'OfficialGPQA', 'OfficialAGI'])
@pytest.mark.parametrize(('raw', 'expected'), [
    ('The answer is B.', 'B'),
    ('The answer is A. Correction: the answer is B.', 'B'),
    ('Therefore, the answer is (A).\nCorrection: the answer is (B).', 'B'),
    ('I considered (B), but cannot determine the answer.', ''),
    ('B', 'B'),
    ('(b)', 'B'),
    ('Answer: A or B', ''),
    ('The answer is A.\nFinal answer: uncertain', ''),
    (r'The answer is A. Final answer: \boxed{B}', 'B'),
    (r'\boxed{A} then \boxed{', ''),
    ('I cannot answer.', ''),
])
def test_final_choice(reference, cls, raw, expected):
    assert reference[cls]()._extract_answer(raw)['answer'] == expected


def test_option_range(reference):
    assert reference['OfficialAGI']()._extract_answer('(E)')['answer'] == 'E'
    assert reference['OfficialMMLU']()._extract_answer('(E)')['answer'] == ''


def test_bbh_actual_score_preserves_signs():
    import numpy
    family = 'reasoning-knowledge-math'
    ref = definitions(embedded(family, 'class OfficialOmega:'),
                      {'OfficialBBH', 'BBH_ANSWER_REGEX', 'extract_bbh_final'}, {'re': re})
    ref = types.SimpleNamespace(**ref)
    exact_module = embedded(family, 'def exact_match_hf_evaluate(')
    import string
    env = definitions(exact_module, {'exact_match_hf_evaluate'}, {'np': numpy, 're': re, 'string': string})
    env.update(ref=ref, metric_config=lambda _: {})
    definitions(source(family), {'instance', 'exact', 'score_one'}, env)
    for target, predicted, expected in [('42', '-42', 0), ('-42', '-42', 1), ('42', '42', 1)]:
        assert env['score_one']('bbh', {'source_config': 'multistep_arithmetic_two', 'target': target},
                                f'So the answer is {predicted}.') == expected


@pytest.fixture
def zebra():
    code = embedded('instruction-zebra-alpaca', 'def extract_last_complete_json(')
    env = {}
    exec(compile(code, '<zebra>', 'exec'), env)
    return env['ZebraLogicMetric']


@pytest.mark.parametrize('raw', [
    '{"solution":{"House 1":{"Name":"Alice"}}}',
    'Unmatched prose {\n{"solution":{"House 1":{"Name":"Alice"}}}',
    '{"solution":{"House 1":{"Name":"Alice"}},"note":"a literal {"}',
    '{"solution":{"House 1":{"Name":"Wrong"}}}\n{"solution":{"House 1":{"Name":"Alice"}}}',
    '{"solution":{"House 1":{"Name":"Alice"}}}\n{"comment":"unrelated metadata"}',
])
def test_zebra_json(zebra, raw):
    doc = {'solution_table': {'House 1': {'Name': 'Alice'}}, 'total_cells': 1}
    assert zebra.process_one_doc(None, [{'doc': doc, 'model_resps': {'continuation': raw}}])['puzzle_accuracy'] == 1


def test_zebra_invalid_answer_still_scores_zero(zebra):
    doc = {'solution_table': {'House 1': {'Name': 'Alice'}}, 'total_cells': 1}
    assert zebra.process_one_doc(None, [{'doc': doc, 'model_resps': {'continuation': 'not JSON'}}])['puzzle_accuracy'] == 0


@pytest.mark.parametrize('registry_import', ['from official_ifeval import instructions_registry', 'from ifbench import instructions_registry'])
@pytest.mark.parametrize('mode', ['strict', 'loose'])
@pytest.mark.parametrize('behavior', ['error', 'wrong', 'correct'])
def test_instruction_failure_is_not_model_failure(monkeypatch, registry_import, mode, behavior):
    code = next(n.value for n in ast.walk(ast.parse(source('instruction-zebra-alpaca')))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and registry_import in n.value and 'class InputExample:' in n.value)
    class Instruction:
        def __init__(self, *_): pass
        def build_description(self, **_): pass
        def get_instruction_args(self): return []
        def check_following(self, response):
            if behavior == 'error':
                raise LookupError('missing tokenizer resource')
            return behavior == 'correct'
    module = types.ModuleType('audit_instruction')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(dataclasses=dataclasses, Dict=typing.Dict, Optional=typing.Optional,
                           Union=typing.Union, eval_logger=logging.getLogger('test'),
                           instructions_registry=types.SimpleNamespace(INSTRUCTION_DICT={'test': Instruction}))
    env = definitions(code, {'InputExample', 'OutputExample', 'test_instruction_following_' + mode}, module.__dict__)
    inp = env['InputExample'](1, ['test'], 'prompt', [{}])
    check = env['test_instruction_following_' + mode]
    if behavior == 'error':
        with pytest.raises(RuntimeError, match='Instruction evaluator failed: test') as caught:
            check(inp, 'answer')
        assert isinstance(caught.value.__cause__, LookupError)
    else:
        assert check(inp, 'answer').follow_all_instructions is (behavior == 'correct')


@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['family'])
def test_frozen_hash(record):
    raw = (ROOT / (record['new_sha256'] + '.py')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['new_sha256']
    compile(raw, '<frozen>', 'exec')


@pytest.mark.parametrize(('prediction', 'gold', 'expected'), [
    ('0.5', r'\frac{1}{2}', True), ('1/2', '0.5', True),
    (r'\dfrac{1}{2}', '0.5', True), ('025', '25', True),
    ('1,000', '1000', True), ('-42', '42', False),
    ('1000001', '1000000', False), ('9007199254740993', '9007199254740992', False),
    ('', '42', False), ('0.5001', '0.5', False),
])
def test_omega_exact_numeric_equivalence(prediction, gold, expected):
    def unexpected_symbolic(*_):
        pytest.fail('Numeric comparisons should not need symbolic parsing')
    env = definitions(embedded('reasoning-knowledge-math', 'class OfficialOmega:'),
                      {'omega_answers_equal'}, {'re': re, 'is_equiv': unexpected_symbolic})
    assert env['omega_answers_equal'](prediction, gold) is expected


def test_omega_routes_symbolic_and_preserves_punctuation():
    calls = []
    env = definitions(embedded('reasoning-knowledge-math', 'class OfficialOmega:'),
                      {'omega_answers_equal', 'OfficialOmega', 'extract_boxed'},
                      {'re': re, 'is_equiv': lambda p, g: calls.append((p, g)) or True})
    assert env['omega_answers_equal'](r'\sqrt{4}', '2')
    assert calls == [(r'\sqrt{4}', '2')]
    assert env['OfficialOmega']()._extract_answer(r'\boxed{(1,2)}')['answer'] == '(1,2)'


def test_symbolic_comparator_has_five_second_deadline():
    # Exercise the bundled comparator's actual deadline path without requiring
    # heavyweight datasets/SymPy in the general repository test environment.
    code = embedded('reasoning-knowledge-math', 'def is_equiv(x1: str, x2: str)')
    seconds = []
    class Deadline:
        def __init__(self, seconds): self.seconds = seconds
        def __enter__(self):
            seconds.append(self.seconds)
            raise TimeoutError
        def __exit__(self, *_): pass
    env = definitions(code, {'is_equiv'}, {'timeout': Deadline, 'eval_logger': logging.getLogger('test')})
    assert env['is_equiv']('x+x', '2*x') is False
    assert seconds == [5]


@pytest.mark.parametrize(('category', 'raw', 'expected'), [
    ('date_understanding', 'work\nSo the answer is (B).', '(B)'),
    ('multistep_arithmetic_two', 'work\nSo the answer is -42.', '-42'),
    ('dyck_languages', 'work\nSo the answer is ] ) }.', '] ) }'),
    ('word_sorting', 'work\nSo the answer is apple pear.', 'apple pear'),
    ('object_counting', 'So the answer is 3.\nMore reasoning', ''),
    ('object_counting', '3', ''),
])
def test_bbh_final_line(category, raw, expected):
    env = definitions(embedded('reasoning-knowledge-math', 'class OfficialOmega:'),
                      {'extract_bbh_final', 'BBH_ANSWER_REGEX'}, {'re': re})
    assert env['extract_bbh_final'](raw, category) == expected


def test_livecodebench_alias_uses_complete_prediction(monkeypatch):
    family = 'livecodebench'
    first = types.ModuleType('extraction_v1')
    monkeypatch.setitem(sys.modules, 'extraction_v1', first)
    exec(compile(embedded(family, 'CODE = {'), '<extraction-v1>', 'exec'), first.__dict__)
    second = {}
    exec(compile(embedded(family, 'from extraction_v1 import repair as repair_v1'), '<extraction-v2>', 'exec'), second)
    for name in ['livecodebench', 'livecodebench_v3']:
        raw = '<think>reasoning</think>\n```python\nprint(42)\n```'
        assert second['repair'](name, raw, '') == ('```python\nprint(42)\n```', 'raw_code_to_original_worker')
        assert second['repair'](name, raw, 'print(7)') == ('print(7)', 'original')


def test_task_update_keeps_queries_and_evaluators_in_sync():
    update = json.loads((ROOT / 'task-update.json').read_text())
    assert len(update['test_sets']) == 16
    changes = {c['before']['name']: c for c in update['test_sets']}
    for name, change in changes.items():
        if name == 'Reasoning · BigBenchHard':
            assert 'So the answer is ANSWER.' in change['after']['inference_query']
            assert change['baseline_action'] == 'rerun_baseline_inference_then_evaluate'
        else:
            assert change['after']['inference_query'] == change['before']['inference_query']
    popqa = changes['Knowledge · PopQA']
    assert popqa['after']['metric'] == 'alias_substring_match'
    assert 'alias_substring_match' in source('reasoning-knowledge-math')
    hashes = {r['new_sha256'] for r in RECORDS}
    for change in changes.values():
        if change['after']['evaluator_sha256'] != change['before']['evaluator_sha256']:
            assert change['after']['evaluator_sha256'] in hashes
