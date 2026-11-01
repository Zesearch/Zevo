"""Build the 32B task's scoring revisions, preserving immutable original scripts."""
import ast
import hashlib
import json
from pathlib import Path
import re

from repair_math_protocol import definition, repair, replace_defs

REVISION = 'olmo-task-v8'
SOURCES = {
    '98d34c255e247630ab559cf6395b0e9590eca17e04458cf475229eb3eb568681': 'livecodebench',
    'c7cfd6776c403e8e66393f5cf185e45c447dd281bb4987b90f93516b0acdf1df': 'reasoning-knowledge-math',
    '7958d4e553daf647b268c7a0e665493d8f3c41c894e5e74a1c63944a65a50f0b': 'instruction-zebra-alpaca',
}


def extract_final_choice(continuation, letters):
    """Last explicit answer field wins, even if ambiguous (then return empty).

    Never recover a choice from arbitrary discussion of an option. A standalone
    choice or boxed choice is also an explicit answer. No gold labels are used.
    """
    text = str(continuation or '').replace('**', '').strip()
    candidates = []
    for match in re.finditer(r'(?i)\b(?:(?:the|correct|final)\s+)*answer\s*(?:is\s*:?|[:=])\s*', text):
        tail = text[match.end():].split('\n', 1)[0]
        candidates.append((match.start(), tail))
    for match in re.finditer(r'\\boxed\s*\{', text):
        # Include malformed boxes as invalid final answers instead of falling back.
        end = text.find('}', match.end())
        candidates.append((match.start(), text[match.end():end] if end >= 0 else ''))
    candidate = max(candidates, default=(-1, text), key=lambda x: x[0])[1].strip()
    candidate = candidate.strip('`$').strip()
    selected = re.fullmatch(r'\(?([' + re.escape(letters) + r'])\)?\s*[.!]?', candidate, re.I)
    return {'answer': selected[1].upper() if selected else '',
            'answer_format_correct': float(bool(selected))}


def extract_last_complete_json(s):
    """Use the JSON decoder so braces/escapes within strings retain their meaning.

    A malformed prose prefix cannot hide a later valid answer object. Skip the
    entire decoded object so nested dictionaries cannot replace its root.
    """
    decoder = json.JSONDecoder()
    last = None
    index = 0
    while index < len(s):
        start = s.find('{', index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict) and isinstance(value.get('solution'), dict):
            last = value
        index = end
    return last


def patch_literal(source, predicate, transform, expected=1):
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and predicate(n.value)]
    if len(nodes) != expected:
        raise ValueError(f'Expected {expected} embedded modules, found {len(nodes)}')
    lines = source.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        if node.lineno != node.end_lineno:
            raise ValueError('Expected single-line frozen source literal')
        raw = lines[node.lineno - 1].encode()
        lines[node.lineno - 1] = (raw[:node.col_offset] + repr(transform(node.value)).encode() + raw[node.end_col_offset:]).decode()
    return ''.join(lines)


def patch_choices(source):
    lines = source.splitlines(keepends=True)
    edits = []
    for cls in ast.parse(source).body:
        if isinstance(cls, ast.ClassDef) and cls.name in {'OfficialMMLU', 'OfficialGPQA', 'OfficialAGI'}:
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_extract_answer')
            letters = 'ABCDE' if cls.name == 'OfficialAGI' else 'ABCD'
            replacement = f"    def _extract_answer(self, continuation):\n        return extract_final_choice(continuation, {letters!r})\n"
            edits.append((method.lineno, method.end_lineno, replacement))
    if len(edits) != 3:
        raise ValueError('Expected all three choice extractors')
    for start, end, value in sorted(edits, reverse=True):
        lines[start - 1:end] = [value]
    return ''.join(lines) + '\n' + definition(extract_final_choice)


def patch_instruction_errors(source):
    lines = source.splitlines(keepends=True)
    handlers = []
    for func in ast.parse(source).body:
        if isinstance(func, ast.FunctionDef) and func.name in {'test_instruction_following_strict', 'test_instruction_following_loose'}:
            handlers.extend(n for n in ast.walk(func) if isinstance(n, ast.ExceptHandler))
    if len(handlers) != 2:
        raise ValueError('Expected strict and loose exception handlers')
    for handler in sorted(handlers, key=lambda n: n.lineno, reverse=True):
        lines[handler.lineno - 1:handler.end_lineno] = [
            '        except Exception as e:\n'
            '            raise RuntimeError(f"Instruction evaluator failed: {instruction_id}") from e\n'
        ]
    return ''.join(lines)


def extract_bbh_final(continuation, category):
    """Read the required final line without deleting mathematical punctuation."""
    lines = str(continuation or '').strip().splitlines()
    if not lines:
        return ''
    match = re.fullmatch(r'\s*So the answer is\s+(.+?)\s*', lines[-1], re.I)
    if not match:
        return ''
    answer = match[1].strip()
    if answer.endswith('.'):
        answer = answer[:-1].rstrip()
    if BBH_ANSWER_REGEX.get(category, 'MC') == 'MC':
        choice = re.fullmatch(r'\(?([A-Z])\)?', answer, re.I)
        return '(' + choice[1].upper() + ')' if choice else ''
    return answer


def omega_answers_equal(prediction, reference):
    """Exact rational fast path, then the frozen five-second symbolic comparator."""
    from fractions import Fraction

    def clean(value):
        return str(value).strip().strip('$').strip().replace(r'\left', '').replace(r'\right', '')

    def number(value):
        value = re.sub(r'\s+', '', value)
        # Only scalar thousands grouping, not arbitrary commas in tuples/sets.
        if re.fullmatch(r'[+-]?\d{1,3}(,\d{3})+(?:\.\d+)?', value):
            value = value.replace(',', '')
        match = re.fullmatch(r'\\(?:d?frac)\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}', value)
        try:
            if match:
                return Fraction(match[1]) / Fraction(match[2])
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            return None

    p, g = clean(prediction), clean(reference)
    if not p or not g:
        return False
    if p == g:
        return True
    pn, gn = number(p), number(g)
    if pn is not None and gn is not None:
        return pn == gn
    # is_equiv is imported from the bundled official_math, which limits each
    # symbolic comparison to five seconds and fails loudly for missing imports.
    return bool(is_equiv(p, g))


def patch_v8_reference(source):
    source += '\n' + definition(extract_bbh_final) + '\n' + definition(omega_answers_equal)
    old = "answer = re.sub(r'(?<=\\d),(?=\\d)', '', answer)"
    if source.count(old) != 1:
        raise ValueError('Expected OMEGA grouping normalization')
    return source.replace(old, '# Preserve punctuation; numeric comparison handles thousands grouping.')


def build(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for old_hash, family in SOURCES.items():
        raw = (source_dir / (old_hash + '.py')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != old_hash:
            raise ValueError('Original evaluator hash mismatch')
        source = raw.decode()
        if family == 'reasoning-knowledge-math':
            source = repair(source, 'olmo-task-v5')
            source = patch_literal(source, lambda s: 'class OfficialGPQA:' in s, patch_choices)
            old = "return exact(task._extract_answer_regex(raw),row['target'],row['source_config']!='dyck_languages')"
            if source.count(old) != 1:
                raise ValueError('Expected BBH comparison')
            # Preserve signs, brackets, decimal points and other semantic symbols.
            source = source.replace(old, "return exact(ref.extract_bbh_final(raw,row['source_config']),row['target'])")
            source = patch_literal(source, lambda s: 'class OfficialGPQA:' in s, patch_v8_reference)
            source = source.replace("return exact(extracted['answer'],row['ground_truth'])", "return float(ref.omega_answers_equal(extracted['answer'],row['ground_truth']))")
            source = source.replace("else 'exact_match'\n result=", "else 'alias_substring_match' if benchmark=='popqa' else 'exact_match'\n result=")
            source = source.replace("'protocol_revision':'olmo-task-v6-boxed'", "'protocol_revision':'olmo-task-v8'")
        elif family == 'livecodebench':
            source = patch_literal(source, lambda s: "CODE = {" in s,
                                   lambda s: s.replace("'livecodebench_v3',", "'livecodebench_v3', 'livecodebench',"))
        else:
            source = patch_literal(source, lambda s: 'def extract_last_complete_json(' in s,
                                   lambda s: replace_defs(s, {'extract_last_complete_json': definition(extract_last_complete_json)}))
            source = patch_literal(source, lambda s: 'def test_instruction_following_strict(' in s,
                                   patch_instruction_errors, expected=2)
            source = source.replace("'protocol_revision':'olmo-task-v4'", "'protocol_revision':'olmo-task-v8'")
        compile(source, '<olmo-v8>', 'exec')
        new_hash = hashlib.sha256(source.encode()).hexdigest()
        (output_dir / (new_hash + '.py')).write_text(source)
        records.append(dict(family=family, old_sha256=old_hash, new_sha256=new_hash, protocol_revision=REVISION))
    (output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


def build_task_update(inventory_path, output_dir):
    inventory = json.loads(inventory_path.read_text())['inventory']
    task = next(t for t in inventory if t['id'] == 'b333424a-c8a5-408d-acef-005da446c376')
    manifest = json.loads((output_dir / 'manifest.json').read_text())
    replacements = {r['old_sha256']: r['new_sha256'] for r in manifest}
    changes = []
    for index, spec in enumerate(task['tests']):
        keys = ['name', 'metric', 'inference_query', 'evaluation_script', 'evaluator_sha256']
        before = {key: spec.get(key, '') for key in keys}
        after = dict(before)
        if before['evaluator_sha256'] in replacements:
            new_hash = replacements[before['evaluator_sha256']]
            after['evaluator_sha256'] = new_hash
            after['evaluation_script'] = f'/app/data/evaluators/{new_hash[:2]}/{new_hash}.py'
        baseline_action = 'rescore_saved_predictions' if after != before else 'unchanged'
        if spec['name'] == 'Reasoning · BigBenchHard':
            after['inference_query'] = before['inference_query'] + (
                '\n\nEnd your response with exactly one final line: So the answer is ANSWER.'
                '\nReplace ANSWER with only the final answer. For multiple-choice questions, '
                'use the option letter in parentheses, e.g. (B). For other questions, '
                'preserve the required numbers, signs, words, or bracket sequence. '
                'Do not write anything after this final line.'
            )
            baseline_action = 'rerun_baseline_inference_then_evaluate'
        if spec['name'] == 'Knowledge · PopQA':
            after['name'] = 'Knowledge · PopQA (alias substring match)'
            after['metric'] = 'alias_substring_match'
        changes.append(dict(index=index, before=before, after=after, baseline_action=baseline_action))
    payload = dict(task_id=task['id'], task_name=task['name'], protocol_revision=REVISION,
                   status='prepared_not_applied',
                   apply_preconditions='All before values must still match. Install hash-verified artifacts first; never mutate in-flight frozen contracts.',
                   test_sets=changes)
    (output_dir / 'task-update.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    import sys
    build(Path(sys.argv[1]), Path(sys.argv[2]))
    if len(sys.argv) > 3:
        build_task_update(Path(sys.argv[3]), Path(sys.argv[2]))
