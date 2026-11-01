"""Build versioned, standalone replacements from the audited frozen evaluators.

Usage: python scripts/evaluators/repair_math_protocol.py SOURCE_DIR OUTPUT_DIR
No database writes: installation and baseline re-scoring are separate operations.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import textwrap

REVISION = 'boxed-math-v1'
SOURCES = {
    '2eb0ba45d014190c32a5396f8131007008245f2d9491b723a6ba983b3beada24': 'math-full',
    'd19c530f3278f8940b8a79b2ca507c5819dbf256bd609ce5d5bce01a7b93e59f': 'math-full-02',
    '11749fbb9f97ea7a3c38ec48e87141bacb89d68221b662d630fbb167a924fa16': 'math-full-v2',
    'c7cfd6776c403e8e66393f5cf185e45c447dd281bb4987b90f93516b0acdf1df': 'olmo-task-v5',
}


def extract_boxed(text):
    """Last boxed command, with balanced braces; incomplete/empty final box fails.

    One token pass, including nested LaTeX and escaped literal braces. Never
    fall back to an earlier intermediate result when the final box is broken.
    """
    if not text:
        return ''
    stack = []
    last_start = None
    answer = ''
    for match in re.finditer(r'\\boxed\s*\{|\\[{}\\]|[{}]', str(text)):
        token = match.group()
        if token.startswith('\\boxed'):
            last_start = match.end()
            answer = ''
            stack.append(last_start)
        elif token == '{':
            stack.append(None)
        elif token == '}' and stack:
            start = stack.pop()
            if start is not None and start == last_start:
                answer = str(text)[start:match.start()].strip()
    return answer


def as_number(s):
    """Exact rational parsing avoids float rounding of large integer answers."""
    from fractions import Fraction
    if not s:
        return None
    s = re.sub(r'(\\times|\\cdot|\*|×)10\^\{?(-?\d+)\}?', r'e\2', s)
    if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', s):
        s = s.replace(',', '')
    s = re.sub(r'^10\^\{?(-?\d+)\}?$', r'1e\1', s)
    ratio = re.fullmatch(r'\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?', s)
    try:
        if ratio:
            return Fraction(ratio[1]) / Fraction(ratio[2])
        return Fraction(s)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _close(pn, gn):
    # An integer reference must match exactly, including a near-integer guess.
    if gn.denominator == 1:
        return pn == gn
    tolerance = globals().get('REL_TOL', globals().get('NUMERIC_TOL', 1e-6))
    scale = max(abs(pn), abs(gn)) if 'REL_TOL' in globals() else max(1, abs(gn))
    return abs(pn - gn) <= tolerance * scale


def is_correct(row_id, pred, gold):
    answer = extract_boxed(pred)
    return bool(answer) and route(row_id)(answer, gold)


def check_minerva(pred, gold):
    """Accept a numeric answer with recognized units, never an expression prefix."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    gn = as_number(g)
    p = _SCI.sub(r'e\2', p)
    # Only known unit symbols, optionally with SI prefixes and integer powers.
    unit = r'(?:[fpnumcdhkMGT]?(?:mol|Pa|Hz|eV|Wb|Bq|Gy|Sv|kat|rad|sr|[msgAKNJWCVFHTSL])|kg|atm|bar|cal|kcal|L|liter|liters)'
    power = r'(?:\^(?:\{-?\d+\}|-?\d+)|[23])?'
    units = unit + power + r'(?:(?:[*/·]|\\cdot)?' + unit + power + r')*'
    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(' + units + r')', p)
    if gn is not None and match:
        return _close(as_number(match[1]), gn)
    return check_general(pred, gold)


class OfficialOmega:
    def _extract_answer(self, continuation):
        answer = extract_boxed(continuation)
        # Keep numeric grouping normalization, but only on the extracted answer.
        answer = re.sub(r'(?<=\d),(?=\d)', '', answer)
        return {'answer': answer, 'answer_format_correct': float(bool(answer))}


class OfficialMath:
    def extract_answers(self, results):
        raw = results[0] if isinstance(results, list) else results
        answer = extract_boxed(raw)
        return [normalize_final_answer(answer)] if answer else []

    def process_results(self, doc, results):
        answers = self.extract_answers(results)
        score = int(bool(answers) and (
            is_equiv(answers[0], doc['answer'])
            or hendrycks_is_equiv(answers[0], doc['answer'])
        ))
        return {'exact_match': score, 'exact_match_flex': score,
                'model_answer': answers[0] if answers else ''}


def replace_defs(source, replacements):
    lines = source.splitlines(keepends=True)
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    found = {n.name for n in nodes}
    if not replacements.keys() <= found:
        raise ValueError(f'Missing definitions: {replacements.keys() - found}')
    for node in reversed(nodes):
        if node.name in replacements:
            lines[node.lineno - 1:node.end_lineno] = [replacements[node.name].rstrip() + '\n']
    return ''.join(lines)


def definition(obj):
    return textwrap.dedent(inspect.getsource(obj))


def repair(source, family):
    if family == 'olmo-task-v5':
        # Replace only the embedded benchmark reference module. All unrelated
        # bundled benchmark implementations and configuration remain byte-identical.
        candidates = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant)
                      and isinstance(n.value, str) and 'class OfficialOmega:' in n.value]
        if len(candidates) != 1:
            raise ValueError('Expected one embedded reference module')
        node = candidates[0]
        embedded = replace_defs(node.value, {c.__name__: definition(c) for c in (OfficialOmega, OfficialMath)})
        embedded += '\n' + definition(extract_boxed)
        lines = source.splitlines(keepends=True)
        if node.lineno != node.end_lineno:
            raise ValueError('Expected a one-line embedded source literal')
        line = lines[node.lineno - 1].encode('utf-8')
        lines[node.lineno - 1] = (line[:node.col_offset] + repr(embedded).encode() + line[node.end_col_offset:]).decode()
        source = ''.join(lines).replace("'protocol_revision':'olmo-task-v5'", "'protocol_revision':'olmo-task-v6-boxed'")
    else:
        funcs = [extract_boxed, as_number, is_correct]
        if family == 'math-full-v2':
            funcs += [_close, check_minerva]
        source = replace_defs(source, {f.__name__: definition(f) for f in funcs})
        if family != 'math-full-v2':
            source = source.replace('return abs(pn - gn) <= NUMERIC_TOL * max(1.0, abs(gn))', 'return _close(pn, gn)')
            # Insert before main() is invoked, not after the __main__ block.
            source = source.replace('def check_integer(', definition(_close) + '\n\ndef check_integer(', 1)
    compile(source, '<repaired-evaluator>', 'exec')
    return '# Evaluation protocol revision: ' + REVISION + '\n' + source


def build(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for old_hash, family in SOURCES.items():
        raw = (source_dir / (old_hash + '.py')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != old_hash:
            raise ValueError(f'Source hash mismatch: {family}')
        patched = repair(raw.decode(), family).encode()
        new_hash = hashlib.sha256(patched).hexdigest()
        (output_dir / (new_hash + '.py')).write_bytes(patched)
        records.append(dict(family=family, old_sha256=old_hash, new_sha256=new_hash,
                            protocol_revision=REVISION))
    (output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


if __name__ == '__main__':
    import sys
    build(Path(sys.argv[1]), Path(sys.argv[2]))
