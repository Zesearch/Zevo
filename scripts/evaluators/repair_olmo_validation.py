"""Build a self-contained, immutable revision of the 32B Validation scorer."""
import hashlib
import json
from pathlib import Path

from repair_math_protocol import definition, replace_defs
from repair_olmo_protocol import extract_final_choice

ORIGINAL = 'eec6cbd60a40d1d97f3a714f7758994fda4b899904fc6eb48c2c4c3810bb1332'
REVISION = 'validation-v2-boxed-final-choice'
RUNTIME = '/app/data/evaluation-runtimes/olmes-5a51f502-alpaca-cd543a1'


def label(text, allowed):
    return extract_final_choice(text, allowed)['answer']


def load_reference():
    # The complete, frozen dependency travels with this evaluator, so no
    # /app/data/evaluators/... reference is needed on the compute node.
    import types
    module = types.ModuleType('official_validation_reference')
    module.__file__ = __file__
    exec(compile(REFERENCE_SOURCE, '<frozen-validation-reference>', 'exec'), module.__dict__)
    return module


def build(source_dir, reference_dir, output_dir):
    original = (source_dir / (ORIGINAL + '.py')).read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL:
        raise ValueError('Original Validation evaluator hash mismatch')
    manifest = json.loads((reference_dir / 'manifest.json').read_text())
    reference_hash = next(r['new_sha256'] for r in manifest if r['family'] == 'reasoning-knowledge-math')
    reference = (reference_dir / (reference_hash + '.py')).read_bytes()
    if hashlib.sha256(reference).hexdigest() != reference_hash:
        raise ValueError('Frozen math dependency hash mismatch')
    source = original.decode()
    lines = source.splitlines(keepends=True)
    if not lines[1].startswith('if sys.prefix != '):
        raise ValueError('Unexpected Validation interpreter bootstrap')
    lines[1] = (f'RUNTIME = {RUNTIME!r}\n'
                "if sys.prefix != RUNTIME: os.execv(RUNTIME+'/bin/python', [RUNTIME+'/bin/python', *sys.argv])\n")
    source = ''.join(lines)
    source = replace_defs(source, {'label': definition(label), 'load_reference': definition(load_reference)})
    start = source.index('REFERENCE=')
    end = source.index('\n', start)
    source = (source[:start] + 'REFERENCE_SOURCE = ' + repr(reference.decode()) + '\n'
              + definition(extract_final_choice) + source[end:])
    source = source.replace("'protocol_revision':'validation-current-label-fix'", f"'protocol_revision':{REVISION!r}")
    compile(source, '<validation-v2>', 'exec')
    digest = hashlib.sha256(source.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / (digest + '.py')).write_text(source)
    (output_dir / 'manifest.json').write_text(json.dumps({
        'old_sha256': ORIGINAL, 'new_sha256': digest, 'protocol_revision': REVISION,
        'embedded_reference_sha256': reference_hash, 'runtime': RUNTIME,
    }, indent=2) + '\n')
    audit = json.loads((source_dir / 'audit.json').read_text())
    settings = []
    for setting in audit['settings']:
        changes = []
        for index, spec in enumerate(setting['validation_sets']):
            if spec['evaluator_sha256'] != ORIGINAL:
                raise ValueError('Unexpected Validation contract; refresh audit before rebuilding')
            before = {k: spec[k] for k in ('name', 'inference_query', 'evaluation_script', 'evaluator_sha256')}
            after = {**before, 'evaluator_sha256': digest,
                     'evaluation_script': f'/app/data/evaluators/{digest[:2]}/{digest}.py'}
            changes.append(dict(index=index, before=before, after=after,
                                baseline_action='rescore_saved_predictions'))
        settings.append(dict(setting_id=setting['id'], setting_name=setting['name'], validation_sets=changes))
    (output_dir / 'setting-update.json').write_text(json.dumps({
        'task_id': 'b333424a-c8a5-408d-acef-005da446c376', 'status': 'prepared_not_applied',
        'preconditions': 'All before-values must match. Install verified script before updating settings. Preserve in-flight frozen contracts.',
        'settings': settings,
    }, indent=2) + '\n')


if __name__ == '__main__':
    import sys
    build(*(Path(p) for p in sys.argv[1:4]))
