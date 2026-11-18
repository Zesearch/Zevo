"""Fail-closed semantic comparison for execution-only Train repairs.

The registered bytes remain the audit identity. Only explicitly named diagnostic
and bookkeeping fields may differ without a training reconfiguration approval.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shlex

import yaml


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("Training YAML requires unique string mapping keys")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
_IMPL = ('training', 'implementation_config')
_DIAGNOSTICS = {
    'startup_max_first_batch_nll', 'startup_max_first_logged_loss',
    'startup_max_reference_loss_mismatch',
}
_MISSING = object()


def _attempt_path(before, after):
    if not isinstance(before, str) or not isinstance(after, str):
        return False
    if not before.startswith('/') or not after.startswith('/'):
        return False
    if any(part in {'.', '..'} for value in (before, after) for part in value.split('/')):
        return False
    normalize = lambda value: re.sub(r'/attempt-[1-9][0-9]*/', '/attempt-*/', value)
    return normalize(before) == normalize(after)


def _launch(before, after):
    if not isinstance(before, str) or not isinstance(after, str):
        return False
    try:
        a, b = shlex.split(before), shlex.split(after)
    except ValueError:
        return False
    if len(a) != len(b):
        return False
    return all(x == y or (i > 0 and a[i-1] == '--inference-config'
        and x.endswith('/baseline_inference_config.yaml') and _attempt_path(x, y))
        for i, (x, y) in enumerate(zip(a, b)))


def _allowed(path, before, after):
    if path == _IMPL + ('output_dir',):
        return _attempt_path(before, after)
    if path == _IMPL + ('launch_command',):
        return _launch(before, after)
    if path[:-1] == _IMPL and path[-1] in _DIAGNOSTICS:
        # Adding or tightening a positive stop threshold is diagnostic. Removing
        # or loosening a guard changes execution safety and needs explicit review.
        valid = lambda v: type(v) in {int, float} and math.isfinite(v) and v > 0
        return valid(after) and (before is _MISSING or (valid(before) and after <= before))
    if path[:3] == _IMPL + ('startup_validation',) and len(path) in {4, 5}:
        field = path[3]
        if field in {'rationale', 'reusable_prior_evidence'} and len(path) == 4:
            return isinstance(before, str) and isinstance(after, str)
        if field in {'checks', 'changed_execution_paths'}:
            if len(path) == 5:
                return isinstance(before, str) and isinstance(after, str)
            return (isinstance(before, list) and isinstance(after, list)
                    and all(isinstance(x, str) for x in before + after))
    if path == _IMPL + ('gpu_memory_preflight', 'note'):
        return isinstance(before, str) and isinstance(after, str)
    if path in {
        _IMPL + ('gpu_memory_preflight', 'estimated_peak_per_rank_gib'),
        _IMPL + ('gpu_memory_preflight', 'components_gib', 'startup_gate_dense_reference_logits_no_grad'),
    }:
        return (type(after) in {int, float} and math.isfinite(after) and after > 0
                and (before is _MISSING or (type(before) in {int, float} and after >= before)))
    if path == _IMPL + ('resolved_defaults_notes',):
        return isinstance(after, str) and (before is _MISSING or isinstance(before, str))
    if len(path) == 3 and path[0] == 'suggestion_decisions' and path[2] == 'rationale':
        return isinstance(before, str) and isinstance(after, str)
    return False


def _diff(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            yield from _diff(before.get(key, _MISSING), after.get(key, _MISSING), (*path, key))
    elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for i, (a, b) in enumerate(zip(before, after)):
            yield from _diff(a, b, (*path, str(i)))
    elif type(before) is not type(after) or before != after:
        yield {"path": '.'.join(path), "before": None if before is _MISSING else before,
               "after": None if after is _MISSING else after,
               "added": before is _MISSING, "removed": after is _MISSING,
               "execution_only": _allowed(path, before, after)}


def compare_registered(meta, path):
    path = Path(path)
    candidate = path.read_bytes()
    expected = meta.get('training_config_sha256')
    actual = hashlib.sha256(candidate).hexdigest()
    audit = {'original_sha256': expected, 'proposed_sha256': actual, 'changes': []}
    if expected and actual == expected:
        return audit
    baseline = meta.get('training_config_snapshot', '').encode('utf-8')
    if not expected or hashlib.sha256(baseline).hexdigest() != expected:
        # Legacy rows may have a preserved original alongside the mutable config.
        # Its name is never evidence: require the registered byte hash.
        attempt = int(meta.get('execution_attempt') or 1)
        baseline = b''
        for name in (f'train_config.attempt{attempt}.yaml', f'train_config.attempt-{attempt}.yaml'):
            original = path.with_name(name)
            if original.is_file():
                contents = original.read_bytes()
                if hashlib.sha256(contents).hexdigest() == expected:
                    baseline = contents
                    break
        if not baseline:
            raise ValueError('Registered training configuration snapshot unavailable; explicit reconfigure review required')
    try:
        before = yaml.load(baseline, Loader=_UniqueLoader)
        after = yaml.load(candidate, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError('Invalid training YAML') from exc
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError('Training configuration must be a mapping')
    try:
        # Reject recursive aliases, non-JSON YAML values and non-finite numbers
        # before diffing or persisting the audit in a JSON database column.
        json.dumps([before, after], allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Training YAML must contain finite JSON-compatible values') from exc
    audit['changes'] = list(_diff(before, after))
    training = after.get('training')
    implementation = training.get('implementation_config') if isinstance(training, dict) else None
    memory = implementation.get('gpu_memory_preflight') if isinstance(implementation, dict) else None
    memory = memory if isinstance(memory, dict) else {}
    for change in audit['changes']:
        if 'gpu_memory_preflight' in change['path'] and change['execution_only']:
            estimate, capacity = memory.get('estimated_peak_per_rank_gib'), memory.get('per_gpu_vram_gib')
            if not (type(estimate) in {int, float} and type(capacity) in {int, float}
                    and 0 < estimate < capacity):
                change['execution_only'] = False
    return audit
