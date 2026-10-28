"""Persistent caches never share writable worker slots concurrently."""
import importlib.util
import sys
from types import SimpleNamespace

from tests.test_parallel_inference import HELPER


def helper():
    spec = importlib.util.spec_from_file_location('cache_test_runner', HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cache_reuse_and_concurrent_isolation(tmp_path, monkeypatch):
    runner = helper()
    monkeypatch.setattr(runner, '_cache_signature', lambda *_: 'compatible')
    base = {'ZEVO_INFERENCE_CACHE_ROOT': str(tmp_path)}
    with runner._cache_slots(base, 'model', {}, 2) as first:
        assert first[0] != first[1]
        artifact = first[0] / 'compiled-kernel'
        artifact.write_text('cached')
        with runner._cache_slots(base, 'model', {}, 1) as concurrent:
            assert concurrent[0] not in first.values()
    with runner._cache_slots(base, 'model', {}, 1) as later:
        assert later[0] == first[0]
        assert (later[0] / 'compiled-kernel').read_text() == 'cached'
    monkeypatch.setattr(runner, '_cache_signature', lambda *_: None)
    with runner._cache_slots(base, 'model', {}, 2) as unknown:
        assert unknown == {}


def test_cache_signature_separates_runtime_and_gpu(monkeypatch):
    import importlib.metadata
    runner = helper()
    gpu = SimpleNamespace(returncode=0, stdout='GB200, 10.0, 580\n')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: gpu)
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '1')
    first = runner._cache_signature('model', {})
    assert first == runner._cache_signature('model', {})
    assert first != runner._cache_signature('other-model', {})
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: '2')
    second = runner._cache_signature('model', {})
    assert first != second
    gpu.stdout = 'H100, 9.0, 580\n'
    assert second != runner._cache_signature('model', {})


def test_ipc_stays_private_while_compiled_cache_survives(tmp_path):
    import tempfile
    from pathlib import Path
    runner = helper()
    cache = tmp_path / 'persistent'
    with tempfile.TemporaryDirectory(prefix='zc-', dir='/tmp') as scratch:
        root = Path(scratch)
        first = runner._worker_environment({'HF_HOME': '/models'}, root / 'a', ['0'], cache)
        second = runner._worker_environment({'HF_HOME': '/models'}, root / 'b', ['0'], cache)
        assert first['HOME'] != second['HOME']
        assert first['TMPDIR'] != second['TMPDIR']
        assert first['TRITON_CACHE_DIR'] == second['TRITON_CACHE_DIR']
        assert first['HF_HOME'] == '/models'
        assert (Path(first['HOME']) / '.cache').resolve() == cache / 'home/.cache'


def test_worker_keeps_cache_lease_after_parent_releases_it(tmp_path, monkeypatch):
    import subprocess
    runner = helper()
    monkeypatch.setattr(runner, '_cache_signature', lambda *_: 'compatible')
    base = {'ZEVO_INFERENCE_CACHE_ROOT': str(tmp_path)}
    with runner._cache_slots(base, 'model', {}, 1) as slots:
        path = slots[0]
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, pass_fds=(slots.fds[0],))
    try:
        with runner._cache_slots(base, 'model', {}, 1) as concurrent:
            assert concurrent[0] != path
    finally:
        child.communicate(timeout=5)
    with runner._cache_slots(base, 'model', {}, 1) as released:
        assert released[0] == path
