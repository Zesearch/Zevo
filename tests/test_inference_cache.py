"""Inference uses the common runtime; only compatible kernels cross checkpoints."""
import importlib.util
import sys
from types import SimpleNamespace

from tests.test_parallel_inference import HELPER


def helper(name='parallel_inference'):
    path = HELPER.with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('cache_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compatibility_separates_runtime_gpu_and_build_flags(monkeypatch):
    runner = helper('gpu_runtime')
    gpu = SimpleNamespace(returncode=0, stdout='GB200, 10.0, 580\n')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: gpu)
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda _: '1')
    monkeypatch.setattr(runner, 'MODULES', ())
    first = runner.digest(runner.compatibility())
    assert first == runner.digest(runner.compatibility())
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda _: '2')
    second = runner.digest(runner.compatibility())
    assert first != second
    gpu.stdout = 'H100, 9.0, 580\n'
    third = runner.digest(runner.compatibility())
    assert second != third
    monkeypatch.setenv('NVCC_FLAGS', '--use_fast_math')
    assert third != runner.digest(runner.compatibility())
    gpu.returncode = 1
    assert runner.compatibility() is None


def test_graph_identity_not_shared_across_checkpoint_or_engine_changes(tmp_path):
    runner = helper('gpu_runtime')
    model = tmp_path / 'model'
    model.mkdir()
    weights = model / 'model.safetensors'
    weights.write_text('first weights')
    before = runner.graph_identity(str(model), {'dtype': 'bf16'})
    assert before != runner.graph_identity(str(model), {'dtype': 'fp16'})
    weights.write_text('new weights after resume')
    assert before != runner.graph_identity(str(model), {'dtype': 'bf16'})
    assert before != runner.graph_identity(str(tmp_path / 'other-model'), {'dtype': 'bf16'})


def test_worker_fallback_ipc_and_caches_are_private():
    import tempfile
    from pathlib import Path
    runner = helper()
    with tempfile.TemporaryDirectory(prefix='zc-', dir='/tmp') as scratch:
        root = Path(scratch)
        first = runner._worker_environment({'HF_HOME': '/models'}, root / 'a', ['0'])
        second = runner._worker_environment({'HF_HOME': '/models'}, root / 'b', ['0'])
        assert first['HOME'] != second['HOME']
        assert first['TMPDIR'] != second['TMPDIR']
        assert first['TRITON_CACHE_DIR'] != second['TRITON_CACHE_DIR']
        assert first['HF_HOME'] == '/models'
