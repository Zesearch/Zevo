"""Node-local runtime cache correctness without a GPU or network."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

HELPER = Path(__file__).resolve().parents[1] / 'playbook/runners/gpu_runtime.py'

@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location('runtime_test', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if hasattr(module, "_test_root"):
        module.shutil.rmtree(module._test_root, ignore_errors=True)


def test_mount_selection_uses_longest_mount_and_rejects_network_tmpfs(runtime, tmp_path, monkeypatch):
    mounts = '1 0 0:1 / / rw - ext4 /dev/x rw\n2 1 0:2 / /zevo-scratch rw - nfs4 server:/tmp rw\n3 1 0:3 / /zevo-scratch/local rw - xfs /dev/y rw'
    assert runtime.mount_type('/zevo-scratch/remote', mounts) == 'nfs4'
    assert runtime.mount_type('/zevo-scratch/local/job', mounts) == 'xfs'
    monkeypatch.setattr(runtime, 'mount_type', lambda p: 'nfs4')
    assert runtime.local_root({'ZEVO_LOCAL_SCRATCH': str(tmp_path)}) is None
    monkeypatch.setattr(runtime, 'mount_type', lambda p: 'tmpfs')
    assert runtime.local_root({'ZEVO_LOCAL_SCRATCH': str(tmp_path)}) is None
    monkeypatch.setattr(runtime, 'mount_type', lambda p: 'xfs')
    assert runtime.local_root({'ZEVO_LOCAL_SCRATCH': str(tmp_path), 'ZEVO_RUNTIME_MIN_FREE_GIB': '0'}).parent == tmp_path


def test_publish_merge_restore_and_interrupted_commit(runtime, tmp_path):
    store = tmp_path / 'store'
    one, two, dest = [tmp_path / name for name in ('one', 'two', 'dest')]
    for p in (one, two, dest): p.mkdir()
    (one / 'kernel-a').write_text('A')
    (two / 'kernel-b').write_text('B')
    (one / 'external').symlink_to('/etc')
    runtime.publish(one, store)
    runtime.publish(two, store)
    assert not (store / 'current/external').exists()
    (store / 'current').rename(store / 'previous') # crash after old current renamed
    runtime.restore(store, dest)
    assert (dest / 'kernel-a').read_text() == 'A'
    assert (dest / 'kernel-b').read_text() == 'B'
    runtime.publish(two, store)
    with runtime.lease(store) as held:
        assert held
        (two / 'kernel-c').write_text('C')
        runtime.publish(two, store)
        assert not (store / 'current/kernel-c').exists()


def configure(runtime, monkeypatch, tmp_path):
    root = Path(tempfile.mkdtemp(prefix='zr-', dir='/tmp')).resolve()
    monkeypatch.setattr(runtime, '_test_root', root, raising=False)
    deps = root / 'dependencies/test'
    deps.mkdir(parents=True)
    (deps / '.ready').write_text('test')
    monkeypatch.setattr(runtime, 'local_root', lambda *a, **k: root)
    monkeypatch.setattr(runtime, 'compatibility', lambda: {'gpu': 'test', 'versions': '1'})
    monkeypatch.setattr(runtime, 'stage_dependencies', lambda root: (deps, 'test'))
    model1, model2 = tmp_path / 'checkpoint1', tmp_path / 'checkpoint2'
    for model in (model1, model2):
        model.mkdir()
        (model / 'model.safetensors').write_text(model.name)
    base = {'ZEVO_RUNTIME_CACHE_ROOT': str(tmp_path / 'persistent'), 'HOME': str(tmp_path),
            'ZEVO_RUNTIME_DEPENDENCIES': str(deps), 'ZEVO_RUNTIME_DEPENDENCY_KEY': 'test'}
    return base, model1, model2


def test_checkpoints_share_kernels_not_graphs_and_workers_are_private(runtime, monkeypatch, tmp_path):
    base, model1, model2 = configure(runtime, monkeypatch, tmp_path)
    first = runtime.Runtime(base, str(model1), {'dtype': 'bf16'})
    env = first.prepare()
    Path(env['TRITON_CACHE_DIR'], 'compiled').write_text('kernel')
    Path(env['TORCHINDUCTOR_CACHE_DIR'], 'graph').write_text('weight-dependent')
    Path(env['VLLM_CACHE_ROOT'], 'modelinfos', 'qwen.json').write_text('architecture')
    concurrent = runtime.Runtime(base, str(model1), {'dtype': 'bf16'})
    other = concurrent.prepare()
    assert other['TRITON_CACHE_DIR'] != env['TRITON_CACHE_DIR']
    assert not Path(other['TRITON_CACHE_DIR'], 'compiled').exists()
    concurrent.finish(False)
    first.finish(True)
    second = runtime.Runtime(base, str(model2), {'dtype': 'bf16'})
    env2 = second.prepare()
    assert Path(env2['TRITON_CACHE_DIR'], 'compiled').read_text() == 'kernel'
    assert Path(env2['VLLM_CACHE_ROOT'], 'modelinfos', 'qwen.json').read_text() == 'architecture'
    assert not Path(env2['TORCHINDUCTOR_CACHE_DIR'], 'graph').exists()
    Path(env2['TRITON_CACHE_DIR'], 'broken').write_text('partial')
    second.finish(False)
    retry = runtime.Runtime(base, str(model1), {'dtype': 'bf16'})
    env3 = retry.prepare()
    assert Path(env3['TORCHINDUCTOR_CACHE_DIR'], 'graph').read_text() == 'weight-dependent'
    assert not Path(env3['TRITON_CACHE_DIR'], 'broken').exists()
    retry.finish(False)
    monkeypatch.setattr(runtime, 'compatibility', lambda: {'gpu': 'different', 'versions': '1'})
    different = runtime.Runtime(base, str(model1))
    assert not Path(different.prepare()['TRITON_CACHE_DIR'], 'compiled').exists()
    different.finish(False)


def test_dependency_copy_invalidates_on_inner_source_change(runtime, tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '__init__.py').write_text('')
    (source / 'header.h').write_text('original')
    from types import SimpleNamespace
    monkeypatch.setattr(runtime, 'MODULES', ('fake',))
    monkeypatch.setattr(runtime.importlib.util, 'find_spec', lambda name: SimpleNamespace(submodule_search_locations=[str(source)]))
    monkeypatch.setattr(runtime.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 2**30))
    root = tmp_path / 'scratch'
    root.mkdir()
    one, key1 = runtime.stage_dependencies(root)
    assert (one / 'fake/header.h').read_text() == 'original'
    assert runtime.stage_dependencies(root) == (one, key1)
    (source / 'header.h').write_text('patched source')
    two, key2 = runtime.stage_dependencies(root)
    assert key1 != key2
    assert (two / 'fake/header.h').read_text() == 'patched source'
    assert (one / 'fake/header.h').read_text() == 'original'


def test_wrapper_preserves_checkpoint_args_and_does_not_publish_failure(runtime, tmp_path, monkeypatch):
    env = dict(os.environ, RESUME_SENTINEL='checkpoint-42')
    monkeypatch.setattr(runtime.Runtime, 'prepare', lambda self: self.env)
    statuses = []
    monkeypatch.setattr(runtime.Runtime, 'finish', lambda self, ok: statuses.append(ok))
    target = tmp_path / 'result.json'
    script = 'import os,sys,json; json.dump([os.environ["RESUME_SENTINEL"],sys.argv[2:]],open(sys.argv[1],"w"))'
    assert runtime.run_command([sys.executable, '-c', script, str(target), '--resume', 'checkpoint-42'], profile='train', base=env) == 0
    assert json.loads(target.read_text()) == ['checkpoint-42', ['--resume', 'checkpoint-42']]
    assert runtime.run_command([sys.executable, '-c', 'raise SystemExit(7)'], profile='train', base=env) == 7
    assert statuses == [True, False]


def test_multiple_readers_can_restore_same_snapshot(runtime, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'kernel').write_text('valid')
    store = tmp_path / 'store'
    runtime.publish(source, store)
    destination = tmp_path / 'dest'
    destination.mkdir()
    with runtime.lease(store, shared=True) as acquired:
        assert acquired
        runtime.restore(store, destination)
    assert (destination / 'kernel').read_text() == 'valid'


def test_shared_preparation_reuses_dependency_overlay_and_local_tokenizer(runtime, tmp_path, monkeypatch):
    base, model, _ = configure(runtime, monkeypatch, tmp_path)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'tokenizer_config.json').write_text('{}')
    (snapshot / 'tokenizer.json').write_text('{}')
    (snapshot / 'model.safetensors').write_text('weights must not be copied')
    import types
    hub = types.ModuleType('huggingface_hub')
    calls = []
    def download(source, **kwargs):
        calls.append((source, kwargs))
        return str(snapshot)
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setattr(runtime, 'validate_tokenizer', lambda path, *args: (path / 'tokenizer.json').read_text())
    updates = runtime.prepare_shared(base, {'tokenizer_source': 'org/model', 'tokenizer_revision': 'commit'}, str(model))
    assert json.loads(updates['ZEVO_LOCAL_TOKENIZERS']) == {'org/model': str(snapshot)}
    assert calls[0][1]['revision'] == 'commit'
    assert updates['HF_HUB_OFFLINE'] == '1'
    local = Path(updates['HF_HUB_CACHE']) / 'models--org--model'
    assert (local / 'refs/commit').read_text() == snapshot.name
    assert (local / 'snapshots' / snapshot.name / 'tokenizer.json').is_file()
    assert not (local / 'snapshots' / snapshot.name / 'model.safetensors').exists()
    monkeypatch.setattr(runtime, 'stage_dependencies', lambda *_: pytest.fail('restaged per worker'))
    root = runtime._test_root
    assert runtime.dependency_overlay({**base, **updates}, root)[1] == 'test'


def test_wrapper_cancellation_reaches_child(runtime, tmp_path):
    import subprocess
    import time
    marker = tmp_path / 'stopped'
    ready = tmp_path / 'ready'
    script = ('import signal,time,pathlib,sys; '
              'signal.signal(signal.SIGTERM,lambda *a:(pathlib.Path(sys.argv[1]).write_text("stopped"),sys.exit(0))); '
              'pathlib.Path(sys.argv[2]).write_text("ready"); time.sleep(60)')
    proc = subprocess.Popen([sys.executable, str(HELPER), '--profile', 'train', '--',
                             sys.executable, '-c', script, str(marker), str(ready)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(.05)
        assert ready.exists()
        proc.terminate()
        assert proc.wait(timeout=10) != 0
        assert marker.read_text() == 'stopped'
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def test_runtime_cli_accepts_new_workload_labels_without_stage_whitelist(runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, 'run_command', lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    for profile in ('data', 'evaluation', 'embedding', 'future_gpu_stage'):
        monkeypatch.setattr(sys, 'argv', ['helper', '--profile', profile, '--', 'python', 'workload.py', '--resume', 'saved'])
        assert runtime.main() == 0
        assert calls[-1][0] == ['python', 'workload.py', '--resume', 'saved']
        assert calls[-1][1]['profile'] == profile
    monkeypatch.setattr(sys, 'argv', ['helper', '--', 'python', 'workload.py'])
    assert runtime.main() == 0
    assert calls[-1][1]['profile'] == 'gpu'


def test_public_launch_and_run_share_spec_and_preserve_original_argv(runtime, monkeypatch):
    calls = []
    class Process:
        def wait(self):
            return 7
    def popen(argv, **options):
        calls.append((argv, options))
        return Process()
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    command = ['python', 'workload.py', '--resume', '/persistent/checkpoint with spaces']
    env = {'CUDA_VISIBLE_DEVICES': '2', 'RESUME': '/persistent/checkpoint'}
    for profile in ('inference', 'train', 'data', 'future'):
        proc = runtime.launch(command, profile=profile, model='/model', config_path='/config.yaml', env=env)
        asynchronous = calls[-1]
        assert proc.wait() == 7
        result = runtime.run(command, profile=profile, model='/model', config_path='/config.yaml', env=env, check=False)
        assert result.returncode == 7
        assert calls[-1] == asynchronous
        argv, options = calls[-1]
        assert argv[argv.index('--') + 1:] == command
        assert options['env'] == env
    with pytest.raises(runtime.subprocess.CalledProcessError) as error:
        runtime.run(command, env=env)
    assert error.value.returncode == 7
    with pytest.raises(ValueError, match='shell'):
        runtime.launch('python workload.py')


def test_public_launcher_executes_original_command_and_captures_output(runtime, tmp_path):
    # Exercise the actual supervisor/child boundary. This test runs without a
    # GPU, using the same explicit fallback as CPU-only development hosts.
    env = {**os.environ, 'ZEVO_RUNTIME_MIN_FREE_GIB': '999999999', 'RESUME_TEST': 'kept'}
    proc = runtime.launch([sys.executable, '-c',
        "import os,sys; print('child:'+os.environ['RESUME_TEST'],flush=True); sys.exit(7)"],
        env=env, stdout=runtime.subprocess.PIPE, stderr=runtime.subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode == 7, stderr
    assert 'child:kept' in stdout
    assert 'runtime_fallback' in stdout
