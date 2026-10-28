"""Portable GPU runtime preparation. No torch/CUDA import in the coordinator.

Usage after activating the assigned environment (on EACH node):
  python zevo_gpu_runtime.py --profile train --model CHECKPOINT -- python train.py
  python zevo_gpu_runtime.py --profile train --model CHECKPOINT -- torchrun ...

Compiled artifacts are an optimization, never checkpoints or training state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

SCHEMA = 1
# Deliberately exclude tmpfs: host RAM is needed by model/checkpoint loading.
LOCAL_FS = {"ext4", "xfs", "btrfs", "zfs", "ext3"}
PACKAGES = ("torch", "vllm", "triton", "flashinfer-python", "transformers", "tokenizers", "apache-tvm-ffi")
MODULES = ("torch", "vllm", "triton", "flashinfer", "transformers", "tokenizers", "tvm_ffi", "huggingface_hub")


def report(stage, **fields):
    print('[runtime] ' + json.dumps({"stage": stage, "t": time.time(), **fields}, sort_keys=True), flush=True)


@contextmanager
def timed(stage):
    start = time.monotonic()
    stop = threading.Event()
    def waiting():
        while not stop.wait(60):
            report(stage, state="waiting", elapsed_seconds=round(time.monotonic() - start, 3))
    thread = threading.Thread(target=waiting, daemon=True)
    report(stage, state="start")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        report(stage, state="finished", elapsed_seconds=round(time.monotonic() - start, 3))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:32]


def mount_type(path, mountinfo=None):
    """Longest matching mount, including escaped spaces and bind mounts."""
    path = Path(path).resolve()
    text = Path('/proc/self/mountinfo').read_text() if mountinfo is None else mountinfo
    matches = []
    for line in text.splitlines():
        left, right = line.split(' - ', 1)
        raw = left.split()[4]
        for escaped, value in ((r'\040', ' '), (r'\011', '\t'), (r'\134', '\\')):
            raw = raw.replace(escaped, value)
        root = Path(raw)
        if path == root or root in path.parents:
            matches.append((len(root.parts), right.split()[0]))
    return max(matches, default=(0, 'unknown'))[1]


def local_root(env, required_bytes=0):
    minimum = max(required_bytes, int(env.get('ZEVO_RUNTIME_MIN_FREE_GIB', '8')) * 2**30)
    for candidate in dict.fromkeys(filter(None, (env.get('ZEVO_LOCAL_SCRATCH'),
                                                env.get('SLURM_TMPDIR'), env.get('TMPDIR'), '/tmp'))):
        try:
            base = Path(candidate).resolve()
            kind = mount_type(base)
            if kind not in LOCAL_FS or not base.is_dir() or not os.access(base, os.W_OK):
                report('scratch_candidate', path=str(base), filesystem=kind, accepted=False)
                continue
            if shutil.disk_usage(base).free < minimum:
                report('scratch_candidate', path=str(base), accepted=False, reason='insufficient_space')
                continue
            root = base / f'zevo-runtime-{os.getuid()}'
            root.mkdir(mode=0o700, exist_ok=True)
            if root.is_symlink() or root.stat().st_uid != os.getuid():
                raise ValueError('runtime directory is not owned by this user')
            root.chmod(0o700)
            report('scratch_selected', path=str(root), filesystem=kind)
            return root
        except OSError as exc:
            report('scratch_candidate', path=candidate, accepted=False, reason=type(exc).__name__)
    return None


def compatibility():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Record installed package sources as well as versions: locally patched
    # installations must not borrow another installation's compiled artifacts.
    sources = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec and spec.origin and Path(spec.origin).is_file():
            file = Path(spec.origin)
            sources[name] = [str(file.resolve()), file.stat().st_mtime_ns, file.stat().st_size]
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        if gpu.returncode or not gpu.stdout.strip():
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {'schema': SCHEMA, 'uid': os.getuid(), 'python': sys.version,
            'environment': sys.prefix, 'architecture': platform.machine(), 'packages': versions,
            'sources': sources, 'gpu': sorted(set(gpu.stdout.strip().splitlines())),
            'build_env': {k: os.environ.get(k, '') for k in
                          ('CC', 'CXX', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'CFLAGS', 'CXXFLAGS', 'NVCC_FLAGS')}}


def graph_identity(model, config):
    """Conservative: never reuse weight-dependent graphs across checkpoints."""
    root = Path(model)
    files = []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin', '.pt'}:
                st = p.stat()
                files.append((p.name, st.st_size, st.st_mtime_ns))
    return {'model': str(root.resolve()) if root.exists() else model, 'files': files, 'config': config}


def tree_size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def copy_tree(source, target):
    # Cached builds can contain symlinks to job-private paths. Do not publish
    # these, nor follow them back into NFS while restoring a local snapshot.
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=lambda directory, names: [n for n in names if
                        Path(directory, n).is_symlink() or n in {'.lease', '__pycache__'}])


@contextmanager
def lease(directory, *, shared=False):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / '.lease').open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def restore(store, destination):
    with lease(store, shared=True) as acquired:
        if not acquired:
            report('cache_restore', path=str(store), hit=False, reason='busy')
            return
        current = store / 'current'
        if not current.exists() and (store / 'previous').is_dir():
            current = store / 'previous'
        if current.is_dir():
            if tree_size(current) > shutil.disk_usage(destination).free - 2**30:
                report('cache_restore', hit=False, reason='insufficient_space')
                return
            copy_tree(current, destination)
            report('cache_restore', path=str(store), hit=True)
        else:
            report('cache_restore', path=str(store), hit=False)


def publish(source, store):
    # Only called after successful workload validation/exit. Concurrent runs
    # use private writable copies. Publishing merges under one exclusive lock.
    with lease(store) as acquired:
        if not acquired:
            report('cache_publish', state='skipped', reason='busy')
            return
        scratch = store / ('.publish-' + uuid.uuid4().hex)
        previous = store / 'previous'
        current = store / 'current'
        try:
            scratch.mkdir()
            prior = current if current.exists() else previous
            if prior.exists():
                copy_tree(prior, scratch)
            copy_tree(source, scratch)
            # A crash during publication leaves previous available for recovery.
            if previous.exists():
                shutil.rmtree(previous)
            if current.exists():
                current.rename(previous)
            scratch.rename(current)
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)


def stage_dependencies(root):
    """Copy importable package trees, not a relocatable virtualenv.

    Original interpreter/stdlib stay unchanged. PYTHONPATH selects these exact
    package copies in fresh children, so __file__-relative headers become local.
    """
    sources = []
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec and spec.submodule_search_locations:
            for source in spec.submodule_search_locations:
                sources.append((name, Path(source).resolve()))
    identity = []
    total = 0
    for name, source in sources:
        # Full metadata manifest notices modifications inside a package even
        # when its version and __init__.py have not changed.
        entries = [(str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
        total += sum(e[1] for e in entries)
        identity.append((name, str(source), entries))
    key = digest(identity)
    destination = root / 'dependencies' / key
    if (destination / '.ready').is_file():
        report('dependencies', state='hit', key=key)
        return destination, key
    if shutil.disk_usage(root).free < total + 8 * 2**30:
        raise OSError('insufficient scratch space for dependency copy')
    with lease(destination.parent / (key + '.lock')) as acquired:
        if not acquired:
            # Another process is staging identical immutable packages. Bounded
            # wait, never consume a partially copied environment.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if (destination / '.ready').is_file():
                    return destination, key
                time.sleep(1)
            raise OSError('dependency staging still in progress')
        if (destination / '.ready').is_file():
            return destination, key
        scratch = Path(tempfile.mkdtemp(prefix='dependencies-', dir=root))
        try:
            for name, source in sources:
                # Dereference installed package links: a local-looking tree
                # must not silently read its headers from a shared filesystem.
                shutil.copytree(source, scratch / name, symlinks=False,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            # Verify local copies are importable using the original interpreter;
            # reject an overlay that falls back to the original shared packages.
            probe = ("import importlib,pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); "
                     "sys.path.insert(0,str(root)); "
                     "mods=[importlib.import_module(n) for n in sys.argv[2:]]; "
                     "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in mods)")
            result = subprocess.run([sys.executable, '-c', probe, str(scratch),
                                     *[name for name, _ in sources]],
                                    capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise OSError('local dependency import validation failed: ' + result.stderr[-500:])
            (scratch / '.ready').write_text(key)
            destination.parent.mkdir(exist_ok=True)
            scratch.rename(destination)
        finally:
            if scratch.exists():
                shutil.rmtree(scratch)
    report('dependencies', state='copied', key=key, bytes=total)
    return destination, key


def local_tokenizer_snapshot(root, source, revision, snapshot):
    """Materialize small tokenizer assets locally, preserving HF cache lookup.

    Never copy weight shards here. Frozen adapter configs still name the same
    repository/revision; offline HF lookup resolves that identity locally.
    """
    snapshot = Path(snapshot).resolve()
    cache = root / 'tokenizers' / digest([source, revision, str(snapshot)]) / 'hub'
    repository = cache / ('models--' + source.replace('/', '--'))
    destination = repository / 'snapshots' / snapshot.name
    if not (destination / '.zevo-ready').exists():
        with lease(repository / '.prepare') as acquired:
            if not acquired:
                raise OSError('tokenizer snapshot preparation is busy')
            destination.parent.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix='.tokenizer-', dir=destination.parent))
            try:
                for file in snapshot.rglob('*'):
                    if file.is_file() and file.suffix in {'.json', '.txt', '.model', '.jinja', '.tiktoken', '.py'}:
                        target = scratch / file.relative_to(snapshot)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target)
                (scratch / '.zevo-ready').touch()
                scratch.rename(destination)
            finally:
                if scratch.exists():
                    shutil.rmtree(scratch)
    reference = Path(str(revision or 'main'))
    if reference.is_absolute() or '..' in reference.parts:
        raise ValueError('invalid tokenizer revision')
    ref = repository / 'refs' / reference
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(snapshot.name)
    return cache


def validate_tokenizer(path, dependencies, config):
    probe = ("import sys; sys.path.insert(0,sys.argv[1]); "
             "from transformers import AutoTokenizer; "
             "AutoTokenizer.from_pretrained(sys.argv[2],local_files_only=True,trust_remote_code=sys.argv[3]=='1')")
    trusted = bool((config or {}).get('implementation_config', {}).get('trust_remote_code', False))
    result = subprocess.run([sys.executable, '-c', probe, str(dependencies), str(path), '1' if trusted else '0'],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise OSError('local tokenizer validation failed: ' + result.stderr[-500:])


def prepare_shared(base, config=None, model=""):
    """Once per node/launcher, before concurrent replicas start."""
    env = dict(base)
    with timed('prepare_dependencies'):
        root = local_root(env)
        if root is None or compatibility() is None:
            return {}
        dependencies, key = stage_dependencies(root)
    updates = {'ZEVO_LOCAL_SCRATCH': str(root.parent),
               'ZEVO_RUNTIME_DEPENDENCIES': str(dependencies), 'ZEVO_RUNTIME_DEPENDENCY_KEY': key}
    source = str((config or {}).get('tokenizer_source') or '')
    if source and not Path(source).is_dir():
        with timed('prepare_tokenizer'):
            sys.path.insert(0, str(dependencies))
            try:
                from huggingface_hub import snapshot_download
                # Prepare once before replicas; preserve frozen YAML identity.
                kwargs = dict(revision=(config or {}).get('tokenizer_revision'),
                    cache_dir=env.get('HF_HUB_CACHE') or env.get('HUGGINGFACE_HUB_CACHE') or
                        str(Path(env.get('HF_HOME') or Path.home() / '.cache/huggingface') / 'hub'),
                    allow_patterns=['*.json', '*.txt', '*.model', '*.jinja', '*.tiktoken', '*.py'])
                try:
                    snapshot = snapshot_download(source, local_files_only=True, **kwargs)
                    assets = Path(snapshot)
                    if not (assets / 'tokenizer_config.json').is_file() or not any(
                            (assets / name).is_file() for name in
                            ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'vocab.json', 'vocab.txt')):
                        raise FileNotFoundError('tokenizer snapshot incomplete')
                    report('tokenizer', state='local_snapshot_hit')
                except (OSError, ValueError):
                    snapshot = snapshot_download(source, **kwargs)
                updates['ZEVO_LOCAL_TOKENIZERS'] = json.dumps({source: str(Path(snapshot).resolve())})
                # Preserve the adapter's frozen config and identity. A local
                # checkpoint plus a staged tokenizer needs no Hub resolution
                # in workers. Remote model weights may still need downloading.
                if model and Path(model).is_dir():
                    cache = local_tokenizer_snapshot(root, source, (config or {}).get('tokenizer_revision'), snapshot)
                    local_snapshot = cache / ('models--' + source.replace('/', '--')) / 'snapshots' / Path(snapshot).name
                    validate_tokenizer(local_snapshot, dependencies, config)
                    updates['HF_HUB_CACHE'] = str(cache)
                    updates['HF_HUB_OFFLINE'] = '1'
                else:
                    report('tokenizer', state='staged', offline=False,
                           reason='model_is_not_a_local_checkpoint')
            finally:
                sys.path.pop(0)
    return updates


def dependency_overlay(base, root):
    path = Path(base.get('ZEVO_RUNTIME_DEPENDENCIES') or root / 'absent')
    key = base.get('ZEVO_RUNTIME_DEPENDENCY_KEY', '')
    if key and path.is_dir() and path.is_relative_to(root / 'dependencies'):
        if path.stat().st_uid == os.getuid() and (path / '.ready').read_text() == key:
            report('dependencies', state='prepared_by_launcher', key=key)
            return path, key
    return stage_dependencies(root)


class Runtime:
    def __init__(self, base, model='', config=None):
        self.env = dict(base)
        self.original_home = base.get("HOME", str(Path.home()))
        self.model, self.config = model, config or {}
        self.work = None
        self.stores = []

    def prepare(self):
        with timed('prepare_runtime'):
            root = local_root(self.env)
            if root is None:
                report('runtime_fallback', reason='no_verified_local_scratch')
                return self.env
            identity = compatibility()
            if identity is None:
                report('runtime_fallback', reason='unknown_gpu_compatibility')
                return self.env
            dependencies, source_key = dependency_overlay(self.env, root)
            identity['dependency_manifest'] = source_key
            key = digest(identity)
            self.work = Path(tempfile.mkdtemp(prefix='job-', dir=root))
            if len(str(self.work / 'tmp')) > 80:
                raise ValueError('local scratch path too long for inference IPC; choose a shorter ZEVO_LOCAL_SCRATCH')
            kernel = self.work / 'kernels'
            graph = self.work / 'graphs'
            kernel.mkdir()
            graph.mkdir()
            persistent = self.env.get('ZEVO_RUNTIME_CACHE_ROOT') or self.env.get('ZEVO_INFERENCE_CACHE_ROOT')
            if not persistent and self.env.get('HF_HOME'):
                persistent = str(Path(self.env['HF_HOME']).parent / 'gpu-runtime-cache')
            if persistent:
                base = Path(persistent) / 'runtime-v1' / f'uid-{os.getuid()}' / key
                self.stores = [(kernel, base / 'kernels')]
                if self.model and Path(self.model).is_dir():
                    self.stores.append((graph, base / 'graphs' / digest(graph_identity(self.model, self.config))))
                else:
                    report('graph_cache', state='private', reason='no_local_checkpoint_identity')
                with timed('restore_cache'):
                    for target, store in self.stores:
                        restore(store, target)
            for variable, path in {
                'TRITON_CACHE_DIR': kernel / 'triton', 'FLASHINFER_CACHE_DIR': kernel / 'flashinfer',
                'CUDA_CACHE_PATH': kernel / 'cuda', 'TORCH_EXTENSIONS_DIR': kernel / 'torch_extensions',
                'TORCHINDUCTOR_CACHE_DIR': graph / 'inductor', 'VLLM_CACHE_ROOT': graph / 'vllm',
                'XDG_CACHE_HOME': self.work / 'home/.cache', 'HOME': self.work / 'home',
                'TMPDIR': self.work / 'tmp', 'TMP': self.work / 'tmp', 'TEMP': self.work / 'tmp',
            }.items():
                path.mkdir(parents=True, exist_ok=True)
                self.env[variable] = str(path)
            # vLLM stores architecture inspection and autotune results beneath
            # its graph root. Separate only these library-keyed subcaches.
            for name in ('modelinfos', 'flashinfer_autotune_cache'):
                target = kernel / name
                target.mkdir(exist_ok=True)
                link = graph / 'vllm' / name
                if link.exists():
                    copy_tree(link, target)
                    shutil.rmtree(link)
                link.symlink_to(target, target_is_directory=True)
            (self.work / 'home/.cache/flashinfer').symlink_to(kernel / 'flashinfer', target_is_directory=True)
            self.env.setdefault('HF_HOME', str(Path(self.original_home) / '.cache/huggingface'))
            self.env['PYTHONPATH'] = str(dependencies) + os.pathsep + self.env.get('PYTHONPATH', '')
            self.env['ZEVO_RUNTIME_PREPARED'] = '1'
            report('runtime_ready', compatibility=key, path=str(self.work),
                   interpreter=sys.executable, dependency_path=str(dependencies))
            return self.env

    def finish(self, success):
        try:
            if success:
                with timed('publish_cache'):
                    for source, store in self.stores:
                        try:
                            publish(source, store)
                        except OSError as exc:
                            report('cache_publish', state='failed', reason=str(exc))
        finally:
            if self.work:
                shutil.rmtree(self.work, ignore_errors=True)


def process_waits(pid):
    """Read-only diagnostics: no process attachment, signals or command/env dump."""
    rows, pending = [], [pid]
    while pending and len(rows) < 32:
        current = pending.pop(0)
        root = Path('/proc') / str(current)
        try:
            stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
            rows.append({'pid': current, 'name': (root / 'comm').read_text().strip(),
                         'state': stat[0], 'wait': (root / 'wchan').read_text().strip()})
            pending.extend(int(v) for v in (root / f'task/{current}/children').read_text().split())
        except (OSError, ValueError, IndexError):
            continue
    return rows


def _launch_argv(command, *, profile='gpu', model='', config_path=''):
    """One launch specification for synchronous, parallel and CLI callers."""
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError('GPU launch requires a non-empty argv list, not a shell string')
    argv = [sys.executable, str(Path(__file__).resolve()), '--profile', profile,
            '--model', str(model)]
    if config_path:
        argv.extend(['--config', str(config_path)])
    return [*argv, '--', *command]


def launch(command, *, profile='gpu', model='', config_path='', env=None, **process_options):
    """Start a GPU workload; the common supervisor owns cache lifecycle/signals.

    Coordinators supply only the original argv, model/config identity and worker
    environment. Returned Popen supervises the workload until cache publication
    and cleanup have finished. No GPU stack is imported in the coordinator.
    """
    if process_options.get('shell'):
        raise ValueError('GPU launch requires argv execution, not shell=True')
    return subprocess.Popen(_launch_argv(command, profile=profile, model=model,
                                        config_path=config_path), env=env, **process_options)


def run(command, *, profile='gpu', model='', config_path='', env=None, check=True):
    """Blocking form of launch, with the same process and cache lifecycle."""
    proc = launch(command, profile=profile, model=model, config_path=config_path, env=env)
    try:
        code = proc.wait()
    except BaseException:
        proc.terminate()
        proc.wait()
        raise
    result = subprocess.CompletedProcess(command, code)
    if check:
        result.check_returncode()
    return result


def prepare_environment(*, model='', config_path='', env=None):
    """Optional once-per-node preparation for any multi-worker coordinator."""
    base = dict(os.environ if env is None else env)
    with tempfile.TemporaryDirectory(prefix='zevo-gpu-prepare-') as directory:
        output = Path(directory) / 'environment.json'
        argv = [sys.executable, str(Path(__file__).resolve()), '--model', str(model),
                '--prepare-only', str(output)]
        if config_path:
            argv.extend(['--config', str(config_path)])
        subprocess.run(argv, env=base, check=True)
        return {**base, **json.loads(output.read_text())}


def run_command(command, *, profile, model='', config=None, base=None):
    runtime = Runtime(base or os.environ, model, config)
    success = False
    try:
        try:
            env = runtime.prepare()
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report('runtime_fallback', reason=str(exc))
            runtime.stores = []
            env = dict(base or os.environ)
        report('launch', profile=profile)
        proc = subprocess.Popen(command, env=env, start_new_session=True)
        old_handlers, stopping = {}, []
        def forward(received, frame):
            if not stopping:
                stopping.append(time.monotonic())
            try:
                os.killpg(proc.pid, received)
            except ProcessLookupError:
                pass
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, forward)
        last_report = time.monotonic()
        try:
            while proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if stopping and time.monotonic() - stopping[0] > 5:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if time.monotonic() - last_report >= 60:
                        report('workload_alive', pid=proc.pid, profile=profile,
                               processes=process_waits(proc.pid))
                        last_report = time.monotonic()
            success = proc.returncode == 0 and not stopping
            return proc.returncode if not stopping else 128 + signal.SIGTERM
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    finally:
        runtime.finish(success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='gpu', help='Diagnostic workload label; does not select cache compatibility or execution policy')
    parser.add_argument('--prepare-only', default='', metavar='ENV_JSON')
    parser.add_argument('--model', default='')
    parser.add_argument('--config', default='')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command and not args.prepare_only:
        parser.error('missing workload command after --')
    config = {}
    if args.config:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text())
    if args.prepare_only:
        try:
            updates = prepare_shared(os.environ, config, args.model)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report('runtime_fallback', reason=str(exc))
            updates = {}
        Path(args.prepare_only).write_text(json.dumps(updates))
        return 0
    return run_command(command, profile=args.profile, model=args.model, config=config)


if __name__ == '__main__':
    raise SystemExit(main())
