"""Transport for the deterministic evaluator; no scoring runs on the SSH host.

Slurm submission stays owned by the engine's allocation controller. Direct
hosts run a ticket-marked process. Both execute the same worker and retain
verified, benchmark-level receipts across attempts.
"""
from __future__ import annotations

import asyncio
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import tarfile
import tempfile
import time
from typing import Any

from zevo.contracts.evaluation import EvaluationResult, EvaluationTaskInput
from zevo.contracts.infrastructure import InfrastructureDeviceInfo
from zevo.engine.agent.drivers.base import DriverRunResult
from zevo.engine.remote_transfer import build_download_command, build_upload_commands, ssh_base_args


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False))
    temp.replace(path)


class EvaluationReceipt:
    """Reuse only scores bound to these exact inputs, scorer and runner."""

    def __init__(self, inp: EvaluationTaskInput, directory: Path):
        self.path = directory / 'evaluation-receipt.json'
        payload = inp.model_dump(mode='json')
        for key in ('activation_contract', 'memory', 'run_context', 'slurm_job'):
            payload.pop(key, None)
        self.identity = {
            'input': payload,
            'files': {field: digest(Path(getattr(inp, field))) for field in
                      ('predictions_path', 'scoring_set', 'sample_submission', 'evaluation_script')
                      if getattr(inp, field)},
            'runner': digest(Path(__file__).with_name('agent') / 'drivers' / 'evaluation_runner.py'),
            'runtime': os.environ.get('ZEVO_EVALUATION_RUNTIME_IDENTITY', ''),
        }

    def load(self) -> EvaluationResult | None:
        try:
            value = json.loads(self.path.read_text())
            if value['identity'] != self.identity:
                return None
            result = EvaluationResult.model_validate(value['result'])
            from zevo.engine.agent.drivers.evaluation_runner import _validated_score
            _validated_score(Path(result.metrics_path))
            if result.status != 'succeeded' or digest(Path(result.metrics_path)) != value['metrics_sha256']:
                return None
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, result: EvaluationResult) -> None:
        atomic_json(self.path, {'identity': self.identity, 'result': result.model_dump(mode='json'),
                               'metrics_sha256': digest(Path(result.metrics_path))})


async def command(argv: list[str], *, stdin: bytes | None = None, timeout: float = 120) -> str:
    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if stdin else None,
                                               stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        # Never include command arguments or injected credentials in errors.
        raise ValueError(f'evaluation transport {argv[0]} exited {proc.returncode}: {err.decode(errors="replace")[-1500:]}')
    return out.decode(errors='replace')


def ssh(info: InfrastructureDeviceInfo, body: str) -> list[str]:
    return [*ssh_base_args(info.ssh), f'{info.ssh.user}@{info.ssh.host}', body]


def _provider_scope():
    # Console supplies account-scoped credentials; standalone Zevo has no
    # account module. Never hide failures inside an installed scope module.
    try:
        return importlib.import_module('zevo.engine.agent.provider_scope')
    except ModuleNotFoundError as exc:
        if exc.name != 'zevo.engine.agent.provider_scope':
            raise
        return None


def judge_environment() -> dict[str, str]:
    scope = _provider_scope()
    env = scope.apply_scope(os.environ.copy()) if scope else os.environ.copy()
    # No compute account credentials, DB credentials or service tokens.
    allowed = {'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY',
               'AWS_BEARER_TOKEN_BEDROCK', 'AWS_REGION', 'HF_TOKEN'}
    return {key: value for key, value in env.items() if key in allowed}


def package_runtime(script: Path, root: Path) -> dict | None:
    """Export a declared frozen Python runtime, never copy its native binaries.

    Legacy task scorers can select a versioned /app/data/evaluation-runtimes
    interpreter. Keep the scorer byte-identical; recreate that interpreter
    prefix inside the compute container using the installed package versions.
    """
    tree = ast.parse(script.read_text())
    prefix = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'RUNTIME' for target in statement.targets
        ) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            prefix = Path(statement.value.value)
    if prefix is None:
        return None
    allowed = Path('/app/data/evaluation-runtimes')
    if not prefix.is_relative_to(allowed) or prefix == allowed:
        raise ValueError('custom evaluator RUNTIME must be a versioned /app/data/evaluation-runtimes directory')
    if not prefix.is_dir():
        raise ValueError(f'frozen evaluation runtime is missing: {prefix}')
    sites = sorted(prefix.glob('lib/python*/site-packages'))
    if len(sites) != 1:
        raise ValueError(f'frozen evaluation runtime has no unambiguous Python environment: {prefix}')
    requirements = []
    for distribution in importlib.metadata.distributions(path=[str(sites[0])]):
        name, version = distribution.metadata['Name'], distribution.version
        if name.lower() not in {'zevo', 'pip', 'setuptools', 'wheel'}:
            requirements.append(f'{name}=={version}')
    if not requirements:
        raise ValueError(f'frozen evaluation runtime has no dependency inventory: {prefix}')
    relative = prefix.relative_to('/app/data')
    destination = root / 'compat' / relative
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'requirements.txt').write_text('\n'.join(sorted(set(requirements))) + '\n')
    # These corpora are immutable evaluator assets, not any Run's labels.
    nltk = Path('/app/data/nltk_data')
    if nltk.is_dir() and not (root / 'compat/nltk_data').exists():
        shutil.copytree(nltk, root / 'compat/nltk_data')
    return {'prefix': str(prefix), 'python': sites[0].parent.name.removeprefix('python')}


def make_bundle(inp: EvaluationTaskInput, local: Path, remote: str) -> Path:
    """Copy only declared scoring inputs and system code, never a whole Run."""
    root = local / 'bundle'
    root.mkdir()
    payload = inp.model_dump(mode='json')
    payload.update(device_info_path='', remote_work_dir='', slurm_job={})
    payload['memory'] = {}
    payload['run_context'] = {}
    entries = [payload, *payload['suite_members']]
    runtimes = {}
    for index, entry in enumerate(entries):
        for field in ('predictions_path', 'scoring_set', 'sample_submission', 'evaluation_script'):
            if not entry.get(field):
                continue
            source = Path(entry[field]).resolve()
            if not source.is_file():
                raise ValueError(f'evaluation input is missing: {source}')
            if field == 'evaluation_script':
                from zevo.evaluator_storage import verify_evaluator
                verify_evaluator(source, entry['evaluator_sha256'])
                runtime = package_runtime(source, root)
                if runtime:
                    runtimes[runtime['prefix']] = runtime
            dest = Path('inputs') / str(index) / field / source.name
            (root / dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, root / dest)
            entry[field] = str(Path(remote) / 'bundle' / dest)
    atomic_json(root / 'input.json', payload)
    package = Path(__file__).resolve().parents[1]
    shutil.copytree(package, root / 'src' / 'zevo', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Install the system package in an isolated environment on the compute
    # node, using the same declared dependencies as the deployed runner.
    project = package.parent.parent / 'pyproject.toml'
    if not project.is_file():
        raise ValueError('remote evaluation requires the deployed pyproject.toml')
    shutil.copyfile(project, root / 'pyproject.toml')
    (root / 'README.md').write_text('Zevo evaluation worker\n')
    atomic_json(root / 'runtimes.json', list(runtimes.values()))
    archive = local / 'evaluation-bundle.tar.gz'
    atomic_json(root / 'manifest.json', {
        str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()
    })
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(root, arcname='bundle')
    return archive


def launch_script(inp: EvaluationTaskInput, info: InfrastructureDeviceInfo, remote: str, *, compatibility_runtime=False, manifest_sha256='') -> str:
    q = shlex.quote
    route = info.cluster or info.instance
    contract = inp.slurm_job
    lines = ['#!/bin/bash']
    if contract.enabled:
        plan = info.resource_plan
        lines += [f'#SBATCH --job-name={contract.job_name}', f'#SBATCH --nodes={contract.nodes}',
                  '#SBATCH --ntasks-per-node=1', f'#SBATCH --gpus-per-node={contract.gpus_per_node}',
                  f'#SBATCH --cpus-per-task={max(1, plan.min_cpus // contract.nodes)}',
                  f'#SBATCH --mem={max(1, plan.min_ram_gb)}G',
                  f'#SBATCH --time={max(1, int(plan.time_limit_hours * 60))}',
                  f'#SBATCH --output={contract.stdout_path}', f'#SBATCH --error={contract.stderr_path}']
        for flag in ('partition', 'account', 'qos'):
            value = getattr(route, flag)
            if value:
                if any(c.isspace() for c in value):
                    raise ValueError(f'invalid Slurm {flag}')
                lines.append(f'#SBATCH --{flag}={value}')
        if route.container_image:
            lines.append(f'#SBATCH --container-image={route.container_image}')
            if compatibility_runtime:
                if any(c.isspace() or c in ',:' for c in remote):
                    raise ValueError('runtime container mount path contains unsupported characters')
                lines.append(f'#SBATCH --container-mounts={remote}:{remote},{remote}/bundle/compat:/app/data')
        elif compatibility_runtime:
            raise ValueError('this frozen scorer requires a compute container image to mount its versioned runtime; configure the cluster container_image')
        lines += ['set -euo pipefail', 'test -n "${SLURM_JOB_ID:-}" || { echo "evaluation requires a Slurm allocation" >&2; exit 1; }',
                  contract.lifecycle_prologue, contract.runtime_prologue]
    else:
        lines += ['set -euo pipefail']
        lines += [f'export CUDA_VISIBLE_DEVICES={q(info.instance.visible_devices)}']
    lines += ['umask 077', f'export ZEVO_TICKET_ID={q(inp.ticket_id)}',
              f'export ZEVO_EVALUATION_MANIFEST_SHA256={q(manifest_sha256)}',
              f'cd {q(remote)}',
              # The lock survives SSH disconnects and prevents duplicate direct
              # activations from scoring concurrently in the same work directory.
              'exec 9>.evaluation.lock', 'flock -n 9 || exit 0',
              'zevo_eval_finish() { rc=$?; trap - EXIT; rm -f .judge-env.json; printf "%s\\n" "$rc" > exit-code.tmp; mv exit-code.tmp exit-code; '
              + ('zevo_slurm_event EXITED "$rc"; ' if contract.enabled else '') + 'exit "$rc"; }',
              'trap zevo_eval_finish EXIT', 'rm -f exit-code results/result.json',
              route.env_setup or ':',
              'tar -xzf evaluation-bundle.tar.gz',
              'python3 -m venv --system-site-packages worker-env',
              'worker-env/bin/python -m pip install --disable-pip-version-check ./bundle > bootstrap.log 2>&1',
              'worker-env/bin/python -m zevo.engine.remote_evaluation --bootstrap-runtimes bundle/runtimes.json',
              'export ZEVO_EVALUATION_WORKER=1',
              'export PATH="$PWD/worker-env/bin:$PATH"',
              'export PYTHONPATH="$PWD/bundle/src${PYTHONPATH:+:$PYTHONPATH}"',
              'worker-env/bin/python -m zevo.engine.remote_evaluation --worker bundle/input.json --output results --environment .judge-env.json']
    return '\n'.join(lines) + '\n'


async def collect(info: InfrastructureDeviceInfo, remote: str, local: Path, ticket_id: str) -> EvaluationResult:
    local.mkdir(parents=True, exist_ok=True)
    await command(build_download_command(info.ssh, remote_paths=[remote + '/results'], local_dir=str(local), recursive=True))
    root = local / 'results'
    result = EvaluationResult.model_validate_json((root / 'result.json').read_text())
    if result.ticket_id != ticket_id:
        raise ValueError('remote evaluation returned a different Ticket')
    if result.status not in {'succeeded', 'failed'}:
        raise ValueError('remote evaluation did not return a terminal result')
    def mapped(value: str) -> str:
        path = Path(value)
        relative = path.relative_to(Path(remote) / 'results')
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file() or target.is_symlink():
            raise ValueError('missing or unsafe remote evaluation artifact')
        from zevo.engine.agent.drivers.evaluation_runner import _validated_score
        _validated_score(target)
        return str(target)
    if result.status == 'succeeded':
        result.metrics_path = mapped(result.metrics_path)
        for member in result.suite_members:
            member.metrics_path = mapped(member.metrics_path)
            from zevo.engine.agent.drivers.evaluation_runner import _validated_score
            if _validated_score(Path(member.metrics_path))[1] != member.score:
                raise ValueError('remote benchmark score disagrees with its metrics artifact')
    return result


async def run_remote(inp: EvaluationTaskInput, workspace_dir: str, event_sink=None) -> DriverRunResult:
    from zevo.engine.agent.drivers.evaluation_runner import _failed, _phase
    info = InfrastructureDeviceInfo.model_validate_json(Path(inp.device_info_path).read_text())
    remote = inp.remote_work_dir
    if not remote.startswith('/') or any(c in remote for c in '\n\r\0'):
        raise ValueError('remote evaluation requires an absolute work directory')
    local = Path(workspace_dir).resolve()
    contract = inp.slurm_job
    if info.provider == 'cluster' and not contract.enabled:
        raise ValueError('cluster evaluation requires an engine-owned Slurm contract')
    try:
        if contract.enabled and contract.phase == 'collect':
            if contract.scheduler_state not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}:
                result = EvaluationResult(ticket_id=inp.ticket_id, status='deferred', metrics_path='', error_message='', notes='Waiting for evaluation compute job')
            else:
                result = await collect(info, remote, local, inp.ticket_id)
            return DriverRunResult(output=result, driver='evaluation_runner')
        # A direct-host reconnect must not overwrite an active worker's inputs
        # or credentials. The lock is held for the whole remote process tree.
        running = False
        if not contract.enabled:
            state = await command(ssh(info, f'if test -d {shlex.quote(remote)}; then cd {shlex.quote(remote)}; '
                'flock -n .evaluation.lock true && echo idle || echo running; else echo idle; fi'))
            running = state.strip() == 'running'
        if running:
            return await wait_direct(inp, info, remote, local, event_sink)
        _phase(event_sink, inp.ticket_id, 'preparing_remote_evaluation')
        await command(ssh(info, f'umask 077; mkdir -p -- {shlex.quote(remote)}'))
        with tempfile.TemporaryDirectory(prefix='zevo-evaluation-') as directory:
            temp = Path(directory)
            archive = make_bundle(inp, temp, remote)
            (local / 'evaluation-bundle.sha256').write_text(digest(temp / 'bundle' / 'manifest.json') + '\n')
            compatibility_runtime = bool(json.loads((temp / 'bundle/runtimes.json').read_text()))
            body = launch_script(inp, info, remote, compatibility_runtime=compatibility_runtime,
                                 manifest_sha256=digest(temp / 'bundle/manifest.json'))
            for argv in build_upload_commands(info.ssh, sources=[str(archive)], remote_dir=remote):
                await command(argv, timeout=600)
        # File staging is allowed on login; installation and scoring remain
        # behind the allocation guard in the submitted script. Container bind
        # sources must exist before Slurm can start the container step.
        await command(ssh(info, f'cd {shlex.quote(remote)} && tar -xzf evaluation-bundle.tar.gz'), timeout=600)
        # Send credentials only on stdin to a private file, never in shell args,
        # task payloads, persisted local artifacts or logged command text.
        await command(ssh(info, 'umask 077; '
                      f'zevo_eval_secret=$(mktemp {shlex.quote(remote + "/.judge-env.XXXXXX")}) && '
                      'cat > "$zevo_eval_secret" && '
                      f'mv -f -- "$zevo_eval_secret" {shlex.quote(remote + "/.judge-env.json")}'),
                      stdin=json.dumps(judge_environment()).encode())
        script = Path(contract.script_path) if contract.enabled else local / 'evaluate-remote.sh'
        script.write_text(body)
        destination = contract.remote_work_dir if contract.enabled else remote
        for argv in build_upload_commands(info.ssh, sources=[str(script)], remote_dir=destination):
            await command(argv)
        if contract.enabled:
            result = EvaluationResult(ticket_id=inp.ticket_id, status='deferred', metrics_path='', error_message='',
                                      notes='Evaluation prepared for engine-owned Slurm submission')
            return DriverRunResult(output=result, driver='evaluation_runner')
        _phase(event_sink, inp.ticket_id, 'scoring_on_compute')
        # SSH is only a transport. A detached marked process owns the work;
        # reconnecting reads the same result and benchmark receipts.
        await command(ssh(info, f'cd {shlex.quote(remote)} && rm -f exit-code && '
            '(nohup bash evaluate-remote.sh > worker.log 2>&1 < /dev/null & echo $! > worker.pid)'))
        return await wait_direct(inp, info, remote, local, event_sink)
    except (OSError, ValueError, asyncio.TimeoutError) as exc:
        return DriverRunResult(output=_failed(inp.ticket_id, str(exc)), exit_code=1, driver='evaluation_runner')


async def wait_direct(inp, info, remote, local, event_sink):
    previous = None
    while True:
        state = await command(ssh(info, f'cd {shlex.quote(remote)} && '
            'if test -s exit-code; then cat exit-code; '
            'elif test -s worker.pid && kill -0 "$(cat worker.pid)" 2>/dev/null; then echo running; '
            'else echo lost; fi'))
        if state.strip() != 'running':
            if state.strip() == 'lost':
                raise ValueError('remote evaluation process disappeared; completed benchmark receipts are retained')
            try:
                result = await collect(info, remote, local, inp.ticket_id)
            except (OSError, ValueError):
                raise ValueError(f'remote evaluation exited {state.strip()}; inspect {remote}/worker.log and bootstrap.log') from None
            return DriverRunResult(output=result, driver='evaluation_runner', exit_code=0 if result.status == 'succeeded' else 1)
        progress = await command(ssh(info, f'cat {shlex.quote(remote + "/results/progress.json")} 2>/dev/null || true'))
        if progress and progress != previous and event_sink:
            value = json.loads(progress)
            if value.get('payload', {}).get('owner') == inp.ticket_id and value.get('type') in {'phase', 'progress'}:
                event_sink(value)
            previous = progress
        await asyncio.sleep(10)


def worker(input_path: Path, output: Path, environment: Path) -> int:
    from zevo.engine.agent.drivers.evaluation_runner import EvaluationRunnerDriver
    provider_scope = _provider_scope()
    env = json.loads(environment.read_text())
    environment.unlink()
    inp = EvaluationTaskInput.model_validate_json(input_path.read_text())
    if inp.device_info_path or inp.slurm_job.enabled:
        raise ValueError('compute worker cannot dispatch another remote job')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = input_path.parent / 'manifest.json'
    if manifest.is_file():
        expected = os.environ.get('ZEVO_EVALUATION_MANIFEST_SHA256')
        if expected and digest(manifest) != expected:
            raise ValueError('evaluation bundle manifest changed after submission')
        inventory = json.loads(manifest.read_text())
        if expected:
            for name, checksum in inventory.items():
                path = (input_path.parent / name).resolve()
                if not path.is_relative_to(input_path.parent.resolve()) or digest(path) != checksum:
                    raise ValueError(f'evaluation bundle file changed after submission: {name}')
        # Input hashes are already bound separately per benchmark. Only code
        # and environment changes invalidate every completed benchmark.
        runtime_files = {name: value for name, value in inventory.items() if not name.startswith('inputs/') and name != 'input.json'}
        import sys
        runtime_files['python'] = sys.version
        runtime_files['packages'] = sorted(
            (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()
            if dist.metadata['Name']
        )
        os.environ['ZEVO_EVALUATION_RUNTIME_IDENTITY'] = hashlib.sha256(json.dumps(runtime_files, sort_keys=True).encode()).hexdigest()
    report = {'schema_version': 1, 'ticket_id': inp.ticket_id,
              'started_at': time.time(), 'status': 'running', 'phases': []}
    atomic_json(output / 'phase-report.json', report)
    def event(value):
        atomic_json(output / 'progress.json', value)
        payload = value.get('payload', {})
        if value.get('type') == 'phase':
            report['phases'].append({'phase': payload.get('phase', ''), 'at': time.time()})
            atomic_json(output / 'phase-report.json', report)
            # Human-readable suite names may contain spaces; the marker phase
            # remains a bounded identifier and progress carries the suite name.
            print(f'__PHASE__:{inp.ticket_id}:scoring@{time.time()}', flush=True)
        elif value.get('type') == 'progress':
            print(f'__PROGRESS__:{inp.ticket_id}:' + json.dumps({**payload, 't': time.time()}), flush=True)
    async def run():
        return await EvaluationRunnerDriver().run_agent(blueprint=None, input_payload=inp,
            workspace_dir=str(output), event_sink=event)
    # Provider scope is explicitly restricted to the transferred judge keys.
    os.environ.update(env)
    token = provider_scope.set_scope(provider_scope.ProviderScope(keys=env)) if provider_scope else None
    try:
        result = asyncio.run(run())
    except Exception as exc:
        from zevo.engine.agent.drivers.evaluation_runner import _failed
        result = DriverRunResult(output=_failed(inp.ticket_id, str(exc)), exit_code=1)
    finally:
        if provider_scope:
            provider_scope.reset_scope(token)
    atomic_json(output / 'result.json', result.output.model_dump(mode='json'))
    report.update(status=result.output.status, finished_at=time.time())
    atomic_json(output / 'phase-report.json', report)
    return result.exit_code


def bootstrap_runtimes(manifest: Path) -> None:
    """Runs only inside the compute environment, never on a login node."""
    import subprocess
    import sys
    for runtime in json.loads(manifest.read_text()):
        prefix = Path(runtime['prefix'])
        # A configured direct host may already have the exact frozen runtime.
        # Otherwise the directory must be the container's staged bind mount.
        requirements = prefix / 'requirements.txt'
        if not requirements.is_file():
            executable = prefix / 'bin/python'
            inventory = manifest.resolve().parent / 'compat' / prefix.relative_to('/app/data') / 'requirements.txt'
            if not executable.is_file():
                raise ValueError(f'{prefix} is not mounted in the compute environment; configure the evaluator runtime before scoring')
            # A direct host's pre-provisioned runtime is shared. Verify it;
            # never install into or mutate a shared absolute interpreter.
            check = ('import importlib.metadata as m,json,sys; '
                     'wanted=json.loads(sys.argv[1]); '
                     'assert all(m.version(n)==v for n,v in wanted), "evaluator runtime versions differ"')
            wanted = [line.split('==', 1) for line in inventory.read_text().splitlines() if line]
            subprocess.run([str(executable), '-c', check, json.dumps(wanted)], check=True)
            continue
        version = f'{sys.version_info.major}.{sys.version_info.minor}'
        python = sys.executable if runtime['python'] == version else shutil.which('python' + runtime['python'])
        if not python:
            raise ValueError(f'frozen evaluation requires Python {runtime["python"]} on the compute node')
        subprocess.run([python, '-m', 'venv', '--system-site-packages', str(prefix)], check=True)
        executable = str(prefix / 'bin/python')
        subprocess.run([executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        subprocess.run([executable, '-m', 'pip', 'install', '--no-deps', str(manifest.resolve().parent)], check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-runtimes', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--environment', type=Path)
    args = parser.parse_args()
    if args.bootstrap_runtimes:
        bootstrap_runtimes(args.bootstrap_runtimes)
        raise SystemExit(0)
    if not (args.worker and args.output and args.environment):
        parser.error('--worker, --output and --environment are required')
    raise SystemExit(worker(args.worker, args.output, args.environment))
