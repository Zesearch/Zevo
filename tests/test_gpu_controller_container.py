"""Exercise the remote controller without requiring a live GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from zevo.engine.run.gpu_controller import CONTROLLER_SOURCE, controller_script


def executable(path, body):
    path.write_text(body)
    path.chmod(0o755)


def request(root, name, body):
    script = root / (name + '.sbatch')
    script.write_text(body)
    (root / 'requests' / (name + '.json')).write_text(json.dumps({
        'script': str(script), 'sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'ticket_id': name, 'run_id': 'run', 'cwd': str(root),
        'stdout': str(root / (name + '.out')), 'stderr': str(root / (name + '.err')),
    }))


def setup(tmp_path):
    for name in ['requests', 'outcomes', 'cancel', 'bin', 'proc']:
        (tmp_path / name).mkdir()
    (tmp_path / 'bin/bash').symlink_to('/bin/bash')
    return dict(os.environ, PATH=str(tmp_path / 'bin'), SLURM_JOB_ID='123')


def test_container_workload_then_native_reuses_host_controller(tmp_path):
    env = setup(tmp_path)
    executable(tmp_path / 'bin/squeue', '#!/bin/bash\nexit 0\n')
    executable(tmp_path / 'bin/scancel', '#!/bin/bash\nexit 0\n')
    executable(tmp_path / 'bin/srun', f'''#!{sys.executable}
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
pathlib.Path({str(tmp_path / 'srun-args.json')!r}).write_text(json.dumps(args))
pos = args.index('bash')
sys.exit(subprocess.call(args[pos:], env=dict(os.environ, INSIDE_CONTAINER='yes')))
''')
    request(tmp_path, '01-container', '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --container-image=/images/test.sqsh
#SBATCH --container-mounts="/shared path:/shared path"
#SBATCH --container-workdir=/shared
[[ "$INSIDE_CONTAINER" == yes ]] || exit 9
echo container-ok
''')
    request(tmp_path, '02-native', '''#!/bin/bash
[[ -z "$INSIDE_CONTAINER" ]] || exit 10
echo native-ok
''')
    result = subprocess.run([sys.executable, '-c', CONTROLLER_SOURCE.replace('"/proc"', repr(str(tmp_path / 'proc'))), str(tmp_path), '1', '0'],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for name in ['01-container', '02-native']:
        assert json.loads((tmp_path / 'outcomes' / (name + '.json')).read_text())['exit_code'] == 0
    args = json.loads((tmp_path / 'srun-args.json').read_text())
    assert '--container-image=/images/test.sqsh' in args
    assert '--container-mounts=/shared path:/shared path' in args
    assert '--container-workdir=/shared' in args
    assert '--ntasks=1' in args
    assert '--overlap' in args


def test_missing_slurm_tool_preserves_workload_exit_and_reports_cause(tmp_path):
    env = setup(tmp_path)
    request(tmp_path, 'work', '#!/bin/bash\nexit 0\n')
    result = subprocess.run([sys.executable, '-c', CONTROLLER_SOURCE, str(tmp_path), '1', '1'],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 125
    assert 'FileNotFoundError' in result.stderr and 'squeue' in result.stderr
    assert 'workload exit=0' in result.stderr
    assert not (tmp_path / 'outcomes/work.json').exists()
    diagnostic = json.loads((tmp_path / 'cleanup-failure-work.json').read_text())
    assert diagnostic == {'workload_exit_code': 0, 'workload_error': '', 'cleanup_confirmed': False}


def test_controller_header_keeps_resources_but_not_container(tmp_path):
    script = controller_script('''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --container-image=/images/test.sqsh
#SBATCH --container-mounts=/data:/data
#SBATCH --container-workdir=/data
''', str(tmp_path), env_setup='source /container-only/env.sh')
    assert '#SBATCH --gres=gpu:4' in script
    assert '#SBATCH --mem=256G' in script
    assert '#SBATCH --container' not in script
    assert 'source /container-only/env.sh' not in script
    assert 'command -v squeue' in script and 'command -v scancel' in script
    assert 'command -v srun' in script
    assert subprocess.run(['/bin/bash', '-n'], input=script, text=True).returncode == 0


def test_reuse_snapshot_preserves_container_of_host_launched_workload():
    from zevo.engine.run.inference_launch import effective_snapshot
    body = '#!/bin/bash\n#SBATCH --container-image=verified.sqsh\necho workload\n'
    source = {'body': body, 'sha256': hashlib.sha256(body.encode()).hexdigest()}
    headers = '#SBATCH --nodes=1\n#SBATCH --gres=gpu:4\n'
    assert effective_snapshot(source, headers, 'host_srun_v1') == source
    assert '--container-image' not in effective_snapshot(source, headers)['body']


def test_saved_stage_yield_retires_retained_allocation(tmp_path):
    env = setup(tmp_path)
    executable(tmp_path / 'bin/squeue', '#!/bin/bash\nexit 0\n')
    executable(tmp_path / 'bin/scancel', '#!/bin/bash\nexit 0\n')
    request(tmp_path, 'data-work', '#!/bin/bash\necho "$ZEVO_STAGE_RESUME_ROOT"\nexit 75\n')
    request(tmp_path, 'later-work', '#!/bin/bash\necho should-not-run\n')
    result = subprocess.run([sys.executable, '-c', CONTROLLER_SOURCE.replace('"/proc"', repr(str(tmp_path / 'proc'))),
                             str(tmp_path), '1', '0'], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 75, result.stderr
    assert json.loads((tmp_path / 'outcomes/data-work.json').read_text())['exit_code'] == 75
    assert not (tmp_path / 'outcomes/later-work.json').exists()
    assert str(tmp_path / '.zevo-resume') in (tmp_path / 'data-work.out').read_text()
    assert '__PHASE__:data-work:starting_workload@' in (tmp_path / 'data-work.out').read_text()
