"""Run probe commands through a shell with site-only Slurm executables."""
import asyncio
import shlex

import pytest

from zevo.engine.run import slurm_capacity
from tests.test_stage_resource_planning import _cluster_info, _partial_node_constraints


def executable(path, body):
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(0o755)


@pytest.mark.parametrize('reservation', ['', 'reserved'])
@pytest.mark.parametrize('qos', ['', 'research'])
def test_capacity_probe_initializes_site_environment(tmp_path, monkeypatch, reservation, qos):
    info = _cluster_info(constraints=_partial_node_constraints())
    info.cluster.reservation = reservation
    info.cluster.qos = qos
    info.cluster.env_setup = (
        'module() { export PATH=' + shlex.quote(str(tmp_path)) + ':"$PATH"; }; '
        'module load shared slurm; echo site-setup-noise'
    )
    executable(tmp_path / 'scontrol', "echo 'ReservationName=reserved StartTime=2000-01-01T00:00:00 EndTime=2099-01-01T00:00:00 Nodes=gpu01'\n")
    executable(tmp_path / 'sinfo', "echo 'gpu01|gpu:8|gpu:4|mix'\n")
    executable(tmp_path / 'sacctmgr', "echo 'research|gres/gpu=8'\n")
    executable(tmp_path / 'squeue', "echo 'research gres/gpu=2'\n")
    real_exec = asyncio.create_subprocess_exec
    commands = []
    async def local_ssh(*args, **kwargs):
        commands.append(args[-1])
        return await real_exec('/bin/bash', '-c', args[-1], **kwargs)
    monkeypatch.setattr(slurm_capacity.asyncio, 'create_subprocess_exec', local_ssh)
    snapshot = asyncio.run(slurm_capacity.probe_slurm_capacity(info))
    assert snapshot.free_gpus_by_node == (4,)
    assert snapshot.qos_free_gpus == (6 if qos else None)
    assert all(command.startswith('bash -lc ') for command in commands)
    assert len(commands) == (2 if reservation else 1)


def test_probe_error_keeps_exit_and_actual_error_after_banner(monkeypatch):
    info = _cluster_info(constraints=_partial_node_constraints())
    class Failed:
        returncode = 127
        async def communicate(self):
            return b'', b'Login banner\n' * 100 + b'bash: scontrol: command not found\n'
    async def fake_exec(*args, **kwargs):
        return Failed()
    monkeypatch.setattr(slurm_capacity.asyncio, 'create_subprocess_exec', fake_exec)
    with pytest.raises(ValueError, match=r'exit 127.*scontrol: command not found') as error:
        asyncio.run(slurm_capacity._query(info, 'scontrol show reservation reserved', required=('scontrol',)))
    assert 'Login banner' not in str(error.value)
