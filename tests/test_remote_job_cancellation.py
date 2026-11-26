from __future__ import annotations

import subprocess

from zevo.engine.run.remote_jobs import (
    _direct_process_kill_command,
    _slurm_job_kill_command,
)


def test_cloud_cleanup_matches_exact_ticket_environment() -> None:
    command = _direct_process_kill_command("train-12345678-002")
    assert "ZEVO_TICKET_ID=train-12345678-002".encode().hex() in command
    assert "SIGTERM" in command and "SIGKILL" in command
    assert "pkill" not in command


def test_instance_cleanup_is_direct_and_ticket_scoped() -> None:
    command = _direct_process_kill_command("train-12345678-002")
    assert "ZEVO_TICKET_ID=train-12345678-002".encode().hex() in command
    assert "squeue" not in command
    assert "srun" not in command
    assert "scancel" not in command


def test_cluster_cleanup_targets_exact_finite_stage_job() -> None:
    command = _slurm_job_kill_command("12345", "train-12345678-002")
    assert "zevo-train-12345678-002" in command
    assert "/etc/profile.d/modules.sh" in command
    assert "module load default-environment" in command
    assert "slurm-squeue-unavailable" in command
    assert "slurm-scancel-unavailable" in command
    assert "stage-job-query-failed" in command
    assert "squeue -h -j 12345" in command
    assert "scancel 12345" in command
    assert "--overlap" not in command
    assert subprocess.run(
        ["bash", "-n", "-c", command], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).returncode == 0


import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zevo.engine.run import remote_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,state,resize,uses_marker', [
    ('per_submission', 'PENDING', False, False),
    ('per_submission', 'RUNNING', False, False),
    ('per_run', 'PENDING', False, False),
    ('per_run', 'RUNNING', False, True),
    ('per_run', 'RUNNING', True, False),
])
async def test_restart_cancels_pending_or_replaced_allocations(
    monkeypatch, mode, state, resize, uses_marker,
):
    row = SimpleNamespace(instance_id='12345', release_reason='', meta={
        'gpu_allocation_mode': mode, 'allocation_state': state,
        'controller_directory': '/run/controller', 'controller_request_id': 'request',
        'controller_outcome_path': '/run/controller/outcomes/request.json',
    })
    info = SimpleNamespace(provider='cluster', cluster=SimpleNamespace(env_setup=''))
    monkeypatch.setattr(remote_jobs, '_device_info_for_ticket', AsyncMock(return_value=info))
    monkeypatch.setattr(remote_jobs, '_open_cluster_job', AsyncMock(return_value=row))
    monkeypatch.setattr(remote_jobs, 'allocation_for_resize', AsyncMock(return_value=row))
    ssh = AsyncMock(return_value={'ok': True})
    monkeypatch.setattr(remote_jobs, '_ssh', ssh)
    db = SimpleNamespace(flush=AsyncMock())
    result = await remote_jobs.cancel_ticket_remote_job(
        db, SimpleNamespace(id='infer-12345678-001', agent_id='inference'),
        preserve_allocation=True, resize_allocation=resize,
    )
    assert result['ok']
    command = ssh.call_args.args[1]
    if uses_marker:
        assert 'scancel' not in command
        assert '/run/controller/cancel/request' in command
    else:
        assert 'scancel 12345' in command
        assert 'cancellation_requested_at' in row.meta
        assert row.release_reason == 'cancellation requested; awaiting scheduler confirmation'
