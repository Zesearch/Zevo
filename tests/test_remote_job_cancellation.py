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
