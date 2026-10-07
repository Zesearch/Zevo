"""Run real non-interactive shells to reproduce missing site Slurm modules."""
import subprocess

from zevo.engine.run.remote_jobs import _slurm_cli_bootstrap_command, _slurm_job_kill_command


def site(tmp_path):
    folder = tmp_path / "site bin"
    folder.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel", "scontrol"):
        script = folder / name
        script.write_text("#!/bin/sh\necho site-" + name + "\n")
        script.chmod(0o755)
    setup = tmp_path / "site env.sh"
    setup.write_text(f'export PATH="{folder}:$PATH"\necho setup-noise\n')
    return f"source '{setup}'"


def test_configured_setup_exposes_slurm_without_polluting_job_id(tmp_path):
    setup = site(tmp_path)
    command = _slurm_cli_bootstrap_command(setup, required=("sbatch", "squeue", "sacct", "scancel"))
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", command + "sbatch"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == "site-sbatch\n"


def test_failed_setup_never_submits_or_falls_back(tmp_path):
    marker = tmp_path / "submitted"
    command = _slurm_cli_bootstrap_command("false") + f"touch '{marker}'"
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 126
    assert "slurm-env-setup-failed" in result.stderr
    assert not marker.exists()


def test_missing_command_is_explicit_and_stops_submission():
    command = _slurm_cli_bootstrap_command("export PATH=/nonexistent", required=("sbatch",)) + "echo submitted"
    result = subprocess.run(["/bin/bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 127
    assert "slurm-sbatch-unavailable" in result.stderr
    assert "submitted" not in result.stdout


def test_cleanup_initializes_same_environment(tmp_path):
    setup = site(tmp_path)
    # Return an empty queue: cleanup succeeds without cancelling another job.
    (tmp_path / "site bin" / "squeue").write_text("#!/bin/sh\nexit 0\n")
    result = subprocess.run(["bash", "-c", _slurm_job_kill_command("123", "train-test", setup)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "setup-noise" not in result.stdout
