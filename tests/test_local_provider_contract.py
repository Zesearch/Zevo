"""The runtime has exactly three GPU provider categories."""
from __future__ import annotations

from typing import get_args

from zevo.contracts.infrastructure import (
    CreateInfraInstanceBody,
    InfraResult,
    InfraTaskInput,
    InfrastructureDeviceInfo,
    SlurmStageJobContract,
    slurm_lifecycle_prologue,
)
from zevo.db.models import InfraInstance
from zevo.engine.run.runner import _commit_slurm_submission


def test_provider_literals_are_exactly_cloud_cluster_instance() -> None:
    expected = {"cluster", "cloud", "instance"}
    for cls in (InfraTaskInput, InfrastructureDeviceInfo, CreateInfraInstanceBody):
        assert set(get_args(cls.model_fields["provider"].annotation)) == expected
    assert set(get_args(InfraResult.model_fields["provider"].annotation)) == expected | {""}


def test_cluster_stage_contract_names_a_finite_sbatch_artifact() -> None:
    status_path = "/remote/run/train-001/.zevo-slurm-status"
    contract = SlurmStageJobContract(
        enabled=True,
        script_path="/tmp/train.sbatch",
        job_name="zevo-train-001",
        status_path=status_path,
        stdout_path="/remote/run/train-001/slurm-%j.out",
        stderr_path="/remote/run/train-001/slurm-%j.err",
        lifecycle_prologue=slurm_lifecycle_prologue(status_path),
        num_gpus=2,
        infra_instance_create_schema={"type": "object"},
        infra_instance_patch_schema={"type": "object"},
        infra_instance_response_schema={"type": "object"},
    )
    assert contract.script_path.endswith(".sbatch")
    assert contract.num_gpus == 2
    assert contract.stdout_path.endswith("/slurm-%j.out")
    assert contract.stderr_path.endswith("/slurm-%j.err")
    assert "zevo_slurm_event RUNNING" in contract.lifecycle_prologue
    assert "zevo_slurm_event EXITED" in contract.lifecycle_prologue


def test_runner_commits_watcher_ownership_only_after_deferred_validation() -> None:
    row = InfraInstance(
        instance_id="12345", provider="cluster", status="provisioning",
        meta={
            "stage_job": True,
            "scheduler_state": "RUNNING",
            "remote_workdir": "/remote/run/train-001",
        },
    )
    assert _commit_slurm_submission(
        row,
        heartbeat_id="heartbeat-1",
        stdout_path="/remote/run/train-001/slurm-%j.out",
        stderr_path="/remote/run/train-001/slurm-%j.err",
    ) == "RUNNING"
    assert row.meta["submission_committed"] is True
    assert row.meta["submission_heartbeat_id"] == "heartbeat-1"
    assert row.meta["submission_committed_at"]
    assert row.meta["log_path"] == "/remote/run/train-001/slurm-12345.out"
    assert row.meta["stderr_path"] == "/remote/run/train-001/slurm-12345.err"
