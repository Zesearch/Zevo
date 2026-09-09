"""Cluster and Instance are the two SSH-backed provider categories."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from zevo.contracts.infrastructure import InfrastructureDeviceInfo, validate_device_info


def _device(provider: str) -> dict:
    is_instance = provider == "instance"
    is_cluster = provider == "cluster"
    return {
        "schema_version": 1,
        "run_id": "12345678-run",
        "ticket_id": "infra-12345678-001",
        "purpose": "inference",
        "provider": provider,
        "cloud_backend": "",
        "instance_id": "" if is_cluster else "job-1",
        "auto_release": False,
        "host": "login.example.com" if is_cluster else "gpu.example.com",
        "ssh": {"host": "login.example.com" if is_cluster else "gpu.example.com", "port": 22, "user": "user", "key_path": "/k"},
        "cluster": {
            "jobid": "",
            "node": "",
            "requested_gpus": 1,
            "workdir": "/scratch/zevo/run",
            "hf_cache": "/scratch/zevo/hf_cache",
        } if is_cluster else None,
        "instance": {
            "workdir": "/scratch/zevo/run",
            "hf_cache": "/scratch/zevo/hf_cache",
            "visible_devices": "0",
        } if is_instance else None,
        "gpu": None if is_cluster else {
            "has_gpu": True, "gpu_count": 1, "gpu_name": "A100",
            "vram_gb": 80, "vram_mb": 81920,
            "devices": [{"index": 0, "name": "A100", "vram_mb": 81920}],
        },
        "cuda": None if is_cluster else {"driver_version": "570", "cuda_version": "12"},
        "cost": {"dph_total": 0},
        "resource_plan": {
            "purpose": "inference",
            "required_working_set_gib": 12,
            "host_memory_components_gib": {"model_and_runtime": 12},
            "host_memory_formula_gib": 32,
            "site_min_ram_gib": 0,
            "num_gpus": 1, "min_vram_gb": 40, "min_ram_gb": 32, "min_cpus": 8,
            "time_limit_hours": 1, "disk_gb": 0,
            "rationale": "use an SSH-backed Slurm system",
        },
        "probe_source": "ssh-environment" if is_cluster else "remote_nvidia-smi",
        "probed_at": "2026-08-29T00:00:00Z",
    }


@pytest.mark.parametrize("provider", ["cluster", "instance"])
def test_ssh_backed_device_info_validates(provider: str) -> None:
    info = InfrastructureDeviceInfo.model_validate(_device(provider))
    assert info.provider == provider
    if provider == "cluster":
        assert info.instance_id == ""
        assert info.cluster is not None and info.cluster.jobid == ""
        assert info.gpu is None and info.cuda is None


def test_cluster_access_rejects_a_holder_allocation() -> None:
    body = _device("cluster")
    body["instance_id"] = "12345"
    body["cluster"]["jobid"] = "12345"
    body["cluster"]["node"] = "gpu-1"
    with pytest.raises(ValidationError, match="holder job|allocation id"):
        InfrastructureDeviceInfo.model_validate(body)


def test_run_gpu_value_is_a_limit_and_zero_is_unlimited(tmp_path) -> None:
    body = _device("cluster")
    body["cluster"]["requested_gpus"] = 2
    body["resource_plan"]["num_gpus"] = 2
    path = tmp_path / "device_info.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    unlimited = validate_device_info(
        path,
        run_id=body["run_id"],
        ticket_id=body["ticket_id"],
        provider="cluster",
        num_gpus=0,
        purpose="inference",
    )
    assert unlimited.resource_plan.num_gpus == 2

    with pytest.raises(ValueError, match="exceeds Run limit"):
        validate_device_info(
            path,
            run_id=body["run_id"],
            ticket_id=body["ticket_id"],
            provider="cluster",
            num_gpus=1,
            purpose="inference",
        )


@pytest.mark.parametrize("provider", ["ssh", "local"])
def test_removed_provider_literals_are_rejected(provider: str) -> None:
    with pytest.raises(ValidationError):
        InfrastructureDeviceInfo.model_validate(_device(provider))
