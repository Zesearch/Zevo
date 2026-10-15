"""A cloud device_info carries the same direct route as a fixed instance.

Run b48530a3 (2026-10-15) rented an EC2 GPU and then failed at Data: ticket
creation requires a device binding for a Hub-pinned dataset, and the Data
input builder reads the remote work directory and HF cache from the cluster or
instance route, which a cloud device was forbidden to carry. The rented box is
reached exactly like a fixed instance, so it declares the same block."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.infrastructure import InfrastructureDeviceInfo, InfrastructureResourcePlan


def _plan(purpose="train"):
    return InfrastructureResourcePlan(
        purpose=purpose, required_working_set_gib=12,
        host_memory_components_gib={"model_and_runtime": 12}, host_memory_formula_gib=32,
        site_min_ram_gib=0, num_gpus=1, min_vram_gb=16, min_ram_gb=32, min_cpus=4,
        time_limit_hours=0, cloud_backend="aws", disk_gb=200, rationale="test",
    )


def _cloud(**overrides):
    body = dict(
        run_id="run-1", ticket_id="infra-run-001", purpose="train", provider="cloud",
        cloud_backend="aws", instance_id="i-0123456789abcdef0", auto_release=True,
        host="203.0.113.5",
        ssh={"host": "203.0.113.5", "port": 22, "user": "ubuntu", "key_path": "/root/.ssh/id_ed25519"},
        gpu={"gpu_count": 1, "gpu_name": "NVIDIA L40S", "vram_gb": 44, "vram_mb": 46068,
             "devices": [{"index": 0, "name": "NVIDIA L40S", "vram_mb": 46068}]},
        cuda={"driver_version": "595.91.07", "cuda_version": "13.0", "recommended_torch_index": ""},
        cost={"dph_total": 1.82}, resource_plan=_plan(), probe_source="remote_nvidia-smi",
        probed_at="2026-10-15T22:44:00Z",
        instance={"workdir": "/home/ubuntu/zevo/run-1", "hf_cache": "/home/ubuntu/zevo/hf_cache",
                  "visible_devices": "0"},
    )
    body.update(overrides)
    return body


def test_cloud_device_carries_the_direct_route() -> None:
    info = InfrastructureDeviceInfo(**_cloud())
    route = info.cluster or info.instance
    assert route is not None
    assert route.workdir == "/home/ubuntu/zevo/run-1" and route.hf_cache == "/home/ubuntu/zevo/hf_cache"


def test_cloud_device_without_the_route_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires instance routing"):
        InfrastructureDeviceInfo(**_cloud(instance=None))


def test_cloud_device_still_rejects_cluster_routing() -> None:
    with pytest.raises(ValidationError, match="must not contain cluster routing"):
        InfrastructureDeviceInfo(**_cloud(cluster={
            "requested_gpus": 1, "nodes": 1, "workdir": "/x/run-1", "hf_cache": "/x/hf",
        }))
