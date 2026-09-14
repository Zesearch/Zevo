"""Coarse stage sizing stays dynamic while honoring each cluster's rules."""
from __future__ import annotations

import pytest

from zevo.contracts.infrastructure import (
    InfrastructureDeviceInfo,
    InfrastructureResourcePlan,
)
from zevo.engine.run.resource_planning import (
    model_parameter_billions,
    plan_stage_resources,
)


def _cluster_info(
    *,
    plan_gpus: int = 8,
    plan_nodes: int = 1,
    constraints: dict | None = None,
) -> InfrastructureDeviceInfo:
    plan = InfrastructureResourcePlan(
        purpose="train",
        required_working_set_gib=40,
        host_memory_components_gib={"model_and_runtime": 40},
        host_memory_formula_gib=64,
        site_min_ram_gib=0,
        num_gpus=plan_gpus,
        nodes=plan_nodes,
        min_vram_gb=40,
        min_ram_gb=64,
        min_cpus=8,
        time_limit_hours=8,
        disk_gb=0,
        rationale="rough Train envelope with safety room",
    )
    return InfrastructureDeviceInfo(
        run_id="run-1",
        ticket_id="infra-run-1",
        purpose="train",
        provider="cluster",
        auto_release=False,
        host="login.example",
        ssh={
            "host": "login.example", "port": 22, "user": "user",
            "key_path": "/key",
        },
        cluster={
            "requested_gpus": plan_gpus,
            "nodes": plan_nodes,
            "workdir": "/project/zevo/run-1",
            "hf_cache": "/project/zevo/cache/hf",
            "gpu_constraints": constraints,
        },
        cost={"dph_total": 0},
        resource_plan=plan,
        probe_source="ssh-environment",
        probed_at="2026-09-14T12:00:00Z",
    )


def _beta_constraints() -> dict:
    return {
        "min_gpus_per_job": 4,
        "allocation_step": 4,
        "gpus_per_node": 4,
        "gpu_vram_gib": 189,
        "whole_node": True,
        "source": "live beta QOS MinTRES and sinfo",
    }


def test_model_size_hint_is_read_without_a_network_lookup() -> None:
    assert model_parameter_billions("allenai/Olmo-3.1-32B-Instruct-SFT") == 32
    assert model_parameter_billions("org/model-without-size") is None


def test_beta_rounds_small_inference_estimate_to_its_four_gpu_minimum() -> None:
    info = _cluster_info(
        plan_gpus=8, plan_nodes=2, constraints=_beta_constraints(),
    )
    selected = plan_stage_resources(
        stage="inference",
        base_model="allenai/Olmo-3.1-32B-Instruct-SFT",
        info=info,
        maximum_gpus=8,
    )
    assert selected.estimated_gpus == 1
    assert (selected.num_gpus, selected.nodes, selected.gpus_per_node) == (4, 1, 4)
    assert "tier 4" in selected.rationale


def test_same_model_uses_two_gpu_tier_on_a_partial_eighty_gib_node() -> None:
    info = _cluster_info(constraints={
        "min_gpus_per_job": 1,
        "allocation_step": 1,
        "gpus_per_node": 8,
        "gpu_vram_gib": 80,
        "whole_node": False,
        "source": "live generic partition",
    })
    selected = plan_stage_resources(
        stage="inference",
        base_model="allenai/Olmo-3.1-32B-Instruct-SFT",
        info=info,
        maximum_gpus=8,
    )
    assert (selected.estimated_gpus, selected.num_gpus, selected.nodes) == (2, 2, 1)


def test_train_retains_the_infrastructure_estimate_and_data_starts_small() -> None:
    info = _cluster_info(
        plan_gpus=8, plan_nodes=2, constraints=_beta_constraints(),
    )
    train = plan_stage_resources(
        stage="train", base_model="org/32B", info=info, maximum_gpus=8,
    )
    data = plan_stage_resources(
        stage="data", base_model="org/32B", info=info, maximum_gpus=8,
    )
    assert (train.num_gpus, train.nodes) == (8, 2)
    assert (data.estimated_gpus, data.num_gpus, data.nodes) == (1, 4, 1)


def test_beta_minimum_never_silently_exceeds_a_run_cap() -> None:
    info = _cluster_info(
        plan_gpus=8, plan_nodes=2, constraints=_beta_constraints(),
    )
    with pytest.raises(ValueError, match="no valid inference GPU tier"):
        plan_stage_resources(
            stage="inference", base_model="org/7B", info=info, maximum_gpus=2,
        )


def test_collect_preserves_the_registered_job_shape() -> None:
    info = _cluster_info(plan_nodes=2, constraints={
        "min_gpus_per_job": 1,
        "allocation_step": 1,
        "gpus_per_node": 4,
        "whole_node": False,
        "source": "live scheduler",
    })
    selected = plan_stage_resources(
        stage="train",
        base_model="org/model",
        info=info,
        maximum_gpus=8,
        registered_gpus=6,
        registered_nodes=2,
    )
    assert (selected.num_gpus, selected.nodes, selected.gpus_per_node) == (6, 2, 3)
    assert selected.source == "registered scheduler job"


def test_old_artifact_uses_known_valid_per_node_shape_conservatively() -> None:
    info = _cluster_info(plan_gpus=8, plan_nodes=2, constraints=None)
    selected = plan_stage_resources(
        stage="inference", base_model="org/7B", info=info, maximum_gpus=8,
    )
    assert (selected.num_gpus, selected.nodes) == (4, 1)
    assert selected.source == "legacy conservative constraints"
