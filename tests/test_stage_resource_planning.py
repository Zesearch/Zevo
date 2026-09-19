"""Coarse stage sizing stays dynamic while honoring each cluster's rules."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from zevo.contracts.infrastructure import (
    InfrastructureDeviceInfo,
    InfrastructureResourcePlan,
)
from zevo.engine.run.resource_planning import (
    SlurmCapacitySnapshot,
    model_parameter_billions,
    plan_stage_resources,
)
from zevo.engine.run.slurm_capacity import parse_slurm_capacity
from zevo.engine.run import runner


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


def test_train_estimates_its_own_gpu_count_and_data_starts_small() -> None:
    info = _cluster_info(
        plan_gpus=8, plan_nodes=2, constraints=_beta_constraints(),
    )
    train = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, maximum_gpus=8,
    )
    data = plan_stage_resources(
        stage="data", base_model="org/32B", info=info, maximum_gpus=8,
    )
    assert (train.estimated_gpus, train.num_gpus, train.nodes) == (4, 4, 1)
    assert "full-parameter Adam/FSDP" in train.rationale
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
        live_capacity=SlurmCapacitySnapshot((0, 0), 0),
    )
    assert (selected.num_gpus, selected.nodes, selected.gpus_per_node) == (6, 2, 3)
    assert selected.source == "registered scheduler job"


def test_missing_cluster_constraints_are_not_silently_guessed() -> None:
    with pytest.raises(ValueError, match="gpu_constraints"):
        _cluster_info(plan_gpus=8, plan_nodes=2, constraints=None)


def _partial_node_constraints() -> dict:
    return {
        "min_gpus_per_job": 1,
        "allocation_step": 1,
        "gpus_per_node": 8,
        "gpu_vram_gib": 180,
        "whole_node": False,
        "source": "sinfo and QoS limits",
    }


def test_train_uses_smallest_safe_tier_even_when_more_gpus_are_idle() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    full = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, live_capacity=SlurmCapacitySnapshot((4, 4), 8),
    )
    assert (full.num_gpus, full.nodes, full.gpus_per_node) == (4, 1, 4)
    assert full.source == "live Slurm capacity"

    reduced = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, live_capacity=SlurmCapacitySnapshot((4, 4), 6),
    )
    assert (reduced.num_gpus, reduced.nodes, reduced.gpus_per_node) == (4, 1, 4)

    queued = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, live_capacity=SlurmCapacitySnapshot((2, 2), 2),
    )
    assert queued.num_gpus == 4
    assert "wait in Slurm" in queued.rationale


def test_full_train_needs_more_gpus_when_each_gpu_has_less_vram() -> None:
    constraints = {**_partial_node_constraints(), "gpu_vram_gib": 80}
    info = _cluster_info(plan_gpus=4, constraints=constraints)
    selected = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info,
    )
    assert (selected.estimated_gpus, selected.num_gpus) == (8, 8)
    with pytest.raises(ValueError, match="no valid train GPU tier"):
        plan_stage_resources(
            stage="train", base_model="org/32B", training_method="full_sft",
            info=info, maximum_gpus=4,
        )


def test_lora_and_inference_do_not_take_extra_gpus_merely_because_they_are_free() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    capacity = SlurmCapacitySnapshot((8, 8), 16)
    lora = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="lora_sft",
        info=info, live_capacity=capacity,
    )
    inference = plan_stage_resources(
        stage="inference", base_model="org/32B",
        info=info, live_capacity=capacity,
    )
    assert lora.num_gpus == 1
    assert inference.num_gpus == 1


def test_large_inference_suite_scales_to_independent_replicas() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="inference", base_model="org/32B", info=info,
        inference_members=17, inference_rows=30_000,
        live_capacity=SlurmCapacitySnapshot((4, 4), 8), maximum_gpus=8,
    )
    assert (selected.estimated_gpus, selected.num_gpus, selected.nodes) == (1, 4, 1)
    assert "4 model replica(s)" in selected.rationale


def test_inference_parallelism_respects_model_group_size_and_live_capacity() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="inference", base_model="org/32B", info=info,
        inference_members=17, inference_rows=30_000,
        inference_gpus_per_replica=2,
        live_capacity=SlurmCapacitySnapshot((4, 4), 6), maximum_gpus=8,
    )
    assert (selected.estimated_gpus, selected.num_gpus) == (2, 4)
    assert "2 model replica(s)" in selected.rationale

    queued = plan_stage_resources(
        stage="inference", base_model="org/32B", info=info,
        inference_members=17, inference_rows=30_000,
        inference_gpus_per_replica=2,
        live_capacity=SlurmCapacitySnapshot((1, 1), 2), maximum_gpus=8,
    )
    assert queued.num_gpus == 2


def test_small_inference_uses_one_model_group_even_with_idle_gpus() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="inference", base_model="org/32B", info=info,
        inference_members=1, inference_rows=50,
        live_capacity=SlurmCapacitySnapshot((8,), 8), maximum_gpus=8,
    )
    assert selected.num_gpus == 1


def test_probe_failure_keeps_minimum_safe_shape() -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, capacity_error="Slurm capacity probe timed out",
    )
    assert selected.num_gpus == 4
    assert "timed out" in selected.rationale


def test_train_uses_infra_fallback_when_model_size_is_unknown() -> None:
    info = _cluster_info(plan_gpus=8, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="train", base_model="org/model-without-size",
        training_method="full_sft", info=info,
    )
    assert selected.num_gpus == 8
    assert "conservative fallback" in selected.rationale


def test_train_uses_infra_fallback_when_method_is_not_selected() -> None:
    info = _cluster_info(plan_gpus=8, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="train", base_model="org/32B", info=info,
    )
    assert selected.num_gpus == 8
    assert "conservative fallback" in selected.rationale


def test_pinned_train_world_size_is_exact_and_must_fit() -> None:
    info = _cluster_info(plan_gpus=8, constraints=_partial_node_constraints())
    selected = plan_stage_resources(
        stage="train", base_model="org/32B", training_method="full_sft",
        info=info, train_world_size_pin=6, maximum_gpus=8,
    )
    assert (selected.estimated_gpus, selected.num_gpus, selected.nodes) == (4, 6, 1)
    assert selected.source == "pinned Train world_size"
    with pytest.raises(ValueError, match="below estimated GPU requirement"):
        plan_stage_resources(
            stage="train", base_model="org/32B", training_method="full_sft",
            info=info, train_world_size_pin=2,
        )


def test_slurm_snapshot_parses_node_and_qos_headroom() -> None:
    raw = (
        "node-a |gpu:b200:8(S:0-1)|gpu:b200:4(IDX:0-3)|mix-\n"
        "node-b |gpu:b200:8(S:0-1)|gpu:b200:4(IDX:0-3)|mix-\n"
        "cpu-only|(null)|(null)|idle\n"
        "node-c |gpu:b200:8(S:0-1)|gpu:b200:0|drain\n"
        "__ZEVO_QOS_LIMIT__\n"
        "research|cpu=180,gres/gpu=10,mem=1440000M\n"
        "__ZEVO_RUNNING_QOS_JOBS__\n"
        "research cpu=16,mem=100G,node=1,gres/gpu=2,gres/gpu:b200=2\n"
        "research cpu=8,mem=64G,node=1,gres/gpu=2,gres/gpu:b200=2\n"
    )
    snapshot = parse_slurm_capacity(raw, qos="research")
    assert snapshot.free_gpus_by_node == (4, 4)
    assert snapshot.qos_free_gpus == 6
    assert snapshot.immediately_free_gpus == 6


def test_incomplete_qos_snapshot_fails_closed_to_static_planning() -> None:
    raw = "node-a|gpu:8|gpu:0|idle\n"
    with pytest.raises(ValueError, match="QoS limits"):
        parse_slurm_capacity(raw, qos="research")


def test_stage_contract_probes_only_before_new_submission(tmp_path, monkeypatch) -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    path = tmp_path / "device_info.json"
    path.write_text(info.model_dump_json(), encoding="utf-8")
    run = SimpleNamespace(gpu_provider="cluster", num_gpus=8, max_queue_wait_hours=24)
    ticket = SimpleNamespace(
        id="train-run-001", run_id="run-1",
        payload={"base_model": "org/32B", "training_method_pin": "full_sft"},
    )
    observed = []
    job_row = None

    async def latest(*_args):
        return job_row

    async def probe(_info):
        observed.append("probe")
        return SlurmCapacitySnapshot((4, 4), 8)

    monkeypatch.setattr(runner, "_latest_slurm_resource_request", latest)
    monkeypatch.setattr(runner, "probe_slurm_capacity", probe)

    async def contract():
        return await runner._slurm_stage_job_contract(
            run=run, ticket=ticket, work_dir=str(tmp_path), filename="train.sbatch",
            device_info_path=str(path), session=None, stage="train",
        )

    submit = asyncio.run(contract())
    assert submit.phase == "submit"
    assert (submit.num_gpus, submit.nodes) == (4, 1)
    assert observed == ["probe"]

    script = tmp_path / "train.sbatch"
    script.write_text("\n".join((
        "#!/bin/bash",
        f"#SBATCH --job-name={submit.job_name}",
        f"#SBATCH --gpus={submit.num_gpus}",
        f"#SBATCH --output={submit.stdout_path}",
        f"#SBATCH --error={submit.stderr_path}",
        submit.runtime_prologue,
        submit.lifecycle_prologue,
        "python train.py",
    )), encoding="utf-8")
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text("test stub", encoding="utf-8")
    training = SimpleNamespace(
        world_size=8, distributed=SimpleNamespace(nodes=1, gpus_per_node=8),
    )
    monkeypatch.setattr(
        runner, "load_train_config", lambda _path: SimpleNamespace(training=training),
    )
    monkeypatch.setattr(runner, "validate_cluster_train_config", lambda _cfg: None)
    problem = runner._slurm_stage_result_problem(
        SimpleNamespace(slurm_job=submit), str(script), "train.py",
        configuration_path=str(config_path),
    )
    assert "world_size differs" in problem
    training.world_size = 4
    training.distributed.gpus_per_node = 4
    assert runner._slurm_stage_result_problem(
        SimpleNamespace(slurm_job=submit), str(script), "train.py",
        configuration_path=str(config_path),
    ) == ""

    job_row = SimpleNamespace(
        id="request-1", gpu_count=6, instance_id="12345", status="RUNNING",
        meta={"nodes": 2, "gpus_per_node": 3},
    )
    collect = asyncio.run(contract())
    assert collect.phase == "collect"
    assert (collect.num_gpus, collect.nodes) == (6, 2)
    assert observed == ["probe"]


def test_inference_stage_contract_uses_prepared_suite_size(tmp_path, monkeypatch) -> None:
    info = _cluster_info(plan_gpus=4, constraints=_partial_node_constraints())
    path = tmp_path / "device_info.json"
    path.write_text(info.model_dump_json(), encoding="utf-8")
    run = SimpleNamespace(
        gpu_provider="cluster", num_gpus=8, max_queue_wait_hours=24,
        holdout={"test_sets": [
            {"name": f"test-{index}", "n_rows": 2000} for index in range(8)
        ]},
    )
    ticket = SimpleNamespace(
        id="infer-run-001", run_id="run-1", lane="held_out_test",
        payload={"base_model": "org/32B"},
    )

    async def latest(*_args):
        return None

    async def probe(_info):
        return SlurmCapacitySnapshot((4, 4), 8)

    monkeypatch.setattr(runner, "_latest_slurm_resource_request", latest)
    monkeypatch.setattr(runner, "probe_slurm_capacity", probe)
    contract = asyncio.run(runner._slurm_stage_job_contract(
        run=run, ticket=ticket, work_dir=str(tmp_path), filename="predict.sbatch",
        device_info_path=str(path), session=None, stage="inference",
    ))
    assert (contract.num_gpus, contract.estimated_gpus) == (4, 1)
    assert "8 member(s), 16000 known row(s)" in contract.resource_plan_rationale
