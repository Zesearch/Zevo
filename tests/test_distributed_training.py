"""Typed distributed-training contracts, launch-command construction, and
multi-node infrastructure.

Grounds the SOTA research report Section 3 / recommendation #5 (FSDP FULL_SHARD
vs DeepSpeed ZeRO-2/3 + CPU offload; multi-node torchrun/deepspeed launch).

Backward-compatibility is the load-bearing property here: the single-GPU
default must remain the plain `python <train_script>` command and a
TrainingConfig that omits the optional `distributed` field must be unchanged.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.contracts.configuration import (
    DistributedConfig,
    DistributedLaunchPlan,
    TrainingConfig,
    build_deepspeed_config,
    build_fsdp_config,
    build_launch_command,
)
from zevo.contracts.infrastructure import (
    InfrastructureResourcePlan,
    SlurmStageJobContract,
)


# --------------------------------------------------------------------------- #
# DistributedConfig typing + validation
# --------------------------------------------------------------------------- #


def test_default_is_single_gpu_one_process() -> None:
    dist = DistributedConfig()
    assert dist.backend == "single_gpu"
    assert dist.nodes == 1
    assert dist.gpus_per_node == 1
    assert dist.world_size == 1


def test_single_gpu_rejects_multi_gpu_and_sharding() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(backend="single_gpu", gpus_per_node=2)
    with pytest.raises(ValidationError):
        DistributedConfig(backend="single_gpu", fsdp_sharding="full_shard")
    with pytest.raises(ValidationError):
        DistributedConfig(backend="single_gpu", zero_stage=3)
    with pytest.raises(ValidationError):
        DistributedConfig(backend="single_gpu", tensor_parallel_size=2)


def test_ddp_world_size_is_nodes_times_gpus_per_node() -> None:
    dist = DistributedConfig(backend="ddp", nodes=2, gpus_per_node=4)
    assert dist.world_size == 8


def test_ddp_cannot_carry_shard_offload_or_model_parallel() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(backend="ddp", gpus_per_node=2, fsdp_sharding="full_shard")
    with pytest.raises(ValidationError):
        DistributedConfig(backend="ddp", gpus_per_node=2, zero_stage=2)
    with pytest.raises(ValidationError):
        DistributedConfig(backend="ddp", gpus_per_node=2, tensor_parallel_size=2)


def test_fsdp_requires_a_sharding_strategy() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(backend="fsdp", gpus_per_node=4)
    dist = DistributedConfig(
        backend="fsdp", gpus_per_node=8, fsdp_sharding="full_shard", fsdp_offload=True
    )
    assert dist.fsdp_sharding == "full_shard"
    assert dist.fsdp_offload is True


def test_fsdp_cannot_set_deepspeed_options() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(
            backend="fsdp", gpus_per_node=4, fsdp_sharding="full_shard", zero_stage=3
        )


def test_deepspeed_requires_zero_stage_2_or_3() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(backend="deepspeed", gpus_per_node=4)  # zero_stage defaults 0
    for stage in (2, 3):
        dist = DistributedConfig(backend="deepspeed", gpus_per_node=4, zero_stage=stage)
        assert dist.zero_stage == stage


def test_deepspeed_param_offload_needs_stage_3() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(
            backend="deepspeed", gpus_per_node=4, zero_stage=2, offload_params=True
        )
    dist = DistributedConfig(
        backend="deepspeed", gpus_per_node=4, zero_stage=3, offload_params=True
    )
    assert dist.offload_params is True


def test_tensor_parallel_must_divide_gpus_per_node() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(
            backend="fsdp", gpus_per_node=6, fsdp_sharding="full_shard",
            tensor_parallel_size=4,
        )
    dist = DistributedConfig(
        backend="fsdp", gpus_per_node=8, fsdp_sharding="hybrid_shard",
        tensor_parallel_size=2,
    )
    assert dist.tensor_parallel_size == 2


def test_sequence_parallel_requires_sharded_backend() -> None:
    with pytest.raises(ValidationError):
        DistributedConfig(backend="ddp", gpus_per_node=2, sequence_parallel=True)
    dist = DistributedConfig(
        backend="fsdp", gpus_per_node=4, fsdp_sharding="full_shard",
        sequence_parallel=True,
    )
    assert dist.sequence_parallel is True


# --------------------------------------------------------------------------- #
# Launch-command construction
# --------------------------------------------------------------------------- #


def test_single_gpu_launch_is_the_plain_python_command() -> None:
    plan = build_launch_command(DistributedConfig(), train_script="train.py")
    assert isinstance(plan, DistributedLaunchPlan)
    assert plan.launcher == "python"
    assert plan.command == ["python", "train.py"]
    assert plan.nnodes == 1 and plan.nproc_per_node == 1
    assert plan.fsdp_config is None and plan.deepspeed_config is None


def test_single_node_ddp_uses_standalone_torchrun() -> None:
    dist = DistributedConfig(backend="ddp", nodes=1, gpus_per_node=4)
    plan = build_launch_command(dist, train_script="train.py")
    assert plan.launcher == "torchrun"
    assert plan.command == [
        "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4", "train.py",
    ]
    assert plan.fsdp_config is None


def test_single_node_fsdp_uses_torchrun_and_carries_fsdp_config() -> None:
    dist = DistributedConfig(
        backend="fsdp", nodes=1, gpus_per_node=8, fsdp_sharding="full_shard",
    )
    plan = build_launch_command(dist, train_script="train.py")
    assert plan.launcher == "torchrun"
    assert "--nproc_per_node=8" in plan.command
    assert plan.command[-1] == "train.py"
    assert plan.fsdp_config is not None
    assert plan.fsdp_config["fsdp_config"]["sharding_strategy"] == "FULL_SHARD"


def test_multi_node_fsdp_emits_c10d_rendezvous() -> None:
    dist = DistributedConfig(
        backend="fsdp", nodes=4, gpus_per_node=8, fsdp_sharding="hybrid_shard",
    )
    plan = build_launch_command(
        dist, train_script="train.py", rdzv_id="run-1", rdzv_endpoint="10.0.0.1:29500",
    )
    assert plan.launcher == "torchrun"
    assert plan.nnodes == 4 and plan.nproc_per_node == 8
    assert "--nnodes=4" in plan.command
    assert "--nproc_per_node=8" in plan.command
    assert "--rdzv_backend=c10d" in plan.command
    assert "--rdzv_id=run-1" in plan.command
    assert "--rdzv_endpoint=10.0.0.1:29500" in plan.command
    assert "--standalone" not in plan.command


def test_multi_node_without_endpoint_emits_shell_placeholder() -> None:
    dist = DistributedConfig(backend="ddp", nodes=2, gpus_per_node=8)
    plan = build_launch_command(dist, train_script="train.py")
    assert "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" in plan.command


def test_deepspeed_uses_deepspeed_launcher_and_zero_config() -> None:
    dist = DistributedConfig(
        backend="deepspeed", nodes=1, gpus_per_node=8, zero_stage=3,
        offload_optimizer=True, offload_params=True,
    )
    plan = build_launch_command(dist, train_script="train.py")
    assert plan.launcher == "deepspeed"
    assert "--num_nodes=1" in plan.command
    assert "--num_gpus=8" in plan.command
    assert "--deepspeed" in plan.command
    assert plan.deepspeed_config is not None
    zero = plan.deepspeed_config["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["offload_optimizer"]["device"] == "cpu"
    assert zero["offload_param"]["device"] == "cpu"


def test_multi_node_deepspeed_carries_master_and_hostfile() -> None:
    dist = DistributedConfig(
        backend="deepspeed", nodes=2, gpus_per_node=8, zero_stage=2,
    )
    plan = build_launch_command(
        dist, train_script="train.py", rdzv_endpoint="10.0.0.1:29500",
    )
    assert "--num_nodes=2" in plan.command
    assert "--hostfile=hostfile.txt" in plan.command
    assert "--master_addr=10.0.0.1" in plan.command
    assert "--master_port=29500" in plan.command


def test_extra_script_args_are_appended() -> None:
    plan = build_launch_command(
        DistributedConfig(), train_script="train.py",
        extra_script_args=["--config", "train_config.yaml"],
    )
    assert plan.command == ["python", "train.py", "--config", "train_config.yaml"]


def test_fsdp_offload_reflected_in_config() -> None:
    dist = DistributedConfig(
        backend="fsdp", gpus_per_node=4, fsdp_sharding="full_shard",
        fsdp_offload=True, activation_checkpointing=True,
    )
    config = build_fsdp_config(dist)
    assert config["fsdp_config"]["cpu_offload"] is True
    assert config["fsdp_config"]["activation_checkpointing"] is True
    assert "offload" in config["fsdp"]


def test_build_config_helpers_reject_wrong_backend() -> None:
    with pytest.raises(ValueError):
        build_fsdp_config(DistributedConfig(backend="ddp", gpus_per_node=2))
    with pytest.raises(ValueError):
        build_deepspeed_config(DistributedConfig(backend="ddp", gpus_per_node=2))


def test_empty_train_script_rejected() -> None:
    with pytest.raises(ValueError):
        build_launch_command(DistributedConfig(), train_script="  ")


# --------------------------------------------------------------------------- #
# TrainingConfig integration + backward compatibility
# --------------------------------------------------------------------------- #


def _base_training_values(**overrides: object) -> dict:
    values: dict = {
        "num_epochs": 1,
        "max_seq_len": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "effective_batch_size": 1,
        "learning_rate": 2e-4,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "precision": "bf16",
        "distributed_strategy": "single_gpu",
        "gradient_checkpointing": False,
        "packing": True,
        "logging_steps": 20,
        "eval_strategy": "steps",
        "eval_steps": 20,
        "checkpoint_retention": {
            "strategy": "none",
            "save_steps": 0,
            "max_intermediate_checkpoints": 0,
            "save_only_model": True,
            "rationale": "",
        },
        "seed": 0,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": ["all-linear"],
        "implementation_config": {},
        "software_versions": {
            "torch": "t", "transformers": "t", "trl": "t", "peft": "t",
        },
    }
    values.update(overrides)
    return values


def test_training_config_defaults_distributed_to_none() -> None:
    """A config that omits `distributed` is the unchanged single-GPU path."""
    config = TrainingConfig.model_validate(_base_training_values())
    assert config.distributed is None
    # The launch plan built from the (absent) config is the plain command.
    plan = build_launch_command(DistributedConfig(), train_script="train.py")
    assert plan.command == ["python", "train.py"]


def test_training_config_accepts_matching_distributed_world_size() -> None:
    config = TrainingConfig.model_validate(
        _base_training_values(
            batch_size=1,
            gradient_accumulation_steps=4,
            world_size=8,
            effective_batch_size=32,
            distributed_strategy="fsdp",
            distributed={
                "backend": "fsdp",
                "nodes": 1,
                "gpus_per_node": 8,
                "fsdp_sharding": "full_shard",
            },
        )
    )
    assert config.distributed is not None
    assert config.distributed.world_size == config.world_size == 8


def test_training_config_rejects_world_size_mismatch() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            _base_training_values(
                world_size=1,
                effective_batch_size=1,
                distributed={
                    "backend": "ddp",
                    "nodes": 2,
                    "gpus_per_node": 4,  # world_size 8 != 1
                },
            )
        )


def test_training_config_multi_node_effective_batch_composes() -> None:
    config = TrainingConfig.model_validate(
        _base_training_values(
            batch_size=2,
            gradient_accumulation_steps=2,
            world_size=16,
            effective_batch_size=64,
            distributed_strategy="deepspeed",
            lora_r=0,
            lora_alpha=0,
            lora_dropout=0.0,
            lora_target_modules=[],
            software_versions={"torch": "t", "transformers": "t", "trl": "t"},
            distributed={
                "backend": "deepspeed",
                "nodes": 2,
                "gpus_per_node": 8,
                "zero_stage": 3,
                "offload_optimizer": True,
            },
        )
    )
    assert config.effective_batch_size == 2 * 2 * 16 == 64


# --------------------------------------------------------------------------- #
# Multi-node infrastructure contracts
# --------------------------------------------------------------------------- #


def _resource_plan(**overrides: object) -> dict:
    values: dict = {
        "purpose": "train",
        "required_working_set_gib": 40,
        "host_memory_components_gib": {"model_and_runtime": 40},
        "host_memory_formula_gib": 64,
        "site_min_ram_gib": 0,
        "num_gpus": 1,
        "min_vram_gb": 80,
        "min_ram_gb": 64,
        "min_cpus": 8,
        "time_limit_hours": 8,
        "disk_gb": 0,
        "rationale": "test plan",
    }
    values.update(overrides)
    return values


def test_resource_plan_defaults_to_single_node() -> None:
    plan = InfrastructureResourcePlan.model_validate(_resource_plan())
    assert plan.nodes == 1
    assert plan.gpus_per_node == 1


def test_resource_plan_multi_node_gpus_per_node() -> None:
    plan = InfrastructureResourcePlan.model_validate(
        _resource_plan(num_gpus=16, nodes=2)
    )
    assert plan.nodes == 2
    assert plan.gpus_per_node == 8


def test_resource_plan_rejects_indivisible_gpu_count() -> None:
    with pytest.raises(ValidationError):
        InfrastructureResourcePlan.model_validate(
            _resource_plan(num_gpus=7, nodes=2)
        )


def test_slurm_stage_defaults_to_single_node() -> None:
    contract = SlurmStageJobContract()
    assert contract.nodes == 1
    assert contract.num_gpus == 1
    assert contract.gpus_per_node == 1


def test_slurm_stage_multi_node_gpus_per_node() -> None:
    contract = SlurmStageJobContract(enabled=False, num_gpus=32, nodes=4)
    assert contract.gpus_per_node == 8


def test_slurm_stage_rejects_indivisible_gpu_count() -> None:
    with pytest.raises(ValidationError):
        SlurmStageJobContract(enabled=False, num_gpus=6, nodes=4)
