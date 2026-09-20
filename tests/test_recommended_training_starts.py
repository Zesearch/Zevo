from __future__ import annotations

import pytest

from zevo.contracts.configuration import (
    RECOMMENDED_LORA_ADAPTER_START,
    RECOMMENDED_TRAINING_STARTS,
    TrainingConfig,
    recommended_training_starts,
    train_method_contracts,
)
from zevo.contracts.training_methods import (
    METHOD_CONFIG_KEYS,
    SELECTABLE_TRAINING_METHODS,
)


def test_every_method_contract_exposes_recommended_training_starts() -> None:
    contracts = train_method_contracts()
    assert set(contracts) == set(SELECTABLE_TRAINING_METHODS)
    for method, contract in contracts.items():
        starts = contract["recommended_training_starts"]
        assert starts["values"] == recommended_training_starts(method)
        assert isinstance(starts["guidance"], str) and starts["guidance"]


def test_explicit_lora_start_encodes_best_practice_adapter() -> None:
    starts = RECOMMENDED_LORA_ADAPTER_START
    assert starts["lora_r"] == 32
    assert starts["lora_alpha"] == 64
    assert starts["lora_alpha"] == 2 * starts["lora_r"]
    assert starts["lora_dropout"] == pytest.approx(0.05)
    assert starts["lora_target_modules"] == ["all-linear"]


def test_sft_defaults_to_full_parameters() -> None:
    starts = RECOMMENDED_TRAINING_STARTS["sft"]
    assert starts["lora_r"] == 0
    assert starts["lora_alpha"] == 0
    assert starts["lora_dropout"] == pytest.approx(0.0)
    assert starts["lora_target_modules"] == []
    assert starts["learning_rate"] == pytest.approx(1e-5)


def test_grpo_start_uses_a_much_lower_rate_and_no_packing() -> None:
    grpo = RECOMMENDED_TRAINING_STARTS["grpo"]
    assert grpo["learning_rate"] == pytest.approx(1e-6)
    assert grpo["packing"] is False


def test_rft_defaults_to_full_parameter_start() -> None:
    rft = RECOMMENDED_TRAINING_STARTS["rft"]
    assert rft["learning_rate"] == pytest.approx(1e-5)
    assert rft["lora_r"] == 0
    assert rft["lora_alpha"] == 0
    assert rft["lora_dropout"] == pytest.approx(0.0)
    assert rft["lora_target_modules"] == []


def test_peft_capable_methods_expose_full_parameter_default() -> None:
    contracts = train_method_contracts()
    for method in SELECTABLE_TRAINING_METHODS:
        keys = METHOD_CONFIG_KEYS[method]
        defaults = contracts[method]["method_config"]["default_values"]
        if "use_peft" in keys:
            assert defaults == {"use_peft": False}
        else:
            assert defaults == {}


def test_lora_adapter_start_surfaces_only_for_peft_capable_methods() -> None:
    contracts = train_method_contracts()
    for method, contract in contracts.items():
        adapter = contract["recommended_training_starts"][
            "lora_adapter_start_when_peft_active"
        ]
        peft_capable = "use_peft" in METHOD_CONFIG_KEYS[method]
        if peft_capable:
            assert adapter == RECOMMENDED_LORA_ADAPTER_START
        else:
            assert adapter is None


def test_recommended_training_starts_returns_isolated_copies() -> None:
    first = recommended_training_starts("sft")
    first["learning_rate"] = 999.0
    assert RECOMMENDED_TRAINING_STARTS["sft"]["learning_rate"] == pytest.approx(1e-5)
    assert recommended_training_starts("SFT")["learning_rate"] == pytest.approx(1e-5)


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo", "unknown_method", ""])
def test_methods_without_a_grounded_start_return_empty(method: str) -> None:
    assert recommended_training_starts(method) == {}


def test_sft_starts_realize_a_valid_training_config() -> None:
    """Guidance must be structurally realizable, not just documentation."""
    starts = RECOMMENDED_TRAINING_STARTS["sft"]
    config = TrainingConfig.model_validate({
        "data_selection": {
            "mode": "all", "source_rows": 100,
            "selected_source_rows": 100, "rationale": "",
        },
        "num_epochs": starts["num_epochs"],
        "max_seq_len": 2048,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "world_size": 1,
        "effective_batch_size": starts["effective_batch_size"],
        "learning_rate": starts["learning_rate"],
        "optimizer": "adamw_torch",
        "lr_scheduler_type": starts["lr_scheduler_type"],
        "warmup_ratio": starts["warmup_ratio"],
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "precision": "bf16",
        "distributed_strategy": "single_gpu",
        "gradient_checkpointing": False,
        "packing": starts["packing"],
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
        "lora_r": starts["lora_r"],
        "lora_alpha": starts["lora_alpha"],
        "lora_dropout": starts["lora_dropout"],
        "lora_target_modules": starts["lora_target_modules"],
        "implementation_config": {},
        "software_versions": {
            "torch": "test", "transformers": "test", "trl": "test",
        },
    })
    assert config.effective_batch_size == 32
    assert config.lora_r == 0
    assert config.lora_target_modules == []
