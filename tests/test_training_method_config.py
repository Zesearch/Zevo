from __future__ import annotations

import pytest
from pydantic import ValidationError

from zevo.api.routers.ui.tasks import setting_identity
from zevo.contracts.configuration import TrainRunConfig
from zevo.contracts.train import IntermediateCheckpoint
from zevo.contracts.prompting import (
    PromptContract,
    derive_loss_contract,
    recommended_loss_objective_config,
    validate_inference_config,
)
from zevo.contracts.training_methods import (
    is_huggingface_model_id,
    method_config_errors,
)


def _train_payload(**updates):
    method = str(updates.pop("training_method", "lora_sft"))
    loss = updates.pop("loss_contract", derive_loss_contract(method, "chat"))
    if hasattr(loss, "objective_config"):
        loss.objective_config = recommended_loss_objective_config(
            method, loss.objective_config,
        )
    method_config = updates.pop("method_config", {})
    if method in {"dpo", "cpo", "gkd", "grpo", "kto", "online_dpo", "orpo", "rft", "rloo"}:
        method_config = {"use_peft": True, **method_config}
    loss_body = loss.model_dump() if hasattr(loss, "model_dump") else dict(loss)
    objective = str(loss_body.get("objective") or "unknown")
    target_scope = str(loss_body.get("target_scope") or "method_defined")
    payload = {
        "iteration": 1,
        "parent_model": "Qwen/Qwen3-4B",
        "parent_kind": "baseline",
        "parent_selection_rationale": "Initial adaptation starts from Baseline.",
        "data_signature": "d" * 64,
        "training_method": method,
        "method_config": method_config,
        "loss_contract": loss,
        "training": {
            "num_epochs": 1,
            "max_seq_len": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "world_size": 1,
            "effective_batch_size": 1,
            "learning_rate": 1e-4,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "precision": "bf16",
            "distributed_strategy": "single_gpu",
            "gradient_checkpointing": False,
            "packing": False,
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
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "lora_target_modules": ["all-linear"],
            "implementation_config": {},
            "software_versions": {
                "torch": "test", "transformers": "test", "trl": "test",
                "peft": "test",
            },
        },
        "prompt": PromptContract(prompt_framing="chat"),
        "tokenizer_source": "Qwen/Qwen3-4B",
        "chat_template_source": "tokenizer.chat_template",
        "chat_template_hash": "b" * 64,
        "template_kwargs": {},
        "special_token_ids": {"eos_token_id": 2},
        "generation_backend": "vllm",
        "inference_config_path": "/tmp/inference_config.yaml",
        "inference_config_sha256": "a" * 64,
        "prompt_alignment": {
            "rendered_prompt": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<INPUT><|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "context_token_count": 16,
            "target_token_count": 2,
            "string_prefix_match": True,
            "token_prefix_match": True,
            "prefix_labels_all_ignored": True,
            "target_starts_with_synthetic_response": True,
        },
        "training_data_example": {
            "source_record": {
                "input": "<INPUT>",
                "target": "<TARGET_RESPONSE>",
            },
            "rendered_sequences": {
                "training": (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n<INPUT><|im_end|>\n"
                    "<|im_start|>assistant\n<TARGET_RESPONSE><|im_end|>"
                ),
            },
            "loss_target_placeholders": ["<TARGET_RESPONSE>"],
            "context_only_placeholders": ["<INPUT>"],
            "sequence_loss_targets": {"training": ["<TARGET_RESPONSE>"]},
            "loss_target_summary": (
                f"The {objective} objective consumes the response span under "
                f"target_scope={target_scope}."
            ),
        },
        "direction": "test one coherent direction",
        "method_diversity_status": "initial",
        "method_selection_rationale": "initial compatible method",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    "model_id",
    ["OpenAssistant/reward-model-deberta-v3-large-v2", "Qwen/Qwen3-8B"],
)
def test_auxiliary_models_accept_explicit_huggingface_ids(model_id: str) -> None:
    assert is_huggingface_model_id(model_id)


@pytest.mark.parametrize("framework_save_only_model", [True, False])
def test_checkpoint_retention_separates_framework_state_from_branch_artifacts(
    framework_save_only_model: bool,
) -> None:
    payload = _train_payload()
    payload["training"]["checkpoint_retention"] = {
        "strategy": "epoch",
        "save_steps": 0,
        "max_intermediate_checkpoints": 2,
        "save_only_model": True,
        "rationale": "Keep two branch points to test possible late degradation.",
    }
    payload["training"]["implementation_config"] = {
        "save_only_model": framework_save_only_model,
        "save_total_limit": 3,
    }
    parsed = TrainRunConfig.model_validate(payload)
    assert parsed.training.checkpoint_retention.max_intermediate_checkpoints == 2
    assert (
        parsed.training.implementation_config["save_only_model"]
        is framework_save_only_model
    )

    payload["training"]["checkpoint_retention"]["save_only_model"] = False
    with pytest.raises(ValidationError, match="save_only_model"):
        TrainRunConfig.model_validate(payload)


def test_training_example_sequence_targets_are_closed_and_present() -> None:
    payload = _train_payload()
    parsed = TrainRunConfig.model_validate(payload)
    assert parsed.training_data_example.sequence_loss_targets == {
        "training": ["<TARGET_RESPONSE>"]
    }

    payload["training_data_example"]["sequence_loss_targets"] = {
        "missing": ["<TARGET_RESPONSE>"]
    }
    with pytest.raises(ValidationError, match="unknown rendered sequences"):
        TrainRunConfig.model_validate(payload)

    payload = _train_payload()
    payload["training_data_example"]["sequence_loss_targets"] = {
        "training": ["<NOT_DECLARED>"]
    }
    with pytest.raises(ValidationError, match="undeclared targets"):
        TrainRunConfig.model_validate(payload)

    payload = _train_payload()
    payload["training_data_example"].pop("sequence_loss_targets")
    with pytest.raises(ValidationError, match="sequence_loss_targets"):
        TrainRunConfig.model_validate(payload)

    payload = _train_payload()
    payload["training_data_example"]["rendered_sequences"]["second"] = (
        "<INPUT><TARGET_RESPONSE>"
    )
    with pytest.raises(ValidationError, match="omits rendered sequences"):
        TrainRunConfig.model_validate(payload)


@pytest.mark.parametrize("save_total_limit", [None, 1, 2, 4, True])
def test_checkpoint_retention_reserves_one_terminal_rotation_slot(
    save_total_limit,
) -> None:
    payload = _train_payload()
    payload["training"]["checkpoint_retention"] = {
        "strategy": "steps",
        "save_steps": 100,
        "max_intermediate_checkpoints": 2,
        "save_only_model": True,
        "rationale": "Keep two measured branch points before the final model.",
    }
    payload["training"]["implementation_config"] = {
        "save_only_model": False,
    }
    if save_total_limit is not None:
        payload["training"]["implementation_config"][
            "save_total_limit"
        ] = save_total_limit

    with pytest.raises(ValidationError, match="save_total_limit=3"):
        TrainRunConfig.model_validate(payload)


@pytest.mark.parametrize("save_only_model", [None, 0, 1, "false"])
def test_checkpoint_retention_requires_explicit_framework_save_mode(
    save_only_model,
) -> None:
    payload = _train_payload()
    payload["training"]["checkpoint_retention"] = {
        "strategy": "steps",
        "save_steps": 100,
        "max_intermediate_checkpoints": 2,
        "save_only_model": True,
        "rationale": "Preserve bounded branch points and resumable state.",
    }
    payload["training"]["implementation_config"] = {
        "save_total_limit": 3,
    }
    if save_only_model is not None:
        payload["training"]["implementation_config"][
            "save_only_model"
        ] = save_only_model

    with pytest.raises(ValidationError, match="explicit boolean"):
        TrainRunConfig.model_validate(payload)


def test_intermediate_checkpoint_requires_a_training_position() -> None:
    with pytest.raises(ValidationError, match="requires step or epoch"):
        IntermediateCheckpoint(
            path="/remote/train-r-001/intermediate/candidate",
            retention_reason="Test a later branch.",
        )


@pytest.mark.parametrize(
    "value",
    ["/models/reward", "./models/reward", "https://huggingface.co/org/model", "model"],
)
def test_auxiliary_models_reject_every_non_hf_reference_shape(value: str) -> None:
    assert not is_huggingface_model_id(value)


def test_gkd_train_ticket_requires_teacher_model() -> None:
    with pytest.raises(ValidationError, match="teacher_model"):
        TrainRunConfig.model_validate(_train_payload(training_method="gkd"))


def test_online_dpo_train_ticket_accepts_reward_model() -> None:
    payload = TrainRunConfig.model_validate(_train_payload(
        training_method="online_dpo",
        method_config={"reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2"},
    ))
    assert payload.method_config == {
        "use_peft": True,
        "reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2",
    }


def test_auxiliary_model_cannot_be_attached_to_another_method() -> None:
    errors = method_config_errors(
        "dpo", {"reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2"},
    )
    assert errors == ["reward_model is not valid for training_method=dpo"]


@pytest.mark.parametrize("field", ["prompt_framing", "loss_contract", "learning_rate"])
def test_method_config_rejects_fields_with_another_owner(field: str) -> None:
    errors = method_config_errors("dpo", {field: "wrong namespace"})
    assert any("another canonical owner" in error for error in errors)


@pytest.mark.parametrize("field", ["loss_type", "beta", "label_smoothing"])
def test_method_config_rejects_loss_objective_fields(field: str) -> None:
    errors = method_config_errors("dpo", {field: 0.1})
    assert any("loss_contract.objective_config" in error for error in errors)


def test_objective_config_is_validated_for_the_selected_method() -> None:
    payload = TrainRunConfig.model_validate(_train_payload(
        training_method="dpo",
        loss_contract={
            "owner": "train_skill",
            "framework": "trl",
            "objective": "dpo",
            "target_scope": "method_defined",
            "objective_config": recommended_loss_objective_config(
                "dpo", {"loss_type": "sigmoid", "beta": 0.1},
            ),
        },
    ))
    assert payload.loss_contract.objective_config["loss_type"] == "sigmoid"
    assert payload.loss_contract.objective_config["beta"] == 0.1


def test_objective_config_rejects_another_methods_fields() -> None:
    with pytest.raises(ValidationError, match="objective_config"):
        TrainRunConfig.model_validate(_train_payload(
            training_method="dpo",
            loss_contract={
                "owner": "train_skill",
                "framework": "trl",
                "objective": "dpo",
                "target_scope": "method_defined",
                "objective_config": {
                    **recommended_loss_objective_config("dpo", {}),
                    "num_generations": 4,
                },
            },
        ))


def test_method_config_rejects_unknown_implementation_keys() -> None:
    errors = method_config_errors("dpo", {"mystery_switch": True})
    assert any("unsupported method_config keys" in error for error in errors)


def test_method_config_use_peft_is_typed() -> None:
    assert method_config_errors("dpo", {"use_peft": False}) == []
    assert method_config_errors("dpo", {"use_peft": "false"}) == [
        "method_config.use_peft must be a boolean"
    ]


def test_executed_train_yaml_requires_explicit_peft_choice() -> None:
    payload = _train_payload(training_method="dpo")
    payload["method_config"].pop("use_peft")
    with pytest.raises(ValidationError, match="requires explicit method_config.use_peft"):
        TrainRunConfig.model_validate(payload)


def test_executed_train_yaml_rejects_missing_common_runtime_value() -> None:
    payload = _train_payload()
    payload["training"].pop("optimizer")
    with pytest.raises(ValidationError, match="optimizer"):
        TrainRunConfig.model_validate(payload)


def test_executed_train_yaml_rejects_incomplete_loss_objective() -> None:
    payload = _train_payload(training_method="dpo")
    payload["loss_contract"] = {
        "owner": "train_skill",
        "framework": "trl",
        "objective": "dpo",
        "target_scope": "method_defined",
        "objective_config": {"loss_type": "sigmoid", "beta": 0.1},
    }
    with pytest.raises(ValidationError, match="objective_config.*incomplete"):
        TrainRunConfig.model_validate(payload)


def test_peft_train_yaml_requires_peft_software_version() -> None:
    payload = _train_payload()
    payload["training"]["software_versions"].pop("peft")
    with pytest.raises(ValidationError, match="software_versions.*peft"):
        TrainRunConfig.model_validate(payload)


def test_method_config_is_part_of_exact_setting_identity() -> None:
    base = {"training_method": "gkd", "method_config": {"teacher_model": "Qwen/Qwen3-8B"}}
    same_from_query = {
        "training_method": "gkd",
        "method_config": '{"teacher_model":"Qwen/Qwen3-8B"}',
    }
    other = {"training_method": "gkd", "method_config": {"teacher_model": "Qwen/Qwen3-14B"}}
    assert setting_identity(base) == setting_identity(same_from_query)
    assert setting_identity(base) != setting_identity(other)


def test_experiment_preferences_are_part_of_exact_setting_identity() -> None:
    base = {
        "prompt_framing": "chat:Qwen/Qwen3-0.6B",
        "loss_objective_config": {"beta": 0.1},
        "decoding_config": {"temperature": 0.0, "seed": 0},
    }
    same_from_query = {
        **base,
        "loss_objective_config": '{"beta":0.1}',
        "decoding_config": '{"seed":0,"temperature":0.0}',
    }
    changed = {**base, "decoding_config": {"temperature": 0.7, "seed": 0}}
    assert setting_identity(base) == setting_identity(same_from_query)
    assert setting_identity(base) != setting_identity(changed)


# ─────────── opt-in multiple-choice option-scoring inference mode ────────────


def test_inference_config_default_generation_needs_no_new_keys() -> None:
    # Existing generation configs must validate unchanged (no scoring_mode).
    out = validate_inference_config(
        {"input_fields": ["question"], "answer_column": "answer",
         "answer_regex": r"([A-D])"}
    )
    assert "scoring_mode" not in out
    assert "option_fields" not in out


def test_inference_config_accepts_explicit_generate_mode() -> None:
    out = validate_inference_config(
        {"input_fields": ["question"], "answer_column": "answer",
         "scoring_mode": "generate"}
    )
    assert out["scoring_mode"] == "generate"


def test_inference_config_option_loglikelihood_requires_option_fields() -> None:
    with pytest.raises(ValueError, match="requires a non-empty option_fields"):
        validate_inference_config(
            {"input_fields": ["question"], "answer_column": "answer",
             "scoring_mode": "option_loglikelihood"}
        )


def test_inference_config_option_loglikelihood_roundtrips() -> None:
    out = validate_inference_config(
        {"input_fields": ["question"], "answer_column": "answer",
         "scoring_mode": "option_loglikelihood",
         "option_fields": ["opt_a", "opt_b", "opt_c", "opt_d"]}
    )
    assert out["scoring_mode"] == "option_loglikelihood"
    assert out["option_fields"] == ["opt_a", "opt_b", "opt_c", "opt_d"]


def test_inference_config_rejects_unknown_scoring_mode() -> None:
    with pytest.raises(ValueError, match="scoring_mode must be"):
        validate_inference_config({"scoring_mode": "logits_magic"})


def test_inference_config_option_fields_needs_scoring_mode() -> None:
    with pytest.raises(ValueError, match="only valid with"):
        validate_inference_config({"option_fields": ["opt_a", "opt_b"]})


def test_inference_config_option_fields_reject_duplicates() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        validate_inference_config(
            {"scoring_mode": "option_loglikelihood",
             "option_fields": ["opt_a", "opt_a"]}
        )
