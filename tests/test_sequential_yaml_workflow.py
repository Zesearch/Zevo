from __future__ import annotations

import json
import io
import runpy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from zevo.contracts.configuration import (
    InferencePromptExample,
    InferenceRunConfig,
    PromptMessageExample,
    TrainRunConfig,
    TrainingConfig,
    _main as configuration_main,
    file_sha256,
    inference_mapping_contract,
    load_inference_config,
    load_train_config,
    train_method_contracts,
    validate_cluster_train_config,
)
from zevo.contracts.evaluation import EvaluationResult
from zevo.contracts.inference import (
    GenerationDiagnostics,
    InferenceResult,
    InferenceTaskInput,
    summarize_generation_diagnostics,
)
from zevo.contracts.infrastructure import (
    HostMemoryPlanningContract,
    InfraResult,
    InfrastructureDeviceInfo,
    InfrastructureResourcePlan,
    planned_host_ram_gib,
    validate_device_info,
)
from zevo.contracts.tickets import (
    CreateTicketResponse,
    _main as tickets_contract_main,
)
from zevo.contracts.model_registry import (
    RegisterResult,
    RegistryEntry,
    validate_registry_entry,
)
from zevo.contracts.data import (
    DataRecipe,
    DataResult,
    InferenceDataProfile,
    _main as data_contract_main,
    data_recipe_signature,
)
from zevo.contracts.prompting import (
    InferenceContract,
    PromptContract,
    derive_loss_contract,
)
from zevo.contracts.tickets import (
    PipelineEvaluationRequestPayload,
    PipelineTrainRequestPayload,
    PipelineRegistryRequestPayload,
    SearchBranchTransition,
    specialist_input_binding_contracts,
    specialist_request_payload_schemas,
    validate_bindings,
    validate_stored_payload,
)
from zevo.db import Ticket
from zevo.contracts.train import TrainExecutionContract, TrainResult, TrainTaskInput
from zevo.contracts.orchestrator import SupervisorAction
from zevo.engine.agent.drivers.stub import _stub_infer, _stub_train
from zevo.engine.run.runner import (
    _build_registry_input,
    _extract_summary_artifact_meta,
    _python_memory_helper_problem,
    _set_current_ticket_outcome_text,
    _tabular_shape,
    _validate_specialist_yaml,
)


def _write_inference_config(tmp_path: Path) -> Path:
    config = InferenceRunConfig(
        base_model="Qwen/Qwen3-0.6B-Base",
        prompt=PromptContract(
            prompt_framing="chat",
            model_reasoning_type="non_thinking",
            system_prompt="",
        ),
        measurement=InferenceContract(
            inference_config={
                "input_fields": ["instruction"],
                "answer_column": "prediction",
            },
        ),
        generation_backend="vllm",
        implementation_config={"trust_remote_code": False},
        tokenizer_source="Qwen/Qwen3-0.6B-Base",
        chat_template_source="tokenizer.chat_template",
        chat_template_hash="a" * 64,
        template_kwargs={},
        special_token_ids={"eos_token_id": 2},
        stop_token_ids=[2],
        prompt_example=InferencePromptExample(
            input_values={"instruction": "<INPUT:instruction>"},
            messages=[
                PromptMessageExample(
                    role="system", content="You are a helpful assistant."
                ),
                PromptMessageExample(role="user", content="<INPUT:instruction>"),
            ],
            rendered_prompt=(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<INPUT:instruction><|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ),
    )
    path = tmp_path / "inference_config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return path


def test_configuration_hashes_are_real_sha256_values(tmp_path: Path) -> None:
    path = _write_inference_config(tmp_path)
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["chat_template_hash"] = "z" * 64
    with pytest.raises(ValidationError):
        InferenceRunConfig.model_validate(body)

    with pytest.raises(ValidationError):
        DataRecipe(
            dataset_name="train.jsonl",
            source_identity="/source/train.jsonl",
            source_fingerprint="sha256:" + "a" * 64,
            training_method="full_sft",
            method_format="messages",
            method_ids=["reformat_jsonl"],
            audit_steps=["mapped source records into messages"],
        )


def test_prompt_framing_format_is_visible_in_the_json_schema() -> None:
    schema = InferenceRunConfig.model_json_schema()
    prompt_ref = schema["properties"]["prompt"]["$ref"].split("/")[-1]
    framing = schema["$defs"][prompt_ref]["properties"]["prompt_framing"]
    assert framing["pattern"] == r"^(?:chat|chat:[^\s]+|completion|text)$"

    hash_field = schema["properties"]["chat_template_hash"]
    assert hash_field["pattern"] == r"^(?:|[0-9a-f]{64})$"
    assert "template_kwargs" in schema["required"]


def test_template_kwargs_are_explicit_and_exclude_execution_controls(
    tmp_path: Path,
) -> None:
    path = _write_inference_config(tmp_path)
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body.pop("template_kwargs")
    with pytest.raises(ValidationError, match="template_kwargs"):
        InferenceRunConfig.model_validate(body)

    body["template_kwargs"] = {"add_generation_prompt": True}
    with pytest.raises(ValidationError, match="execution-control"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["template_kwargs"] = {"enable_thinking": True}
    with pytest.raises(ValidationError, match="contradicts"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["template_kwargs"] = {"enable_thinking": False}
    with pytest.raises(ValidationError, match="non-thinking models omit"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["prompt_example"]["rendered_prompt"] += "<think>\n\n</think>\n"
    with pytest.raises(ValidationError, match="reasoning content"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["prompt_example"]["rendered_prompt"] += "<think>reason</think>\n"
    with pytest.raises(ValidationError, match="must not contain think tags"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["prompt"]["model_reasoning_type"] = "thinking"
    body["template_kwargs"] = {"enable_thinking": True}
    with pytest.raises(ValidationError, match="thinking-template"):
        InferenceRunConfig.model_validate(body)
    body["prompt_example"]["rendered_prompt"] += "<think>\n"
    assert InferenceRunConfig.model_validate(body).prompt.model_reasoning_type == "thinking"

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["template_kwargs"] = {"enable_thinking": False}
    body["implementation_config"]["enable_thinking"] = False
    with pytest.raises(ValidationError, match="must not be duplicated"):
        InferenceRunConfig.model_validate(body)


def test_inference_stop_token_ids_are_explicit_verified_and_single_authority(
    tmp_path: Path,
) -> None:
    path = _write_inference_config(tmp_path)
    body = yaml.safe_load(path.read_text(encoding="utf-8"))

    missing = dict(body)
    missing.pop("stop_token_ids")
    with pytest.raises(ValidationError, match="stop_token_ids"):
        InferenceRunConfig.model_validate(missing)

    body["stop_token_ids"] = [3]
    with pytest.raises(ValidationError, match="unknown ids"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["stop_token_ids"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        InferenceRunConfig.model_validate(body)

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["implementation_config"]["sampling_params"] = {"stop_token_ids": [2]}
    with pytest.raises(ValidationError, match="one top-level authority"):
        InferenceRunConfig.model_validate(body)


def test_generation_diagnostics_require_one_dense_record_per_request() -> None:
    valid = {
        "schema_version": 1,
        "records": [{
            "request_index": 0,
            "row_index": 0,
            "turn": 1,
            "finish_reason": "stop",
            "stop_reason": 2,
            "generated_tokens": 5,
        }],
    }
    diagnostics = GenerationDiagnostics.model_validate(valid)
    assert diagnostics.records[0].stop_reason == 2
    summary = summarize_generation_diagnostics(diagnostics)
    assert summary.total_requests == 1
    assert summary.stopped_requests == 1
    assert summary.length_limited_requests == 0
    assert summary.stop_reason_counts == {"2": 1}

    valid["records"][0]["request_index"] = 1
    with pytest.raises(ValidationError, match="dense and ordered"):
        GenerationDiagnostics.model_validate(valid)


def test_data_recipe_validator_checks_the_bound_source_bytes(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"source-v1")
    training = tmp_path / "dataset.jsonl"
    training.write_text('{"messages":[]}\n', encoding="utf-8")
    recipe = DataRecipe(
        dataset_name="source.json",
        source_identity=str(source),
        source_fingerprint=file_sha256(source),
        training_method="full_sft",
        method_format="messages",
        method_ids=["reformat_jsonl"],
        audit_steps=["mapped source records into messages"],
    )
    recipe_path = tmp_path / "data_recipe.json"
    recipe_path.write_text(recipe.model_dump_json(), encoding="utf-8")

    assert data_contract_main([
        "validate-recipe", str(recipe_path), str(training),
        "--source-identity", str(source),
        "--source-file", str(source),
    ]) == 0
    source.write_bytes(b"source-v2")
    assert data_contract_main([
        "validate-recipe", str(recipe_path), str(training),
        "--source-identity", str(source),
        "--source-file", str(source),
    ]) == 1
    assert "expected" in capsys.readouterr().err


def test_data_recipe_identity_uses_method_ids_but_not_audit_prose(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"messages":[]}\n', encoding="utf-8")
    recipe = DataRecipe(
        dataset_name="train.jsonl",
        source_identity="data/files/train.jsonl",
        source_fingerprint="a" * 64,
        training_method="full_sft",
        method_format="messages",
        method_ids=["reformat_jsonl"],
        audit_steps=["mapped instruction and response fields"],
    )
    baseline = data_recipe_signature(recipe, str(dataset))
    reworded = recipe.model_copy(update={"audit_steps": ["mapped the two text fields"]})
    renamed = recipe.model_copy(update={"dataset_name": "friendly name"})
    changed_method = recipe.model_copy(update={"method_ids": ["inline_transform"]})

    assert data_recipe_signature(reworded, str(dataset)) == baseline
    assert data_recipe_signature(renamed, str(dataset)) == baseline
    assert data_recipe_signature(changed_method, str(dataset)) != baseline


def test_yaml_examples_expose_rendering_and_loss_boundaries(tmp_path: Path) -> None:
    inference_path = _write_inference_config(tmp_path)
    train = _stub_train({
        "operation": "train",
        "ticket_id": "train-example-001",
        "iteration": 1,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "base_model",
        "parent_selection_rationale": "Initial adaptation starts from Baseline.",
        "inference_config_path": str(inference_path),
        "training_method_pin": "full_sft",
        "data_signature": "d" * 64,
    }, tmp_path / "train-example")

    inference_body = yaml.safe_load(inference_path.read_text(encoding="utf-8"))
    train_body = yaml.safe_load(Path(train.train_config_path).read_text(encoding="utf-8"))
    assert inference_body["schema_version"] == 5
    assert inference_body["prompt_example"]["input_values"] == {
        "instruction": "<INPUT:instruction>"
    }
    assert "You are a helpful assistant." in inference_body["prompt_example"]["rendered_prompt"]
    assert "<INPUT:instruction>" in inference_body["prompt_example"]["rendered_prompt"]
    assert train_body["schema_version"] == 7
    assert train_body["template_kwargs"] == {}
    assert train_body["prompt_alignment"]["rendered_prompt"] == inference_body[
        "prompt_example"
    ]["rendered_prompt"]
    assert train_body["prompt_alignment"]["prefix_labels_all_ignored"] is True
    assert train_body["training_data_example"]["loss_target_placeholders"] == [
        "<TARGET_RESPONSE>"
    ]
    assert train_body["training_data_example"]["sequence_loss_targets"] == {
        "training": ["<TARGET_RESPONSE>"]
    }
    assert "<TARGET_RESPONSE>" in train_body["training_data_example"]["rendered_sequences"]["training"]


def test_ticket_payloads_expose_only_execution_operations() -> None:
    data = validate_stored_payload(
        agent_id="data",
        input_format="typed",
        payload={
            "operation": "prepare_run_data",
            "dataset": "/tmp/train.jsonl",
            "training_method": "full_sft",
        },
    )
    assert data["operation"] == "prepare_run_data"
    with pytest.raises(ValidationError, match="unsupported training_method"):
        validate_stored_payload(
            agent_id="data",
            input_format="typed",
            payload={
                "operation": "prepare_run_data",
                "dataset": "/tmp/train.jsonl",
                "training_method": "imaginary_method",
            },
        )
    with pytest.raises(ValidationError, match="hyperparameters are forbidden"):
        validate_stored_payload(
            agent_id="data",
            input_format="typed",
            payload={
                "operation": "prepare_run_data",
                "dataset": "/tmp/train.jsonl",
                "training_method": "full_sft",
                "configuration_suggestions": {"learning_rate": 1e-5},
            },
        )
    with pytest.raises(ValidationError, match="Validation-blind"):
        validate_stored_payload(
            agent_id="data",
            input_format="typed",
            payload={
                "operation": "prepare_run_data",
                "dataset": "/tmp/train.jsonl",
                "training_method": "full_sft",
                "scoring_set": "/tmp/validation.jsonl",
                "answer_fields": ["answer"],
            },
        )
    with pytest.raises(ValidationError):
        validate_stored_payload(
            agent_id="inference",
            input_format="typed",
            payload={"operation": "declare_intent"},
        )


def test_orchestrator_receives_closed_payload_and_binding_authorities() -> None:
    schemas = specialist_request_payload_schemas()
    methods = schemas["data"]["properties"]["training_method"]["enum"]
    assert "full_sft" in methods
    train_advice = schemas["train"]["properties"]["configuration_suggestions"]
    assert train_advice["additionalProperties"] is False
    assert set(train_advice["properties"]) == {"direction", "training_method"}

    bindings = specialist_input_binding_contracts()
    assert bindings["train"]["variants"]["checkpoint"]["required"] == {
        "training_dataset": "training_dataset",
        "validation_dataset": "validation_dataset",
        "inference_config": "inference_config",
        "device_info": "device_info",
        "parent_checkpoint": "checkpoint",
        "parent_train_config": "train_config",
    }
    assert bindings["registry"]["variants"]["true"]["required"]["device_info"] \
        == "device_info"
    train_lineage = bindings["train"]["variants"]["checkpoint"]["lineage"]
    assert train_lineage["inference_config"] == {
        "same_run": True,
        "source_status": ["succeeded", "degraded"],
        "source_agent": "inference",
        "source_iteration": "not_later_than_child",
        "source_payload": {"model_source": "base_model"},
    }
    assert train_lineage["parent_train_config"]["same_source_ticket_as"] \
        == "parent_checkpoint"
    registry_lineage = bindings["registry"]["variants"]["true"]["lineage"]
    assert registry_lineage["metrics"]["must_evaluate_input"] == "checkpoint"


def test_search_branch_transition_requires_complete_exhaustion_evidence() -> None:
    with pytest.raises(ValidationError, match="validation_evidence"):
        SearchBranchTransition(
            level="method",
            exhausted_branch="lora_sft / data A",
            next_branch="full_sft / data A",
        )
    transition = SearchBranchTransition(
        level="base_model",
        exhausted_branch="model A with all compatible method/data branches",
        validation_evidence="Validation plateaued across the completed branches.",
        next_branch="model B",
    )
    assert transition.level == "base_model"


def test_stored_execution_payloads_are_complete_and_closed() -> None:
    inference = {
        "operation": "run_inference",
        "model_source": "base_model",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "scoring_set": "/tmp/validation.jsonl",
        "sample_submission": "/tmp/sample_submission.csv",
        "configuration_suggestions": {},
        "configuration_pins": {},
    }
    validate_stored_payload(
        agent_id="inference", input_format="typed", payload=inference,
    )
    with pytest.raises(ValidationError, match="scoring_set"):
        validate_stored_payload(
            agent_id="inference", input_format="typed",
            payload={key: value for key, value in inference.items() if key != "scoring_set"},
        )
    with pytest.raises(ValidationError, match="unsupported Inference"):
        validate_stored_payload(
            agent_id="inference", input_format="typed",
            payload={
                **inference,
                "configuration_pins": {"temprature": 0.0},
            },
        )
    with pytest.raises(ValidationError, match="require training_method_pin"):
        validate_stored_payload(
            agent_id="train", input_format="typed",
            payload={
                "operation": "train", "base_model": "base/model",
                "model_source": "base_model",
                "parent_selection_rationale": "Initial adaptation starts from Baseline.",
                "method_config_pins": {"use_peft": True},
            },
        )


def test_registry_location_and_result_status_have_one_authority() -> None:
    base_payload = {
        "base_model": "base/model", "training_method": "full_sft",
        "dataset_source": "/app/data/files/train.jsonl", "task_objective": "Improve",
        "metric": "accuracy", "metric_direction": "max",
    }
    inputs = {
        name: {"artifact_role": role, "path": f"/tmp/{name}"}
        for name, role in (
            ("checkpoint", "checkpoint"), ("metrics", "metrics"),
            ("train_config", "train_config"),
        )
    }
    local = validate_stored_payload(
        agent_id="registry", input_format="typed", payload=base_payload,
    )
    validate_bindings(agent_id="registry", payload=local, inputs=inputs)
    remote = validate_stored_payload(
        agent_id="registry", input_format="typed",
        payload={**base_payload, "checkpoint_is_remote": True},
    )
    with pytest.raises(ValueError, match="device_info"):
        validate_bindings(agent_id="registry", payload=remote, inputs=inputs)
    PipelineRegistryRequestPayload.model_validate({})
    with pytest.raises(ValidationError):
        PipelineRegistryRequestPayload.model_validate({"base_model": "caller-owned"})
    with pytest.raises(ValidationError, match="non-empty error_message"):
        DataResult(
            status="failed", ticket_id="data-r-001", operation="prepare_run_data",
            error_message="", notes="",
        )

    ticket = Ticket(
        id="registry-12345678-001",
        run_id="12345678-full",
        agent_id="registry",
        iteration=1,
        payload={},
    )
    built = _build_registry_input(
        ticket,
        base_payload,
        inputs,
        "/tmp/run/registry-12345678-001",
    )
    assert built.registry_path == "/tmp/run/registry-12345678-001/registry.yaml"
    assert built.dataset_source == "data/files/train.jsonl"


def test_inference_data_profile_uses_one_closed_current_schema() -> None:
    profile = InferenceDataProfile.model_validate({
        "schema_version": 1,
        "source": "validation_questions_only",
        "n_rows": 1,
        "task_shape": "single-turn instruction following",
        "record_fields": {"id": "stable row id", "instruction": "user input"},
        "input_fields": ["instruction"],
        "answer_fields_removed": ["answer"],
        "submission_format": "csv",
        "submission_columns": ["id", "prediction"],
        "prediction_encoding": "plain text",
        "row_order_preserved": True,
        "stable_ids_present": True,
        "contains_answer_values": False,
        "contains_evaluation_logic": False,
    })
    assert profile.answer_fields_removed == ["answer"]


def test_orchestrator_result_must_echo_ticket_id() -> None:
    with pytest.raises(ValidationError, match="ticket_id"):
        SupervisorAction(action="wait")
    action = SupervisorAction(ticket_id="orchestrate-r-001", action="wait")
    assert action.ticket_id == "orchestrate-r-001"


def test_successful_rerun_clears_stale_ticket_error() -> None:
    ticket = Ticket(
        id="registry-12345678-001",
        run_id="12345678-run",
        agent_id="registry",
        status="succeeded",
        payload={},
        error_message="previous registry validation failure",
    )
    _set_current_ticket_outcome_text(
        ticket, status="succeeded", summary="Registry committed successfully.",
    )
    assert ticket.summary == "Registry committed successfully."
    assert ticket.error_message == ""


def test_create_ticket_response_uses_id_as_its_only_identity_key(
    monkeypatch, capsys,
) -> None:
    body = {
        "id": "infra-12345678-001",
        "run_id": "12345678-full-run-id",
        "agent_id": "infrastructure",
        "status": "queued",
        "input_format": "typed",
        "lane": "optimization",
        "iteration": 0,
        "payload": {"operation": "provision", "purpose": "inference"},
        "customization": {},
        "inputs": {},
        "summary": "",
        "error_message": "",
        "created_at": "2026-08-20T12:00:00Z",
        "updated_at": "2026-08-20T12:00:00Z",
    }
    schema = CreateTicketResponse.model_json_schema()
    assert "id" in schema["required"]
    assert "ticket_id" not in schema["properties"]

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(body)))
    assert tickets_contract_main(["created-ticket-id"]) == 0
    assert capsys.readouterr().out.strip() == "infra-12345678-001"


def test_create_ticket_response_parser_rejects_guessed_ticket_id(
    monkeypatch, capsys,
) -> None:
    guessed = {
        "ticket_id": "infra-12345678-001",
        "run_id": "12345678-full-run-id",
        "agent_id": "infrastructure",
        "status": "queued",
        "input_format": "typed",
        "lane": "optimization",
        "iteration": 0,
        "payload": {"operation": "provision", "purpose": "inference"},
        "customization": {},
        "inputs": {},
        "summary": "",
        "error_message": "",
        "created_at": "2026-08-20T12:00:00Z",
        "updated_at": "2026-08-20T12:00:00Z",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(guessed)))
    assert tickets_contract_main(["created-ticket-id"]) == 1
    error = capsys.readouterr().err
    assert "INVALID CreateTicketResponse" in error
    assert "id" in error


def test_jsonl_shape_uses_physical_newlines_only(tmp_path: Path) -> None:
    path = tmp_path / "unicode.jsonl"
    path.write_text(
        '{"id":1,"instruction":"first\u2028still first"}\n'
        '{"id":2,"instruction":"second\u2029still second"}\n',
        encoding="utf-8",
    )
    assert _tabular_shape(str(path)) == (2, {"id", "instruction"})


def test_side_effect_free_config_validator_uses_current_schema(
    tmp_path: Path, capsys,
) -> None:
    valid = _write_inference_config(tmp_path)
    assert configuration_main(["validate", "inference", str(valid)]) == 0
    assert "VALID InferenceRunConfig" in capsys.readouterr().out

    body = yaml.safe_load(valid.read_text(encoding="utf-8"))
    body["operation"] = "run_inference"
    body["measurement"]["top_k"] = -1
    invalid = tmp_path / "invalid_inference.yaml"
    invalid.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert configuration_main(["validate", "inference", str(invalid)]) == 1
    error = capsys.readouterr().err
    assert "INVALID InferenceRunConfig" in error
    assert "top_k" in error and "operation" in error


def test_cluster_train_validator_requires_zero_dataloader_workers(
    tmp_path: Path, capsys,
) -> None:
    inference_path = _write_inference_config(tmp_path)
    result = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-001",
        "iteration": 1,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "base_model",
        "parent_selection_rationale": "Start from Baseline.",
        "inference_config_path": str(inference_path),
        "expected_inference_config_sha256": file_sha256(inference_path),
        "data_signature": "d" * 64,
    }, tmp_path / "train")
    config_path = Path(result.train_config_path)
    body = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert configuration_main([
        "validate", "train", str(config_path), "--cluster",
    ]) == 1
    assert "dataloader_num_workers=0" in capsys.readouterr().err

    body["training"]["implementation_config"]["dataloader_num_workers"] = 2
    config_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="dataloader_num_workers=0"):
        validate_cluster_train_config(load_train_config(config_path))

    body["training"]["implementation_config"]["dataloader_num_workers"] = 0
    config_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert configuration_main([
        "validate", "train", str(config_path), "--cluster",
    ]) == 0
    assert "VALID TrainRunConfig" in capsys.readouterr().out


def test_adaptive_vllm_memory_plan_replaces_fixed_gpu_fraction(
    tmp_path: Path, capsys,
) -> None:
    config_path = _write_inference_config(tmp_path)
    body = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    memory_plan = {
        "strategy": "model_and_workload_sized",
        "runtime_weight_gib": 3.0,
        "peak_live_tokens": 131072,
        "kv_bytes_per_token": 65536,
        "kv_cache_gib": 8.0,
        "tensor_parallel_size": 1,
        "target_gpu_memory_gib": 16,
        "utilization_step": 0.05,
        "min_utilization": 0.1,
        "max_utilization": 0.9,
        "free_memory_margin_gib": 2.0,
    }
    body["implementation_config"] = {
        "llm_kwargs": {"tensor_parallel_size": 1},
        "memory_plan": memory_plan,
    }
    config_path.write_text(yaml.safe_dump(body), encoding="utf-8")

    assert configuration_main([
        "validate", "inference", str(config_path),
        "--adaptive-vllm-memory",
    ]) == 0
    assert "VALID InferenceRunConfig" in capsys.readouterr().out

    helper = runpy.run_path(str(
        Path(__file__).parents[1]
        / "playbook" / "runners" / "inference_memory.py"
    ))
    assert helper["gpu_memory_utilization"](memory_plan, 184.0) == 0.1
    assert helper["required_free_memory_gib"](
        memory_plan, 184.0,
    ) == pytest.approx(20.4)

    body["implementation_config"]["llm_kwargs"][
        "gpu_memory_utilization"
    ] = 0.85
    config_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert configuration_main([
        "validate", "inference", str(config_path),
        "--adaptive-vllm-memory",
    ]) == 1
    assert "device-derived" in capsys.readouterr().err


def test_host_ram_plan_scales_with_stage_working_set_and_keeps_headroom() -> None:
    contract = HostMemoryPlanningContract()

    assert planned_host_ram_gib(25, "inference", contract) == 48
    assert planned_host_ram_gib(25, "train", contract) == 48
    assert planned_host_ram_gib(3, "inference", contract) == 24
    assert planned_host_ram_gib(3, "train", contract) == 32
    with pytest.raises(ValueError, match="positive finite"):
        planned_host_ram_gib(0, "inference", contract)


def test_resource_plan_recomputes_and_enforces_host_memory() -> None:
    valid = {
        "purpose": "inference",
        "required_working_set_gib": 25,
        "host_memory_components_gib": {"model": 18, "kv_cache": 7},
        "host_memory_formula_gib": 48,
        "site_min_ram_gib": 0,
        "num_gpus": 1,
        "min_vram_gb": 16,
        "min_ram_gb": 48,
        "min_cpus": 4,
        "time_limit_hours": 1,
        "disk_gb": 0,
        "rationale": "Model plus live KV cache with Zevo headroom.",
    }
    assert InfrastructureResourcePlan.model_validate(valid).min_ram_gb == 48
    with pytest.raises(ValidationError, match="planning formula"):
        InfrastructureResourcePlan.model_validate({
            **valid, "host_memory_formula_gib": 128, "min_ram_gb": 128,
        })
    with pytest.raises(ValidationError, match="must equal max"):
        InfrastructureResourcePlan.model_validate({**valid, "min_ram_gb": 128})


def test_vllm_predict_script_must_call_memory_helper(tmp_path: Path) -> None:
    script = tmp_path / "predict.py"
    script.write_text(
        "from zevo_inference_memory import gpu_memory_utilization\n"
        "# imported but never used\n",
        encoding="utf-8",
    )
    assert "must import and call" in _python_memory_helper_problem(str(script))
    script.write_text(
        "from zevo_inference_memory import gpu_memory_utilization as choose_memory\n"
        "value = choose_memory(plan, total)\n",
        encoding="utf-8",
    )
    assert _python_memory_helper_problem(str(script)) == ""


def test_cross_agent_artifacts_have_exact_machine_contracts(tmp_path: Path) -> None:
    plan = InfrastructureResourcePlan(
        purpose="inference",
        required_working_set_gib=12,
        host_memory_components_gib={"model_and_runtime": 12},
        host_memory_formula_gib=32,
        site_min_ram_gib=0,
        num_gpus=1,
        min_vram_gb=16,
        min_ram_gb=32,
        min_cpus=4,
        time_limit_hours=0,
        cloud_backend="vastai",
        disk_gb=0,
        rationale="test",
    )
    info = InfrastructureDeviceInfo(
        run_id="run-1",
        ticket_id="infra-run-001",
        purpose="inference",
        provider="cloud",
        cloud_backend="vastai",
        instance_id="cloud-1",
        auto_release=True,
        host="gpu.example",
        ssh={
            "host": "gpu.example", "port": 22, "user": "root",
            "key_path": "/root/.ssh/id_ed25519",
        },
        gpu={
            "gpu_count": 1,
            "gpu_name": "NVIDIA GPU",
            "vram_gb": 24,
            "vram_mb": 24576,
            "devices": [{"index": 0, "name": "NVIDIA GPU", "vram_mb": 24576}],
        },
        cuda={
            "driver_version": "550.54",
            "cuda_version": "12.4",
            "recommended_torch_index": "cu121",
        },
        cost={"dph_total": 0.5},
        resource_plan=plan,
        probe_source="remote_nvidia-smi",
        probed_at="2026-08-20T12:00:00Z",
    )
    device_path = tmp_path / "device_info.json"
    device_path.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    validated = validate_device_info(
        device_path,
        run_id="run-1",
        ticket_id="infra-run-001",
        provider="cloud",
        num_gpus=2,
    )
    assert validated.gpu.devices[0].vram_mb == 24576
    malformed = info.model_dump(mode="json")
    malformed["gpu"]["gpus"] = malformed["gpu"].pop("devices")
    device_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError, match="devices|gpus"):
        validate_device_info(
            device_path,
            run_id="run-1",
            ticket_id="infra-run-001",
            provider="cloud",
            num_gpus=2,
        )

    tag = "M-12345678"
    registry_path = tmp_path / "registry.yaml"
    entry = RegistryEntry(
        run_id="12345678-full",
        ticket_id="registry-run-001",
        iteration=1,
        base_model="org/model",
        training_method="full_sft",
        dataset_source="org/data@revision",
        model_path="data/runs/12345678/models/M-12345678",
        task_objective="Improve the model.",
        metric="accuracy",
        metric_direction="max",
        eval={"score": 0.7, "n": 10},
        registered_at="2026-08-20T12:00:00Z",
    )
    registry_path.write_text(
        yaml.safe_dump({"models": {tag: entry.model_dump(mode="json")}}),
        encoding="utf-8",
    )
    assert validate_registry_entry(
        registry_path, run_id="12345678-full", version_tag=tag,
    ).eval.score == 0.7
    registry_schema = RegistryEntry.model_json_schema()
    assert registry_schema["additionalProperties"] is False


def test_results_do_not_duplicate_authoritative_artifact_facts() -> None:
    forbidden = {
        DataResult: {
            "dataset_source", "resolved_dataset_source", "method_ids",
            "audit_steps",
            "data_intent_signature", "data_recipe", "data_signature",
        },
        InfraResult: {
            "resource_plan", "has_gpu", "gpu_count", "gpu_name", "vram_gb",
            "cuda_version", "host", "dph_total",
        },
        TrainResult: {
            "base_model", "training_method", "method_rationale",
            "training_objective", "training_objective_rationale",
        },
        InferenceResult: {
            "base_model", "predict_script_reused", "predict_script_source_path",
        },
        EvaluationResult: {"headline_score"},
        RegisterResult: {"version_tag", "retained", "model_path", "score_recorded"},
        SupervisorAction: {"headline_score", "registry_version_tag"},
    }

    for result_type, duplicated_fields in forbidden.items():
        assert duplicated_fields.isdisjoint(result_type.model_fields)


def test_inference_script_reuse_provenance_is_engine_derived(tmp_path: Path) -> None:
    config_path = _write_inference_config(tmp_path)
    source_script = tmp_path / "prior_predict.py"
    realized_script = tmp_path / "current_predict.py"
    source_script.write_text("print('predict')\n", encoding="utf-8")
    realized_script.write_text("print('predict')\n", encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("id,prediction\n1,A\n", encoding="utf-8")
    generation_diagnostics = tmp_path / "generation_diagnostics.json"
    generation_diagnostics.write_text(json.dumps({
        "schema_version": 1,
        "records": [{
            "request_index": 0,
            "row_index": 0,
            "turn": 1,
            "finish_reason": "stop",
            "stop_reason": 2,
            "generated_tokens": 1,
        }],
    }), encoding="utf-8")
    inp = InferenceTaskInput(
        ticket_id="infer-r-002",
        run_id="r",
        iteration=1,
        model_source="checkpoint",
        configuration_mode="reuse",
        base_model="Qwen/Qwen3-0.6B-Base",
        checkpoint_path="/tmp/model",
        scoring_set="/tmp/questions.jsonl",
        sample_submission="/tmp/submission.csv",
        inference_config_path=str(config_path),
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate inference "
            "<absolute-yaml-path>"
        ),
        memory_helper_path=str(tmp_path / "zevo_inference_memory.py"),
        predictions_validation_command=(
            "python -m zevo.engine.artifact_validation validate-predictions "
            "--predictions <absolute-predictions-csv-path>"
        ),
        reusable_predict_script_path=str(source_script),
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path),
    )
    result = InferenceResult(
        status="succeeded",
        ticket_id="infer-r-002",
        inference_config_path=str(config_path),
        predict_script_path=str(realized_script),
        predictions_path=str(predictions),
        generation_diagnostics_path=str(generation_diagnostics),
        n_rows=1,
        n_requests=1,
        error_message="",
        notes="",
    )

    _, _, _, meta = _extract_summary_artifact_meta(
        result,
        inp=inp,
        agent_id="inference",
        work_dir=str(tmp_path),
    )

    assert meta["predict_script_reuse"] == {
        "reused": True,
        "source_path": str(source_script),
    }


def test_model_chain_bindings_require_selected_checkpoint_and_yaml() -> None:
    payload = {
        "operation": "train",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "checkpoint",
        "parent_selection_rationale": "Branch from the strongest earlier checkpoint.",
    }
    base = {
        "training_dataset": {"artifact_role": "training_dataset", "path": "/d.jsonl"},
        "validation_dataset": {"artifact_role": "validation_dataset", "path": "/v.jsonl"},
        "inference_config": {"artifact_role": "inference_config", "path": "/i.yaml"},
        "device_info": {"artifact_role": "device_info", "path": "/gpu.json"},
    }
    with pytest.raises(ValueError, match="parent_checkpoint"):
        validate_bindings(agent_id="train", payload=payload, inputs=base)
    valid = validate_bindings(
        agent_id="train",
        payload=payload,
        inputs={
            **base,
            "parent_checkpoint": {"artifact_role": "checkpoint", "path": "/m"},
            "parent_train_config": {"artifact_role": "train_config", "path": "/t.yaml"},
        },
    )
    assert valid["parent_train_config"]["artifact_role"] == "train_config"


def test_baseline_selects_and_candidates_reuse_configuration(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        '{"schema_version":1,"source":"validation_questions_only","n_rows":1,'
        '"task_shape":"single turn","record_fields":{"instruction":"string"},'
        '"input_fields":["instruction"],"answer_fields_removed":["answer"],'
        '"submission_format":"csv","submission_columns":["id","prediction"],'
        '"prediction_encoding":"string","row_order_preserved":true,'
        '"stable_ids_present":false,"contains_answer_values":false,'
        '"contains_evaluation_logic":false}',
        encoding="utf-8",
    )
    baseline = _stub_infer({
        "operation": "run_inference",
        "ticket_id": "infer-r-001",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "generation_backend": "vllm",
        "inference_data_profile_path": str(profile),
        "configuration_pins": {},
    }, tmp_path / "baseline")
    assert baseline.status == "succeeded"
    original_hash = file_sha256(baseline.inference_config_path)

    later_lineage = InferenceTaskInput(
        ticket_id="infer-r-010",
        run_id="r",
        iteration=3,
        model_source="base_model",
        configuration_mode="select",
        base_model="allenai/OLMo-2-0425-1B",
        scoring_set="/tmp/questions.jsonl",
        sample_submission="/tmp/submission.csv",
        inference_data_profile_path=str(profile),
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate inference "
            "<absolute-yaml-path>"
        ),
        memory_helper_path=str(tmp_path / "zevo_inference_memory.py"),
        predictions_validation_command=(
            "python -m zevo.engine.artifact_validation validate-predictions "
            "--predictions <absolute-predictions-csv-path>"
        ),
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path / "later-lineage-baseline"),
    )
    assert later_lineage.iteration == 3

    candidate = _stub_infer({
        "operation": "run_inference",
        "ticket_id": "infer-r-002",
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "generation_backend": "vllm",
        "inference_config_path": baseline.inference_config_path,
    }, tmp_path / "candidate")
    assert candidate.inference_config_path == baseline.inference_config_path
    assert file_sha256(candidate.inference_config_path) == original_hash

    # A stray non-empty suggestion on a reuse ticket is dropped (not fatal), so
    # the run proceeds instead of wedging on every wakeup.
    reuse_with_stray_suggestion = InferenceTaskInput(
        ticket_id="infer-r-003",
        run_id="r",
        iteration=1,
        model_source="checkpoint",
        configuration_mode="reuse",
        base_model="Qwen/Qwen3-0.6B-Base",
        checkpoint_path="/tmp/model",
        scoring_set="/tmp/questions.jsonl",
        sample_submission="/tmp/submission.csv",
        inference_config_path=baseline.inference_config_path,
        inference_config_schema=InferenceRunConfig.model_json_schema(),
        inference_mapping_contract=inference_mapping_contract(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate inference "
            "<absolute-yaml-path>"
        ),
        memory_helper_path=str(tmp_path / "zevo_inference_memory.py"),
        predictions_validation_command=(
            "python -m zevo.engine.artifact_validation validate-predictions "
            "--predictions <absolute-predictions-csv-path>"
        ),
        configuration_suggestions={"temperature": 0.8},
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path / "candidate-2"),
    )
    assert reuse_with_stray_suggestion.configuration_suggestions == {}


def test_train_yaml_copies_prompt_and_records_immediate_parent(tmp_path: Path) -> None:
    inference_path = _write_inference_config(tmp_path)
    first = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-001",
        "iteration": 1,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "base_model",
        "parent_selection_rationale": "Initial adaptation starts from Baseline.",
        "inference_config_path": str(inference_path),
        "training_method_pin": "lora_sft",
        "configuration_pins": {},
        "configuration_suggestions": {},
        "data_signature": "d" * 64,
    }, tmp_path / "train1")
    first_config = load_train_config(first.train_config_path)
    inference_config = load_inference_config(inference_path)
    assert first_config.parent_kind == "baseline"
    assert first_config.prompt == inference_config.prompt
    assert first_config.tokenizer_source == inference_config.tokenizer_source
    assert first_config.chat_template_source == inference_config.chat_template_source
    assert first_config.chat_template_hash == inference_config.chat_template_hash
    assert first_config.template_kwargs == inference_config.template_kwargs
    assert first_config.special_token_ids == inference_config.special_token_ids
    assert first_config.method_diversity_status == "user_pinned"

    second = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-002",
        "iteration": 2,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "checkpoint",
        "parent_selection_rationale": "Continue from iteration 1 for refinement.",
        "parent_checkpoint_path": first.checkpoint_path,
        "parent_train_config_path": first.train_config_path,
        "inference_config_path": str(inference_path),
        "training_method_pin": "lora_sft",
        "configuration_pins": {},
        "configuration_suggestions": {"learning_rate": 5e-5},
        "data_signature": "d" * 64,
    }, tmp_path / "train2")
    second_config = load_train_config(second.train_config_path)
    assert second_config.parent_kind == "run_checkpoint"
    assert second_config.parent_model == first.checkpoint_path
    assert second_config.training.learning_rate == 5e-5
    assert second_config.training_method == "lora_sft"
    assert second_config.method_diversity_status == "user_pinned"

    unpinned_input = TrainTaskInput(
        ticket_id="train-r-002",
        run_id="r",
        iteration=2,
            dataset_path="/tmp/data.jsonl",
            validation_dataset_path="/tmp/validation.jsonl",
            validation_answer_fields=["answer"],
        data_signature="d" * 64,
        inference_config_path=str(inference_path),
        expected_inference_config_sha256=file_sha256(inference_path),
        train_config_schema=TrainRunConfig.model_json_schema(),
        training_method_contracts=train_method_contracts(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate train "
            "<absolute-yaml-path>"
        ),
        telemetry_helper_path=str(tmp_path / "zevo_train_telemetry.py"),
        execution_contract=TrainExecutionContract(
            required_environment={"RUN_ID": "r", "TICKET_ID": "train-r-002"},
            slurm_step_name="zevo-train-r-002",
        ),
        base_model="Qwen/Qwen3-0.6B-Base",
        model_source="checkpoint",
        parent_selection_rationale="Continue from iteration 1 for refinement.",
        parent_checkpoint_path=first.checkpoint_path,
        parent_train_config_path=first.train_config_path,
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path / "validate"),
        generation_backend="vllm",
    )
    warnings: list[str] = []
    _validate_specialist_yaml(
        second, unpinned_input, advisory_warnings=warnings,
    )
    assert any("active unpinned method branch" in warning for warning in warnings)

    mismatched_kwargs = tmp_path / "train2-mismatched-template-kwargs.yaml"
    body = yaml.safe_load(Path(second.train_config_path).read_text(encoding="utf-8"))
    body["template_kwargs"] = {"enable_thinking": True}
    mismatched_kwargs.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="template_kwargs"):
        _validate_specialist_yaml(
            second.model_copy(update={"train_config_path": str(mismatched_kwargs)}),
            unpinned_input,
        )

    mismatched_prefix = tmp_path / "train2-mismatched-rendered-prefix.yaml"
    body = yaml.safe_load(Path(second.train_config_path).read_text(encoding="utf-8"))
    body["prompt_alignment"]["rendered_prompt"] = "<DRIFT>"
    mismatched_prefix.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_alignment.rendered_prompt"):
        _validate_specialist_yaml(
            second.model_copy(update={"train_config_path": str(mismatched_prefix)}),
            unpinned_input,
        )


def test_unpinned_train_retains_method_until_orchestrator_switches_branch(tmp_path: Path) -> None:
    inference_path = _write_inference_config(tmp_path)
    first = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-001",
        "iteration": 1,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "base_model",
        "parent_selection_rationale": "Initial adaptation starts from Baseline.",
        "inference_config_path": str(inference_path),
        "training_method_pin": "",
        "configuration_pins": {},
        "configuration_suggestions": {},
        "data_signature": "d" * 64,
    }, tmp_path / "train1")
    first_config = load_train_config(first.train_config_path)
    assert first_config.method_diversity_status == "initial"

    second = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-002",
        "iteration": 2,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "checkpoint",
        "parent_selection_rationale": "Continue the active method branch.",
        "parent_checkpoint_path": first.checkpoint_path,
        "parent_train_config_path": first.train_config_path,
        "inference_config_path": str(inference_path),
        "training_method_pin": "",
        "configuration_pins": {},
        "configuration_suggestions": {},
        "data_signature": "d" * 64,
    }, tmp_path / "train2")
    second_config = load_train_config(second.train_config_path)
    assert first_config.training_method == "lora_sft"
    assert second_config.training_method == "lora_sft"
    assert second_config.method_diversity_status == "retained_in_branch"

    third = _stub_train({
        "operation": "train",
        "ticket_id": "train-r-003",
        "iteration": 3,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "model_source": "checkpoint",
        "parent_selection_rationale": "Start the next method branch after exhaustion.",
        "parent_checkpoint_path": second.checkpoint_path,
        "parent_train_config_path": second.train_config_path,
        "inference_config_path": str(inference_path),
        "training_method_pin": "",
        "configuration_pins": {},
        "configuration_suggestions": {"training_method": "full_sft"},
        "branch_transition": {
            "level": "method",
            "exhausted_branch": "lora_sft with supervised data branch A",
            "validation_evidence": "Validation plateaued across two inner directions.",
            "next_branch": "full_sft with compatible supervised data",
        },
        "data_signature": "d" * 64,
    }, tmp_path / "train3")
    third_config = load_train_config(third.train_config_path)
    assert third_config.training_method == "full_sft"
    assert third_config.method_diversity_status == "varied"
    assert "exhausted" in third_config.method_selection_rationale.lower()


def test_train_input_rejects_broken_iteration_chain(tmp_path: Path) -> None:
    common = dict(
        ticket_id="train-r-002",
        run_id="r",
        iteration=2,
        dataset_path="/tmp/data.jsonl",
        validation_dataset_path="/tmp/validation.jsonl",
        validation_answer_fields=["answer"],
        data_signature="d" * 64,
        inference_config_path="/tmp/inference.yaml",
        expected_inference_config_sha256="e" * 64,
        train_config_schema=TrainRunConfig.model_json_schema(),
        training_method_contracts=train_method_contracts(),
        config_validation_command=(
            "python -m zevo.contracts.configuration validate train "
            "<absolute-yaml-path>"
        ),
        telemetry_helper_path=str(tmp_path / "zevo_train_telemetry.py"),
        execution_contract=TrainExecutionContract(
            required_environment={"RUN_ID": "r", "TICKET_ID": "train-r-002"},
            slurm_step_name="zevo-train-r-002",
        ),
        base_model="Qwen/Qwen3-0.6B-Base",
        model_source="checkpoint",
        parent_selection_rationale="Use an earlier successful Run checkpoint.",
        device_info_path="/tmp/device.json",
        work_dir=str(tmp_path),
        generation_backend="vllm",
    )
    with pytest.raises(ValidationError, match="parent_checkpoint_path"):
        TrainTaskInput(**common)

    later_baseline = TrainTaskInput(**{**common, "model_source": "base_model"})
    assert later_baseline.iteration == 2
    assert later_baseline.parent_checkpoint_path == ""


def test_evaluation_remains_a_typed_system_stage() -> None:
    payload = validate_stored_payload(
        agent_id="evaluation",
        input_format="typed",
        payload={
            "metric": "token_f1", "evaluation_config": {},
            "scoring_set": "/validation.jsonl", "answer_fields": ["answer"],
            "sample_submission": "/validation_sample.csv",
        },
    )
    bindings = validate_bindings(
        agent_id="evaluation",
        payload=payload,
        inputs={
            "predictions": {
                "artifact_role": "predictions",
                "path": "/tmp/predictions.csv",
            }
        },
    )
    assert bindings["predictions"]["artifact_role"] == "predictions"


def test_nested_configuration_key_contracts_are_explicit() -> None:
    methods = train_method_contracts()
    assert methods["full_sft"]["method_config"]["allowed_keys"] == []
    assert methods["dpo"]["method_config"]["required_keys"] == ["use_peft"]
    assert "loss_type" in methods["full_sft"]["loss_objective_config"][
        "required_realized_keys"
    ]
    mapping = inference_mapping_contract()
    assert mapping["required_keys"] == ["input_fields", "answer_column"]
    assert "batch_size" in mapping["allowed_keys"]


def test_orchestrator_request_payloads_do_not_accept_api_owned_keys() -> None:
    PipelineEvaluationRequestPayload.model_validate({})
    with pytest.raises(ValidationError, match="evaluation_config"):
        PipelineEvaluationRequestPayload.model_validate({"evaluation_config": {}})
    with pytest.raises(ValidationError, match="configuration_pins"):
        PipelineTrainRequestPayload.model_validate({
            "operation": "train",
            "base_model": "base/model",
            "model_source": "base_model",
            "parent_selection_rationale": "Initial adaptation starts from Baseline.",
            "configuration_pins": {},
        })
