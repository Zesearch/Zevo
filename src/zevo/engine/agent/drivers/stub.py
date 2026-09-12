"""Stub driver — deterministic agent outputs for hermetic tests.

Does NOT spawn any subprocess, hit any API, or read any file. Returns
a canned Pydantic result based on the agent's output_schema +
input_payload. Used by:

  - tests/test_stub_workflow_smoke.py — walks one deterministic baseline/Train cycle
  - contract and artifact tests that need no LLM or GPU

When you add a new agent / schema, add a STUB_FACTORIES entry so the
test-suite can exercise it. The factory takes the input payload + a
work_dir and returns the typed Result. Factories SHOULD write a tiny
artifact file to work_dir so downstream agents have a real path to
resolve against (matches the contract of real drivers).
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable

import yaml
from pydantic import BaseModel

from zevo.engine.agent.loader import AgentBlueprint
from zevo.engine.agent.drivers.base import DriverRunResult


# Each factory takes (input_payload_dict, workspace_dir) → typed Result
StubFactory = Callable[[dict, Path], BaseModel]


def _stub_data(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.data import (
        DataRecipe,
        DataResult,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    operation = str(payload.get("operation") or "prepare_run_data")
    if operation == "scope_problem":
        return _stub_scoping(payload, work_dir)
    prepare_script = work_dir / "prepare_data.py"
    prepare_script.write_text("# deterministic stub data preparation\n", encoding="utf-8")
    if operation == "prepare_holdout_data":
        public = work_dir / "scoring_public.csv"
        public.write_text("id,instruction\n1,stub question\n", encoding="utf-8")
        return DataResult(
            status="succeeded", operation=operation,
            ticket_id=payload.get("ticket_id", "data-holdout-stub"),
            scoring_public_path=str(public),
            prepare_script_path=str(prepare_script),
            error_message="", notes="stub questions-only held-out copy",
        )

    out = work_dir / "dataset.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "Q1?"},
                      {"role": "assistant", "content": "A1"}]},
        {"messages": [{"role": "user", "content": "Q2?"},
                      {"role": "assistant", "content": "A2"}]},
        {"messages": [{"role": "user", "content": "Q3?"},
                      {"role": "assistant", "content": "A3"}]},
    ]
    training_rows = rows
    out.write_text(
        "".join(json.dumps(r) + "\n" for r in training_rows), encoding="utf-8"
    )
    source = str(payload.get("expected_source_identity") or "")
    methods = list(
        (payload.get("configuration_pins") or {}).get("method_ids")
        or (payload.get("configuration_suggestions") or {}).get("method_ids")
        or ["passthrough_jsonl"]
    )
    recipe_intent = dict(payload.get("recipe_intent") or {})
    recipe_intent.pop("schema_version", None)
    recipe = DataRecipe(
        **recipe_intent,
        dataset_name=(Path(source).name or "stub dataset"),
        source_identity=source,
        source_fingerprint=(
            hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if Path(str(source)).is_file()
            else hashlib.sha256(str(source).encode()).hexdigest()
        ),
        training_method=str(payload.get("training_method") or "full_sft"),
        method_format="messages",
        method_ids=methods,
        audit_steps=["validated and materialized the stub training records"],
    )
    recipe_path = work_dir / "data_recipe.json"
    recipe_path.write_text(recipe.model_dump_json(indent=2), encoding="utf-8")
    return DataResult(
        status="succeeded",
        operation="prepare_run_data",
        ticket_id=payload.get("ticket_id", "data-stub"),
        training_dataset_path=str(out),
        prepare_script_path=str(prepare_script),
        data_recipe_path=str(recipe_path),
        n_rows_in=3, n_rows_out=len(training_rows),
        error_message="", notes="stub-generated 3-row chat-jsonl",
    )


def _stub_scoping(payload: dict, work_dir: Path, *, rows: int = 1000) -> BaseModel:
    """Auto mode: a deterministic "public benchmark" scoping outcome.

    Writes a held-out CSV WITH answers (large enough for settlement to carve a
    20% / >= 200-row Validation set), a sample submission, and the
    ``scoping_result.json`` the engine settles from. No network, no teacher.
    """
    from zevo.contracts.data import DataResult
    from zevo.contracts.scoping import BenchmarkProvenance, ScopingResult

    work_dir.mkdir(parents=True, exist_ok=True)
    test_set = work_dir / "benchmark_test.csv"
    with test_set.open("w", encoding="utf-8", newline="") as fh:
        fh.write("id,question,choices,answer\n")
        for i in range(rows):
            fh.write(f'{i},"stub question {i}?","[""A"",""B"",""C"",""D""]",{"ABCD"[i % 4]}\n')
    sample = work_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n0,A\n", encoding="utf-8")
    result = ScopingResult(
        metric="accuracy",
        metric_direction="max",
        eval_source="public_benchmark",
        test_set_path=str(test_set),
        test_answer_fields=["answer"],
        test_sample_submission_path=str(sample),
        test_rows=rows,
        rationale=(
            "stub: a multiple-choice public benchmark matching the objective "
            f"{str(payload.get('task_objective') or '')[:80]!r}; accuracy is the "
            "standard reported metric"
        ),
        candidates_considered=["stub/other-benchmark: rejected, wrong domain"],
        benchmark=BenchmarkProvenance(
            hub_id="stub/benchmark", config="default", split="test", rows=rows,
        ),
    )
    scoping_path = work_dir / "scoping_result.json"
    scoping_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return DataResult(
        status="succeeded",
        operation="scope_problem",
        ticket_id=payload.get("ticket_id", "scope-stub"),
        scoping_result_path=str(scoping_path),
        error_message="",
        notes="stub scoping: public benchmark, accuracy/max",
    )


def _stub_infra(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.infrastructure import (
        InfraResult,
        InfrastructureResourcePlan,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    provider = payload.get("provider") or "cloud"
    if provider not in ("cloud", "cluster", "instance"):
        provider = "cloud"
    if bool(payload.get("release", False)):
        return InfraResult(
            status="succeeded",
            ticket_id=payload.get("ticket_id", "infra-stub"),
            operation="release",
            device_info_path="",
            provider=provider,
            instance_id=str(payload.get("instance_id") or "stub-0"),
            cloud_backend="vastai" if provider == "cloud" else "",
            auto_release=False,
            error_message="", notes="stub resource released",
        )
    di = work_dir / "device_info.json"
    auto_release = bool(payload.get("auto_release", provider == "cloud"))
    gpu_count = max(1, int(payload.get("num_gpus", 1) or 1))
    purpose = str(payload.get("purpose") or "inference")
    if purpose not in ("train", "inference"):
        purpose = "inference"
    plan = InfrastructureResourcePlan(
        purpose=purpose,
        required_working_set_gib=40,
        host_memory_components_gib={
            "model_and_runtime": 24,
            "data_and_checkpoint": 16,
        },
        host_memory_formula_gib=64,
        site_min_ram_gib=0,
        num_gpus=gpu_count,
        min_vram_gb=16,
        min_ram_gb=64,
        min_cpus=8,
        time_limit_hours=8 if provider == "cluster" else 0,
        cloud_backend="vastai" if provider == "cloud" else "",
        gpu_type="",
        docker_image=(
            "nvidia/cuda:12.1.1-devel-ubuntu22.04"
            if provider == "cloud" else ""
        ),
        disk_gb=50 if provider == "cloud" else 0,
        slurm_partition=str(payload.get("slurm_partition") or ""),
        slurm_account=str(payload.get("slurm_account") or ""),
        slurm_qos=str(payload.get("slurm_qos") or ""),
        rationale="Stub plan derived from the supplied run context.",
    )
    info = {
        "schema_version": 1,
        "ticket_id": str(payload.get("ticket_id") or "infra-stub"),
        "purpose": purpose,
        "provider": provider,
        "cloud_backend": "vastai" if provider == "cloud" else "",
        "run_id": str(payload.get("run_id") or ""),
        "host": "stub-host", "instance_id": "" if provider == "cluster" else "stub-0",
        "auto_release": auto_release,
        "ssh": {"host": "stub-host", "port": 22, "user": "stub", "key_path": "/tmp/stub-key"},
        "gpu": None if provider == "cluster" else {
            "has_gpu": True, "gpu_count": gpu_count, "gpu_name": "stub-gpu",
            "vram_gb": 24, "vram_mb": 24576,
            "devices": [
                {"index": index, "name": "stub-gpu", "vram_mb": 24576}
                for index in range(gpu_count)
            ],
        },
        "cuda": None if provider == "cluster" else {
            "driver_version": "stub", "cuda_version": "12.1",
            "recommended_torch_index": "cu121",
        },
        "cost": {"dph_total": 0.1 if provider == "cloud" else 0.0},
        "resource_plan": plan.model_dump(),
        "probe_source": "ssh-environment" if provider == "cluster" else "nvidia-smi",
        "probed_at": datetime.now(timezone.utc).isoformat(),
    }
    if provider == "cluster":
        info["cluster"] = {
            "jobid": "",
            "node": "",
            "requested_gpus": gpu_count,
            "partition": "stub",
            "account": "", "qos": "",
            "container_image": "",
            "env_setup": "", "workdir": "/tmp/zevo/stub", "hf_cache": "/tmp/zevo/hf_cache",
        }
    elif provider == "instance":
        info["instance"] = {
            "env_setup": "",
            "workdir": "/tmp/zevo/stub",
            "hf_cache": "/tmp/zevo/hf_cache",
            "visible_devices": ",".join(str(index) for index in range(gpu_count)),
        }
    di.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return InfraResult(
        status="succeeded",
        ticket_id=payload.get("ticket_id", "infra-stub"),
        operation="provision",
        device_info_path=str(di),
        error_message="", notes="stub route/device",
    )


def _stub_train(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.train import TrainResult
    from zevo.contracts.configuration import (
        PromptAlignmentEvidence, SuggestionDecision, TrainRunConfig,
        TrainingConfig, TrainingDataExample, file_sha256,
    )
    from zevo.contracts.prompting import derive_loss_contract, recommended_loss_objective_config
    from zevo.contracts.configuration import load_inference_config, load_train_config

    work_dir.mkdir(parents=True, exist_ok=True)
    iteration = int(payload.get("iteration") or 1)
    inference = load_inference_config(str(payload.get("inference_config_path") or ""))
    pins = dict(payload.get("configuration_pins") or {})
    suggestions = dict(payload.get("configuration_suggestions") or {})
    previous = None
    previous_path = str(payload.get("parent_train_config_path") or "")
    if previous_path:
        previous = load_train_config(previous_path)
    pinned_method = str(
        payload.get("training_method_pin") or pins.get("training_method") or ""
    )
    suggested_method = str(suggestions.get("training_method") or "")
    if pinned_method:
        method = pinned_method
        diversity_status = "user_pinned"
        method_rationale = "The user pinned the training method for every iteration."
    elif previous is None:
        method = suggested_method or "lora_sft"
        diversity_status = "initial"
        method_rationale = "Selected the first method from guidance and available evidence."
    else:
        method = suggested_method or previous.training_method
        if method != previous.training_method:
            diversity_status = "varied"
            transition = dict(payload.get("branch_transition") or {})
            exhaustion = str(transition.get("validation_evidence") or "").strip()
            method_rationale = (
                "The Orchestrator marked the previous method branch exhausted "
                "and explicitly selected the next compatible method branch."
                + (f" Validation evidence: {exhaustion}" if exhaustion else "")
            )
        else:
            diversity_status = "retained_in_branch"
            method_rationale = (
                "Retained the active method while its data and training search "
                "branch is still being explored."
            )
    method_config = dict(
        payload.get("method_config_pins")
        or pins.get("method_config")
        or (
            previous.method_config
            if previous and previous.training_method == method
            else {}
        )
    )
    if method in {
        "dpo", "cpo", "gkd", "grpo", "kto", "online_dpo", "orpo", "rft", "rloo"
    } and "use_peft" not in method_config:
        method_config["use_peft"] = True
    loss = derive_loss_contract(method, inference.prompt.prompt_framing)
    loss.objective_config = recommended_loss_objective_config(
        method,
        dict(payload.get("loss_objective_pins") or pins.get("loss_objective_config") or {}),
    )
    training_values = (
        previous.training.model_dump() if previous else {
            "num_epochs": 1,
            "max_seq_len": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "world_size": 1,
            "effective_batch_size": 1,
            "learning_rate": 1e-4 if method == "lora_sft" else 2e-5,
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
            "lora_r": 16 if method == "lora_sft" else 0,
            "lora_alpha": 32 if method == "lora_sft" else 0,
            "lora_dropout": 0.0,
            "lora_target_modules": ["all-linear"] if method == "lora_sft" else [],
            "implementation_config": {},
            "software_versions": {
                "torch": "stub", "transformers": "stub", "trl": "stub",
                **({"peft": "stub"} if method == "lora_sft" else {}),
            },
        }
    )
    uses_peft = method == "lora_sft" or bool(method_config.get("use_peft", False))
    if uses_peft:
        training_values["software_versions"]["peft"] = "stub"
    if not uses_peft:
        training_values["lora_r"] = 0
        training_values["lora_alpha"] = 0
        training_values["lora_dropout"] = 0.0
        training_values["lora_target_modules"] = []
    elif training_values["lora_r"] == 0:
        training_values["lora_r"] = 16
        training_values["lora_alpha"] = 32
        training_values["lora_target_modules"] = ["all-linear"]
    for key in tuple(training_values):
        if key in suggestions and suggestions[key] not in (None, "", 0):
            training_values[key] = suggestions[key]
        if key in pins:
            training_values[key] = pins[key]
    parent_model = str(payload.get("parent_checkpoint_path") or payload.get("base_model") or "stub/base")
    decisions = []
    for key, suggested in suggestions.items():
        if suggested in (None, "", 0, {}, []):
            continue
        realized = method if key == "training_method" else training_values.get(key, suggested)
        decisions.append(SuggestionDecision(
            key=key,
            decision="accepted" if realized == suggested else "adjusted",
            suggested=suggested,
            realized=realized,
            rationale="Deterministic stub selected a compatible realized value.",
        ))
    input_values = dict(inference.prompt_example.input_values)
    prompt_prefix = inference.prompt_example.rendered_prompt
    context_placeholders = list(input_values.values())
    thinking_enabled = inference.prompt.model_reasoning_type == "thinking"

    def render_target(target: str) -> str:
        thinking_trace = (
            "<THINKING_TRACE>\n</think>\n\n" if thinking_enabled else ""
        )
        return prompt_prefix + thinking_trace + target + "<|im_end|>"

    if method in {"dpo", "cpo", "orpo"}:
        source_record = {
            "prompt": input_values,
            "chosen": "<CHOSEN_RESPONSE>",
            "rejected": "<REJECTED_RESPONSE>",
        }
        rendered_sequences = {
            "chosen": render_target("<CHOSEN_RESPONSE>"),
            "rejected": render_target("<REJECTED_RESPONSE>"),
        }
        loss_targets = ["<CHOSEN_RESPONSE>", "<REJECTED_RESPONSE>"]
    elif method == "gkd":
        source_record = {
            "prompt": input_values,
            "teacher_response": "<TEACHER_RESPONSE>",
        }
        rendered_sequences = {
            "teacher_sequence": render_target("<TEACHER_RESPONSE>"),
        }
        loss_targets = ["<TEACHER_RESPONSE>"]
    elif method in {"grpo", "online_dpo", "rft", "rloo"}:
        source_record = {
            "prompt": input_values,
            "generated_response": "<GENERATED_RESPONSE>",
        }
        rendered_sequences = {
            "rollout": render_target("<GENERATED_RESPONSE>"),
        }
        loss_targets = ["<GENERATED_RESPONSE>"]
    else:
        source_record = {
            "input": input_values,
            "target_response": "<TARGET_RESPONSE>",
        }
        rendered_sequences = {
            "training": render_target("<TARGET_RESPONSE>"),
        }
        loss_targets = ["<TARGET_RESPONSE>"]
    if thinking_enabled:
        source_record["thinking_trace"] = "<THINKING_TRACE>"
        loss_targets = ["<THINKING_TRACE>", *loss_targets]
    training_example = TrainingDataExample(
        source_record=source_record,
        rendered_sequences=rendered_sequences,
        loss_target_placeholders=loss_targets,
        context_only_placeholders=context_placeholders,
        sequence_loss_targets={
            name: [target for target in loss_targets if target in sequence]
            for name, sequence in rendered_sequences.items()
        },
        loss_target_summary=(
            f"The {loss.objective} objective uses the declared response span(s); "
            f"prompt/input placeholders are context under target_scope={loss.target_scope}."
        ),
    )
    train_config = TrainRunConfig(
        iteration=iteration,
        parent_model=parent_model,
        parent_kind="run_checkpoint" if payload.get("parent_checkpoint_path") else "baseline",
        parent_selection_rationale=str(
            payload.get("parent_selection_rationale")
            or "Use the baseline for the initial training experiment."
        ),
        data_signature=str(payload.get("data_signature") or "0" * 64),
        training_method=method,
        method_config=method_config,
        loss_contract=loss,
        training=TrainingConfig.model_validate(training_values),
        prompt=inference.prompt,
        tokenizer_source=inference.tokenizer_source,
        chat_template_source=inference.chat_template_source,
        chat_template_hash=inference.chat_template_hash,
        template_kwargs=inference.template_kwargs,
        special_token_ids=inference.special_token_ids,
        generation_backend=str(payload.get("generation_backend") or "vllm"),
        inference_config_path=str(payload.get("inference_config_path") or ""),
        inference_config_sha256=str(
            payload.get("expected_inference_config_sha256")
            or file_sha256(str(payload.get("inference_config_path") or ""))
        ),
        prompt_alignment=PromptAlignmentEvidence(
            rendered_prompt=prompt_prefix,
            context_token_count=max(1, len(prompt_prefix)),
            target_token_count=1,
            string_prefix_match=True,
            token_prefix_match=True,
            prefix_labels_all_ignored=True,
            target_starts_with_synthetic_response=True,
        ),
        training_data_example=training_example,
        direction=(
            str(suggestions.get("direction") or "")
            or (
                "establish the initial training recipe"
                if iteration == 1 else "test a compatible training-method change"
            )
        ),
        method_diversity_status=diversity_status,
        method_selection_rationale=method_rationale,
        suggestion_decisions=decisions,
    )
    train_config_path = work_dir / "train_config.yaml"
    train_config_path.write_text(
        yaml.safe_dump(train_config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    model_dir = work_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": payload.get("base_model", ""),
                    "training_method": method}, indent=2), encoding="utf-8",
    )
    (model_dir / "adapter_model.safetensors").write_bytes(b"\x00stub\x00")
    train_script = work_dir / "train.py"
    train_script.write_text(
        "from zevo_train_telemetry import ZevoTrainerTelemetryCallback\n"
        "callback = ZevoTrainerTelemetryCallback('train-stub')\n",
        encoding="utf-8",
    )
    train_log = work_dir / "train.log"
    train_log.write_text("stub training completed\n", encoding="utf-8")
    slurm_contract = dict(payload.get("slurm_job") or {})
    slurm_script_path = ""
    if slurm_contract.get("enabled"):
        slurm_script = Path(str(slurm_contract["script_path"]))
        slurm_script.write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH --job-name={slurm_contract['job_name']}\n"
            f"#SBATCH --gpus={slurm_contract['num_gpus']}\n"
            "set -euo pipefail\npython train.py\n",
            encoding="utf-8",
        )
        slurm_script_path = str(slurm_script)
    return TrainResult(
        status="succeeded", operation="train",
        ticket_id=payload.get("ticket_id", "train-stub"),
        train_config_path=str(train_config_path),
        train_script_path=str(train_script),
        slurm_script_path=slurm_script_path,
        log_path=str(train_log),
        checkpoint_path=str(model_dir),
        n_examples=3, final_loss=1.23, training_seconds=1,
        tracking_url=str((payload.get("execution_contract") or {}).get("tracking_url") or ""),
        error_message="", notes="stub model — no real training",
    )


def _stub_infer(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.inference import InferenceResult
    from zevo.contracts.configuration import (
        InferencePromptExample, InferenceRunConfig, PromptMessageExample,
        SuggestionDecision, load_inference_config,
    )
    from zevo.contracts.data import InferenceDataProfile
    from zevo.contracts.prompting import InferenceContract, PromptContract

    work_dir.mkdir(parents=True, exist_ok=True)
    supplied_path = str(payload.get("inference_config_path") or "")
    if supplied_path:
        config = load_inference_config(supplied_path)
        config_path = Path(supplied_path)
    else:
        profile = InferenceDataProfile.model_validate_json(
            Path(str(payload.get("inference_data_profile_path") or "")).read_text(
                encoding="utf-8"
            )
        )
        pins = dict(payload.get("configuration_pins") or {})
        suggestions = dict(payload.get("configuration_suggestions") or {})
        prompt_values = {
            "prompt_framing": pins.get("prompt_framing") or suggestions.get("prompt_framing") or "chat",
            "model_reasoning_type": (
                "thinking"
                if payload.get("verified_model_reasoning_type") == "thinking"
                else "non_thinking"
            ),
            "system_prompt": pins.get("system_prompt") or suggestions.get("system_prompt") or "You are a helpful assistant.",
        }
        suggested_mapping = dict(suggestions.get("inference_config") or {})
        pinned_mapping = dict(pins.get("inference_config") or {})
        inference_values = {
            "inference_config": {
                "input_fields": profile.input_fields,
                "answer_column": (
                    "prediction"
                    if "prediction" in profile.submission_columns
                    else profile.submission_columns[-1]
                ),
                **suggested_mapping,
                **pinned_mapping,
            },
            "decoding_strategy": "greedy",
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "seed": 0,
            **{
                key: suggestions[key]
                for key in (
                    "decoding_strategy", "max_new_tokens", "temperature", "top_p", "top_k",
                    "repetition_penalty", "seed",
                )
                if suggestions.get(key) not in (None, "", 0)
            },
            **dict(suggestions.get("decoding_config") or {}),
            **dict(pins.get("decoding_config") or {}),
        }
        explicitly_selected_strategy = any((
            suggestions.get("decoding_strategy") not in (None, ""),
            "decoding_strategy" in dict(suggestions.get("decoding_config") or {}),
            "decoding_strategy" in dict(pins.get("decoding_config") or {}),
        ))
        if not explicitly_selected_strategy:
            inference_values["decoding_strategy"] = (
                "sampling" if float(inference_values["temperature"]) > 0 else "greedy"
            )
        prompt = PromptContract.model_validate(prompt_values)
        measurement = InferenceContract.model_validate(inference_values)
        realized_values = {
            **prompt.model_dump(),
            **measurement.model_dump(),
        }
        decisions = [
            SuggestionDecision(
                key=key,
                decision=(
                    "accepted" if realized_values.get(key, suggested) == suggested
                    else "adjusted"
                ),
                suggested=suggested,
                realized=realized_values.get(key, suggested),
                rationale="Deterministic stub selected a compatible realized value.",
            )
            for key, suggested in suggestions.items()
            if suggested not in (None, "", 0, {}, [])
        ]
        template = "stub-chat-template-v1"
        input_values = {
            field: f"<INPUT:{field}>"
            for field in measurement.inference_config["input_fields"]
        }
        if measurement.inference_config.get("task_instruction"):
            from zevo.contracts.task_protocol import render_task_user_content

            input_text = render_task_user_content(
                measurement.inference_config, input_values,
            )
        else:
            input_text = "\n".join(
                f"{field}: {value}" for field, value in input_values.items()
            )
        if prompt.prompt_framing == "chat" or prompt.prompt_framing.startswith("chat:"):
            template_kwargs = (
                {"enable_thinking": True}
                if prompt.model_reasoning_type == "thinking"
                else {}
            )
            messages = [
                PromptMessageExample(role="system", content=prompt.system_prompt),
                PromptMessageExample(role="user", content=input_text),
            ]
            rendered_prompt = (
                f"<|im_start|>system\n{prompt.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{input_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
                + ("<think>\n" if prompt.model_reasoning_type == "thinking" else "")
            )
        else:
            template_kwargs = {}
            messages = []
            rendered_prompt = input_text
        config = InferenceRunConfig(
            base_model=str(payload.get("base_model") or "stub/base"),
            prompt=prompt,
            measurement=measurement,
            generation_backend=str(payload.get("generation_backend") or "vllm"),
            implementation_config={},
            tokenizer_source=str(payload.get("base_model") or "stub/base"),
            chat_template_source="tokenizer.chat_template",
            chat_template_hash=hashlib.sha256(template.encode()).hexdigest(),
            template_kwargs=template_kwargs,
            special_token_ids={"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
            stop_token_ids=[2],
            prompt_example=InferencePromptExample(
                input_values=input_values,
                messages=messages,
                rendered_prompt=rendered_prompt,
            ),
            suggestion_decisions=decisions,
        )
        config_path = work_dir / "inference_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    supplied_script = str(payload.get("reusable_predict_script_path") or "")
    predict_script = Path(supplied_script) if supplied_script else work_dir / "predict.py"
    if not supplied_script:
        predict_script.write_text("# stub predict\n", encoding="utf-8")
    preds = work_dir / "predictions.csv"
    preds.write_text("id,prediction\n1,A\n2,B\n3,A\n", encoding="utf-8")
    generation_diagnostics = work_dir / "generation_diagnostics.json"
    generation_diagnostics.write_text(json.dumps({
        "schema_version": 1,
        "records": [
            {
                "request_index": index,
                "row_index": index,
                "turn": 1,
                "finish_reason": "stop",
                "stop_reason": 2,
                "generated_tokens": 1,
            }
            for index in range(3)
        ],
    }, indent=2), encoding="utf-8")
    slurm_contract = dict(payload.get("slurm_job") or {})
    slurm_script_path = ""
    if slurm_contract.get("enabled"):
        slurm_script = Path(str(slurm_contract["script_path"]))
        slurm_script.write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH --job-name={slurm_contract['job_name']}\n"
            f"#SBATCH --gpus={slurm_contract['num_gpus']}\n"
            "set -euo pipefail\npython predict.py\n",
            encoding="utf-8",
        )
        slurm_script_path = str(slurm_script)
    return InferenceResult(
        status="succeeded", operation="run_inference",
        ticket_id=payload.get("ticket_id", "infer-stub"),
        inference_config_path=str(config_path),
        predict_script_path=str(predict_script),
        slurm_script_path=slurm_script_path,
        predictions_path=str(preds),
        generation_diagnostics_path=str(generation_diagnostics),
        n_rows=3, n_requests=3, n_unparseable=0,
        error_message="", notes="stub predictions",
    )


def _stub_eval(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.evaluation import EvaluationResult
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics = work_dir / "metrics.json"
    metrics_data = {
        "score": 0.67,
        "metric": str(payload.get("metric") or "score"),
        "n_correct": 2,
        "n_total": 3,
    }
    metrics.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    return EvaluationResult(
        status="succeeded",
        ticket_id=payload.get("ticket_id", "eval-stub"),
        metrics_path=str(metrics),
        error_message="", notes="stub eval",
    )


def _stub_registry(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.model_registry import (
        RegisterResult,
        RegistryEntry,
        model_tag_for_run,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    register_script = work_dir / "register.py"
    register_script.write_text("# stub register\n", encoding="utf-8")
    registry_path = Path(payload.get("registry_path") or (work_dir / "registry.yaml"))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    version_tag = str(
        payload.get("expected_version_tag")
        or model_tag_for_run(str(payload.get("run_id") or ""))
    )
    metrics = json.loads(Path(str(payload.get("metrics_path") or "")).read_text(
        encoding="utf-8"
    ))
    candidate_score = float(metrics["score"])
    direction = str(payload.get("metric_direction") or "max")
    candidate_model_path = str(payload.get("checkpoint_path") or "")
    selected = RegistryEntry(
            run_id=str(payload.get("run_id") or ""),
            ticket_id=str(payload.get("ticket_id") or "registry-stub"),
            iteration=int(payload.get("iteration") or 0),
            base_model=str(payload.get("base_model") or "stub-base"),
            training_method=str(payload.get("training_method") or "lora_sft"),
            dataset_source=(
                str(payload.get("dataset_source") or "stub-dataset")[len("/app/"):]
                if str(payload.get("dataset_source") or "").startswith("/app/")
                else str(payload.get("dataset_source") or "stub-dataset")
            ),
            model_path=candidate_model_path,
            task_objective=str(payload.get("task_objective") or "stub objective"),
            metric=str(payload.get("metric") or "score"),
            metric_direction=direction,
            eval=metrics,
            registered_at=datetime.now(timezone.utc),
        )
    # JSON is valid YAML. Keep the stub dependency-free while still producing
    # the Ticket-local registry manifest the runner and database mirror expect.
    registry_path.write_text(json.dumps({
        "models": {version_tag: selected.model_dump(mode="json")},
    }, indent=2), encoding="utf-8")
    return RegisterResult(
        status="succeeded",
        ticket_id=payload.get("ticket_id", "registry-stub"),
        registry_path=str(registry_path),
        register_script_path=str(register_script),
        error_message="", notes="stub registry snapshot",
    )


def _stub_orchestrator(payload: dict, work_dir: Path) -> BaseModel:
    from zevo.contracts.orchestrator import SupervisorAction
    # Governance actions such as setup expansion, Ticket creation, Journal
    # PATCHes, and Run finalization are authenticated API side effects. The
    # in-process stub deliberately performs none of them; it only supplies a
    # valid no-op result for tests that exercise the Agent interface itself.
    return SupervisorAction(
        ticket_id=str(payload.get("ticket_id") or "orchestrate-stub"),
        action="wait",
        child_ticket_id="",
        summary="stub orchestrator does not perform control-plane side effects",
    )


# Map agent_id → factory. Add a new agent here when you ship one.
STUB_FACTORIES: dict[str, StubFactory] = {
    "data":            _stub_data,
    "infrastructure":  _stub_infra,
    "train":      _stub_train,
    "inference":       _stub_infer,
    "evaluation":      _stub_eval,
    "registry":  _stub_registry,
    "orchestrator":    _stub_orchestrator,
}


class StubDriver:
    """Deterministic, no-IO driver for hermetic tests. Picks a factory
    by blueprint.id; falls back to raising if unknown so a new agent
    can't silently sneak into the suite without coverage."""

    name = "stub"

    async def run_agent(
        self,
        *,
        blueprint: AgentBlueprint,
        input_payload: BaseModel,
        workspace_dir: str,
        stdout_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        conversation: list[dict[str, Any]] | None = None,
        max_turns: int = 60,
        model_override: str = "",
        sandbox_mode: str = "",   # accepted for interface parity; stub does not sandbox
    ) -> DriverRunResult:
        wd = Path(workspace_dir)
        wd.mkdir(parents=True, exist_ok=True)

        factory = STUB_FACTORIES.get(blueprint.id)
        if factory is None:
            raise RuntimeError(
                f"stub driver has no factory for agent {blueprint.id!r}. "
                f"Add an entry to zevo.engine.agent.drivers.stub.STUB_FACTORIES."
            )

        payload_dict = input_payload.model_dump() if hasattr(input_payload, "model_dump") else dict(input_payload)
        output = factory(payload_dict, wd)

        # The stub is a real dry-run implementation of the Agent interface, so
        # it obeys the same mandatory runtime-contract telemetry as production
        # Train and Inference drivers. A test double that omits this would now
        # correctly fail and could no longer exercise the downstream pipeline.
        if (
            stdout_sink is not None
            and blueprint.id in {"train", "inference"}
            and payload_dict.get("operation") in {"train", "run_inference"}
        ):
            config = {
                key: payload_dict[key]
                for key in (
                    "base_model", "model_source", "training_method",
                    "loss_contract", "method_config", "generation_backend",
                    "num_epochs", "max_seq_len", "batch_size", "learning_rate",
                    "lora_r", "lora_alpha",
                    "decoding_strategy", "max_new_tokens", "temperature", "top_p", "top_k",
                    "repetition_penalty", "seed", "prompt_framing",
                    "model_reasoning_type", "system_prompt",
                )
                if key in payload_dict
            }
            config.update({
                "prompt": "<|user|>{INPUT}<|assistant|>",
                "tokenizer_source": "stub/exact-tokenizer",
                "chat_template_hash": "stub-chat-template-sha256",
                "special_token_ids": {"bos": 1, "eos": 2, "pad": 0},
            })
            scope = dict(payload_dict.get("loss_contract") or {}).get("target_scope")
            if scope in {"assistant_messages", "completion", "all_tokens"}:
                config.update({
                    "assistant_only_loss": scope == "assistant_messages",
                    "completion_only_loss": scope == "completion",
                })
            for key, value in dict(payload_dict.get("inference_config") or {}).items():
                config[key] = value
            stdout_sink(
                f"__CONFIG__:{payload_dict.get('ticket_id', '')}:"
                f"{json.dumps(config, sort_keys=True)}\n"
            )

        # Emit synthetic events so the runner's event flusher / cost
        # accounting / transcript don't blow up.
        if event_sink is not None:
            event_sink({"type": "agent_message",
                        "payload": {"message": f"stub: {blueprint.id} ran"}})
            event_sink({"type": "turn_completed", "payload": {
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "cached_input_tokens": 0, "reasoning_output_tokens": 0},
            }})
            event_sink({"type": "task_complete",
                        "payload": {"result_preview": f"stub for {blueprint.id}"}})

        return DriverRunResult(
            output=output,
            raw_stdout=f"stub driver: {blueprint.id}\n",
            raw_stderr="",
            exit_code=0,
            driver=self.name,
            model=model_override or "stub-model",
        )
