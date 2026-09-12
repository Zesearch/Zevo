"""Durable YAML configurations produced by Inference and Train.

The files modeled here are the execution contracts.  Orchestrator suggestions
are advisory inputs; the Specialist decides the realized values and records the
decision once in YAML.  Later stages consume the YAML instead of reconstructing
configuration from prose or from another Ticket payload.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from zevo.contracts.data import is_sha256
from zevo.contracts.prompting import (
    InferenceContract,
    LOSS_OBJECTIVE_CONFIG_KEYS,
    LOSS_OBJECTIVE_RECOMMENDED_STARTS,
    LossContract,
    PromptContract,
    derive_loss_contract,
    validate_inference_config,
    validate_loss_objective_config,
)
from zevo.contracts.training_methods import (
    AUXILIARY_MODEL_FIELD_BY_METHOD,
    METHOD_CONFIG_KEYS,
    method_config_errors,
    normalize_method_config,
)


def inference_mapping_contract() -> dict[str, Any]:
    """The closed nested namespace not expressible by dict[str, Any]."""
    return {
        "allowed_keys": {
            "input_fields": "list[non-empty string]",
            "inference_query": "non-empty string; supports {input} or {field} placeholders",
            "answer_regex": "string",
            "answer_column": "string",
            "batch_size": "integer >= 1",
            "stop": "list[non-empty string]",
            # Opt-in multiple-choice option scoring. Absent = default free
            # generation. When "option_loglikelihood", predict.py emits per-
            # option log-likelihoods (see playbook/runners/option_scoring.py)
            # and option_fields names the answer-option columns.
            "scoring_mode": "'generate' | 'option_loglikelihood'",
            "option_fields": "list[non-empty string]",
        },
        "required_keys": ["input_fields", "answer_column"],
        "authority": "measurement.inference_config",
    }


# LoRA adapter starting points recommended whenever PEFT is active for ANY
# method. Grounded in 2025-2026 practice ("LoRA Without Regret", Thinking
# Machines; "LoRA Learns Less and Forgets Less", arXiv 2405.09673; Raschka):
# adapt ALL linear layers (not just attention q/v), keep alpha = 2*rank, and a
# small dropout. The learning rate that pairs with these lives per-method in
# RECOMMENDED_TRAINING_STARTS because a LoRA LR is ~10x a full-finetuning LR.
RECOMMENDED_LORA_ADAPTER_START: dict[str, Any] = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": ["all-linear"],
}

# Recommended common-trainer starting values, surfaced to the Train agent as
# GUIDANCE (parallel to LOSS_OBJECTIVE_RECOMMENDED_STARTS in prompting.py).
# These are non-binding starting points, NOT caps: the Train agent may realize
# different values with a recorded rationale, and TrainingConfig performs no
# validation against this table. Only method-appropriate knobs the research
# report grounds are listed; a method absent here has no specific start beyond
# the generic YAML defaults. LoRA fields appear only where PEFT is the
# recommended default; when a method is run full-parameter, zero/empty them.
RECOMMENDED_TRAINING_STARTS: dict[str, dict[str, Any]] = {
    "lora_sft": {
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "effective_batch_size": 32,
        "packing": True,
        **RECOMMENDED_LORA_ADAPTER_START,
    },
    "full_sft": {
        "num_epochs": 3,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "effective_batch_size": 32,
        "packing": True,
        "lora_r": 0,
        "lora_alpha": 0,
        "lora_dropout": 0.0,
        "lora_target_modules": [],
    },
    "rft": {
        "num_epochs": 2,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "effective_batch_size": 32,
        "packing": True,
        **RECOMMENDED_LORA_ADAPTER_START,
    },
    "grpo": {
        "num_epochs": 1,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "effective_batch_size": 32,
        "packing": False,
    },
}


def recommended_training_starts(training_method: object) -> dict[str, Any]:
    """Non-binding common-trainer starting values for a method, if grounded.

    Returns a fresh copy so callers cannot mutate the shared table. A method
    without an entry returns an empty mapping, meaning "no specific start".
    """
    method = (
        training_method.strip().lower()
        if isinstance(training_method, str) else ""
    )
    return dict(RECOMMENDED_TRAINING_STARTS.get(method, {}))


def train_method_contracts() -> dict[str, Any]:
    """Exact method/loss key sets supplementing the generic YAML mappings."""
    contracts: dict[str, Any] = {}
    for method in sorted(METHOD_CONFIG_KEYS):
        method_keys = sorted(METHOD_CONFIG_KEYS[method])
        dependency = AUXILIARY_MODEL_FIELD_BY_METHOD.get(method)
        method_required = [
            key for key in method_keys if key == "use_peft" or key == dependency
        ]
        contracts[method] = {
            "method_config": {
                "allowed_keys": method_keys,
                "required_keys": method_required,
                "value_contracts": {
                    key: (
                        "boolean" if key == "use_peft"
                        else "Hugging Face model id in owner/model form"
                    )
                    for key in method_keys
                },
            },
            "loss_objective_config": {
                "allowed_keys": sorted(LOSS_OBJECTIVE_CONFIG_KEYS[method]),
                "required_realized_keys": sorted(
                    LOSS_OBJECTIVE_RECOMMENDED_STARTS[method]
                ),
                "recommended_start": dict(
                    LOSS_OBJECTIVE_RECOMMENDED_STARTS[method]
                ),
            },
            "recommended_training_starts": {
                "guidance": (
                    "Non-binding best-practice starting values for the common "
                    "TrainingConfig knobs (LoRA rank/alpha/dropout/target_modules, "
                    "learning_rate, effective_batch_size, warmup_ratio, "
                    "lr_scheduler_type, num_epochs, packing). Start here, then "
                    "deviate for one coherent direction with a recorded rationale; "
                    "these are guidance, not caps, and the schema does not enforce "
                    "them. LoRA fields apply when PEFT is active; zero/empty them "
                    "for full-parameter training."
                ),
                "values": recommended_training_starts(method),
                "lora_adapter_start_when_peft_active": (
                    dict(RECOMMENDED_LORA_ADAPTER_START)
                    if (method == "lora_sft" or "use_peft" in method_keys)
                    else None
                ),
            },
            "loss_contract": {
                "owner": "train_skill",
                "framework": "trl",
                "by_prompt_framing": {
                    framing: derive_loss_contract(method, framing).model_dump(mode="json")
                    for framing in ("chat", "completion", "text")
                },
                "instruction": (
                    "Select the entry matching inference_config.yaml prompt.prompt_framing; "
                    "copy owner/framework/objective/target_scope literally, then fill only "
                    "the documented loss_objective_config keys."
                ),
            },
            "runtime_contract": {
                "required_software_versions": ["torch", "transformers", "trl"],
                "peft_activation": (
                    "always"
                    if method == "lora_sft"
                    else (
                        "when method_config.use_peft is true"
                        if "use_peft" in method_keys else "never"
                    )
                ),
                "additional_software_version_when_peft_active": "peft",
                "lora_fields_when_peft_inactive": {
                    "lora_r": 0,
                    "lora_alpha": 0,
                    "lora_dropout": 0.0,
                    "lora_target_modules": [],
                },
                "hash_encoding": (
                    "exactly 64 lowercase hexadecimal SHA-256 characters "
                    "with no prefix"
                ),
            },
        }
    return contracts


class SuggestionDecision(BaseModel):
    """How a Specialist handled one non-binding Orchestrator suggestion."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    decision: Literal["accepted", "adjusted", "self_selected"]
    suggested: Any = None
    realized: Any
    rationale: str = Field(min_length=1)


_PLACEHOLDER_RE = re.compile(r"^<[^<>\r\n]+>$")
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)


def _placeholder_values(value: Any) -> set[str]:
    """Collect whole-string synthetic placeholders from a JSON-like value."""
    if isinstance(value, str):
        return {value} if _PLACEHOLDER_RE.fullmatch(value) else set()
    if isinstance(value, dict):
        return set().union(*(_placeholder_values(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_placeholder_values(item) for item in value), set())
    return set()


def _contains_think_tag(value: str) -> bool:
    return bool(_THINK_TAG_RE.search(value))


def _think_blocks(value: str) -> list[str]:
    return _THINK_BLOCK_RE.findall(value)


class PromptMessageExample(BaseModel):
    """One synthetic message before tokenizer/template rendering."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class InferencePromptExample(BaseModel):
    """Human-readable proof of the exact prompt shape sent to the model."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic_placeholders"] = "synthetic_placeholders"
    input_values: dict[str, str] = Field(
        min_length=1,
        description=(
            "Ordered task input mapping using <INPUT:field> placeholders; never "
            "copy a Validation/Test answer into this example."
        ),
    )
    messages: list[PromptMessageExample] = Field(
        default_factory=list,
        description="Pre-template messages for chat framing; empty for text/completion framing.",
    )
    rendered_prompt: str = Field(
        min_length=1,
        description=(
            "Exact tokenizer/template output for the synthetic input, including "
            "special tokens and the assistant generation prefix when applicable."
        ),
    )
    generation_starts_at: Literal["end_of_rendered_prompt"] = "end_of_rendered_prompt"

    @model_validator(mode="after")
    def require_synthetic_inputs(self) -> "InferencePromptExample":
        expected = {key: f"<INPUT:{key}>" for key in self.input_values}
        if self.input_values != expected:
            raise ValueError(
                "prompt_example.input_values must use exact <INPUT:field> placeholders"
            )
        if not any(value in self.rendered_prompt for value in self.input_values.values()):
            raise ValueError("rendered_prompt must contain at least one declared input placeholder")
        return self


class TrainingDataExample(BaseModel):
    """Synthetic view of one method-shaped record and its realized training text."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic_placeholders"] = "synthetic_placeholders"
    source_record: dict[str, Any] = Field(
        min_length=1,
        description="Method-shaped normalized record using synthetic placeholders only.",
    )
    rendered_sequences: dict[str, str] = Field(
        min_length=1,
        description=(
            "Named exact tokenizer/template outputs consumed by the selected trainer "
            "(for example training, chosen, and rejected sequences)."
        ),
    )
    loss_target_placeholders: list[str] = Field(
        min_length=1,
        description="Placeholder spans that participate in the method's objective.",
    )
    context_only_placeholders: list[str] = Field(
        default_factory=list,
        description="Placeholder spans supplied as context but excluded from direct token loss.",
    )
    sequence_loss_targets: dict[str, list[str]] = Field(
        ...,
        description=(
            "Per-rendered-sequence objective spans. Keys must cover every entry "
            "in rendered_sequences; values preserve supervised-span order. This "
            "is the display and audit authority for arbitrary normalized record "
            "shapes, including multi-turn SFT and preference branches."
        ),
    )
    loss_target_summary: str = Field(
        min_length=1,
        description=(
            "Concise explanation containing the exact loss objective name and "
            "target_scope=<value> used by this config."
        ),
    )

    @model_validator(mode="after")
    def require_auditable_placeholder_example(self) -> "TrainingDataExample":
        if any(not key.strip() or not value for key, value in self.rendered_sequences.items()):
            raise ValueError("training_data_example rendered sequence names/text must be non-empty")
        targets = self.loss_target_placeholders
        context = self.context_only_placeholders
        declared = targets + context
        if len(declared) != len(set(declared)):
            raise ValueError("training_data_example placeholders must be unique")
        if any(not _PLACEHOLDER_RE.fullmatch(value) for value in declared):
            raise ValueError("training_data_example placeholders must use <PLACEHOLDER> form")
        record_placeholders = _placeholder_values(self.source_record)
        undeclared = sorted(record_placeholders - set(declared))
        if undeclared:
            raise ValueError(
                f"training_data_example source_record has undeclared placeholders: {undeclared}"
            )
        rendered = "\n".join(self.rendered_sequences.values())
        absent = sorted(value for value in targets if value not in rendered)
        if absent:
            raise ValueError(
                f"training_data_example rendered_sequences omit loss targets: {absent}"
            )
        unknown_sequences = sorted(
            set(self.sequence_loss_targets) - set(self.rendered_sequences)
        )
        if unknown_sequences:
            raise ValueError(
                "training_data_example sequence_loss_targets names unknown "
                f"rendered sequences: {unknown_sequences}"
            )
        missing_sequences = sorted(
            set(self.rendered_sequences) - set(self.sequence_loss_targets)
        )
        if missing_sequences:
            raise ValueError(
                "training_data_example sequence_loss_targets omits rendered "
                f"sequences: {missing_sequences}"
            )
        for name, sequence_targets in self.sequence_loss_targets.items():
            if not sequence_targets:
                raise ValueError(
                    "training_data_example sequence_loss_targets values must be non-empty"
                )
            if len(sequence_targets) != len(set(sequence_targets)):
                raise ValueError(
                    "training_data_example sequence_loss_targets must not contain duplicates"
                )
            undeclared_targets = sorted(set(sequence_targets) - set(targets))
            if undeclared_targets:
                raise ValueError(
                    "training_data_example sequence_loss_targets contains undeclared "
                    f"targets: {undeclared_targets}"
                )
            missing_from_sequence = sorted(
                target for target in sequence_targets
                if target not in self.rendered_sequences[name]
            )
            if missing_from_sequence:
                raise ValueError(
                    f"training_data_example sequence {name!r} omits its declared "
                    f"targets: {missing_from_sequence}"
                )
        mapped_targets = {
            target
            for sequence_targets in self.sequence_loss_targets.values()
            for target in sequence_targets
        }
        unmapped_targets = sorted(set(targets) - mapped_targets)
        if unmapped_targets:
            raise ValueError(
                "training_data_example sequence_loss_targets omits declared loss "
                f"targets: {unmapped_targets}"
            )
        return self


class PromptAlignmentEvidence(BaseModel):
    """Measured proof that Train reused Baseline prompt rendering and masking."""

    model_config = ConfigDict(extra="forbid")

    rendered_prompt: str = Field(
        min_length=1,
        description=(
            "Prompt rendered by Train from the Baseline synthetic messages and "
            "template_kwargs; it must exactly equal Inference prompt_example."
        ),
    )
    context_token_count: int = Field(ge=1)
    target_token_count: int = Field(ge=1)
    string_prefix_match: Literal[True]
    token_prefix_match: Literal[True]
    prefix_labels_all_ignored: Literal[True]
    target_starts_with_synthetic_response: Literal[True]


class InferenceRunConfig(BaseModel):
    """Complete, reusable configuration fixed by baseline Inference."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[5] = 5
    base_model: str = Field(min_length=1)
    prompt: PromptContract
    measurement: InferenceContract
    generation_backend: Literal["hf", "vllm"]
    implementation_config: dict[str, Any] = Field(
        ...,
        description=(
            "Every fixed backend/framework argument and deterministic runtime "
            "policy not already represented by measurement. Device-derived "
            "arguments such as vLLM gpu_memory_utilization are computed from "
            "the recorded memory_plan and logged at execution time. This "
            "namespace is backend-specific and reused unchanged."
        ),
    )
    tokenizer_source: str = Field(min_length=1)
    chat_template_source: str = ""
    chat_template_hash: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "Empty when no chat template applies; otherwise the bare SHA-256 "
            "of the exact template text: 64 lowercase hexadecimal characters "
            "with no 'sha256:' prefix."
        ),
    )
    template_kwargs: dict[str, JsonValue] = Field(
        ...,
        description=(
            "Exact extra keyword arguments passed to the tokenizer chat template. "
            "Use an empty mapping when no extra argument is supported or needed. "
            "Execution controls such as tokenize and add_generation_prompt do not "
            "belong here."
        ),
    )
    special_token_ids: dict[str, int | None] = Field(default_factory=dict)
    stop_token_ids: list[int] = Field(
        min_length=1,
        description=(
            "Verified tokenizer token ids that terminate one generated response, "
            "including EOS and any chat end-of-turn marker. Inference must pass "
            "these ids to the generation backend; string stop matching is only an "
            "additional boundary and cannot replace this list."
        ),
    )
    prompt_example: InferencePromptExample
    suggestion_decisions: list[SuggestionDecision] = Field(default_factory=list)

    @field_validator("template_kwargs")
    @classmethod
    def validate_template_kwargs(
        cls, value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        reserved = {
            "tokenize", "add_generation_prompt", "continue_final_message",
            "return_dict", "return_tensors",
        }
        invalid = sorted(
            key for key in value
            if not key.strip() or key in reserved
        )
        if invalid:
            raise ValueError(
                "template_kwargs contains empty or execution-control keys: "
                + ", ".join(invalid)
            )
        return value

    @model_validator(mode="after")
    def require_template_identity_for_chat(self) -> "InferenceRunConfig":
        self.measurement.inference_config = validate_inference_config(
            self.measurement.inference_config,
            require_complete=True,
        )
        framing = self.prompt.prompt_framing
        input_fields = list(self.measurement.inference_config.get("input_fields") or [])
        if list(self.prompt_example.input_values) != input_fields:
            raise ValueError(
                "prompt_example.input_values must preserve measurement.inference_config "
                "input_fields order"
            )
        inference_query = str(
            self.measurement.inference_config.get("inference_query") or ""
        )
        if inference_query:
            from zevo.contracts.prompting import render_inference_query

            expected_user = render_inference_query(
                inference_query, dict(self.prompt_example.input_values),
            )
            if framing == "chat" or framing.startswith("chat:"):
                user_messages = [
                    message.content for message in self.prompt_example.messages
                    if message.role == "user"
                ]
                if user_messages != [expected_user]:
                    raise ValueError(
                        "prompt_example user turn differs from inference_query"
                    )
            elif expected_user not in self.prompt_example.rendered_prompt:
                raise ValueError("rendered_prompt omits inference_query")
        if (framing == "chat" or framing.startswith("chat:")) and not (
            self.chat_template_source and self.chat_template_hash
        ):
            raise ValueError(
                "chat inference config requires chat_template_source and chat_template_hash"
            )
        if self.chat_template_hash and not is_sha256(self.chat_template_hash):
            raise ValueError("chat_template_hash must be lowercase hexadecimal SHA-256")
        if (
            any(type(token_id) is not int or token_id < 0 for token_id in self.stop_token_ids)
            or len(self.stop_token_ids) != len(set(self.stop_token_ids))
        ):
            raise ValueError(
                "stop_token_ids must contain unique non-negative integer token ids"
            )
        known_special_ids = {
            token_id for token_id in self.special_token_ids.values()
            if type(token_id) is int and token_id >= 0
        }
        unknown_stop_ids = sorted(set(self.stop_token_ids) - known_special_ids)
        if unknown_stop_ids:
            raise ValueError(
                "stop_token_ids must be verified members of special_token_ids; "
                f"unknown ids: {unknown_stop_ids}"
            )
        eos_id = self.special_token_ids.get("eos_token_id")
        if type(eos_id) is int and eos_id not in self.stop_token_ids:
            raise ValueError("stop_token_ids must include special_token_ids.eos_token_id")
        sampling_params = self.implementation_config.get("sampling_params")
        if isinstance(sampling_params, dict) and "stop_token_ids" in sampling_params:
            raise ValueError(
                "stop_token_ids has one top-level authority and must not be "
                "duplicated in implementation_config.sampling_params"
            )
        if framing == "chat" or framing.startswith("chat:"):
            if not self.prompt_example.messages:
                raise ValueError("chat prompt_example requires pre-template messages")
            first = self.prompt_example.messages[0]
            if first.role != "system" or first.content != self.prompt.system_prompt:
                raise ValueError(
                    "chat prompt_example must begin with the realized system prompt"
                )
            if self.prompt.system_prompt not in self.prompt_example.rendered_prompt:
                raise ValueError("rendered_prompt omits the realized system prompt")
        elif self.prompt_example.messages:
            raise ValueError("text/completion prompt_example must not invent chat messages")
        if not (framing == "chat" or framing.startswith("chat:")) and self.template_kwargs:
            raise ValueError("non-chat inference config requires empty template_kwargs")
        duplicate_template_keys = sorted(
            set(self.template_kwargs) & set(self.implementation_config)
        )
        if duplicate_template_keys:
            raise ValueError(
                "template kwargs must not be duplicated in implementation_config: "
                + ", ".join(duplicate_template_keys)
            )
        if "enable_thinking" in self.template_kwargs:
            enable_thinking = self.template_kwargs["enable_thinking"]
            if not isinstance(enable_thinking, bool):
                raise ValueError("template_kwargs.enable_thinking must be boolean")
            if enable_thinking is not True:
                raise ValueError(
                    "non-thinking models omit template_kwargs.enable_thinking"
                )
            if self.prompt.model_reasoning_type != "thinking":
                raise ValueError(
                    "template_kwargs.enable_thinking contradicts the selected "
                    "model reasoning type"
                )
        rendered_prompt = self.prompt_example.rendered_prompt
        if any(not block.strip() for block in _think_blocks(rendered_prompt)):
            raise ValueError("thinking-model rendering requires reasoning content")
        if self.prompt.model_reasoning_type == "non_thinking" and _contains_think_tag(
            rendered_prompt
        ):
            raise ValueError("non-thinking model rendering must not contain think tags")
        if self.prompt.model_reasoning_type == "thinking" and not _contains_think_tag(
            rendered_prompt
        ):
            raise ValueError(
                "thinking model requires its verified thinking-template "
                "rendering in prompt_example"
            )
        if len({d.key for d in self.suggestion_decisions}) != len(self.suggestion_decisions):
            raise ValueError("suggestion_decisions keys must be unique")
        if (
            self.measurement.decoding_strategy == "greedy"
            and self.measurement.temperature != 0
        ):
            raise ValueError("greedy decoding requires temperature=0")
        if (
            self.measurement.decoding_strategy == "sampling"
            and self.measurement.temperature <= 0
        ):
            raise ValueError("sampling decoding requires temperature>0")
        return self


class CheckpointRetentionConfig(BaseModel):
    """Optional, bounded retention of branchable intermediate model weights.

    Reported intermediate checkpoints are compact experiment branch points.
    A finite Slurm job may keep full Trainer snapshots while it is active so it
    can resume exactly after walltime; after successful completion, selected
    branch points are compacted to model-only artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["none", "steps", "epoch", "selective"]
    save_steps: int = Field(
        0,
        ge=0,
        description="Positive save cadence only when strategy=steps; otherwise 0.",
    )
    max_intermediate_checkpoints: int = Field(
        ge=0,
        le=8,
        description=(
            "Hard cap on retained intermediate weight directories, excluding "
            "the Train Ticket's final checkpoint. When retention is enabled, "
            "Transformers/TRL save_total_limit must equal this value plus one: "
            "the extra temporary slot prevents an automatic terminal checkpoint "
            "from rotating out the oldest intended branch point."
        ),
    )
    save_only_model: Literal[True] = Field(
        True,
        description=(
            "Always true for the intermediate artifacts retained and reported "
            "after successful training. This does not require active Trainer "
            "recovery snapshots to be model-only."
        ),
    )
    rationale: str = ""

    @model_validator(mode="after")
    def validate_bounded_policy(self) -> "CheckpointRetentionConfig":
        if self.strategy == "none":
            if self.save_steps or self.max_intermediate_checkpoints:
                raise ValueError(
                    "checkpoint strategy=none requires save_steps=0 and "
                    "max_intermediate_checkpoints=0"
                )
            return self
        if self.max_intermediate_checkpoints < 1:
            raise ValueError(
                "checkpoint retention requires max_intermediate_checkpoints >= 1"
            )
        if not self.rationale.strip():
            raise ValueError("checkpoint retention requires a concrete rationale")
        if self.strategy == "steps" and self.save_steps < 1:
            raise ValueError("checkpoint strategy=steps requires save_steps >= 1")
        if self.strategy != "steps" and self.save_steps:
            raise ValueError("save_steps is valid only for checkpoint strategy=steps")
        return self


# ---------------------------------------------------------------------------
# Typed distributed / parallelism plan (single-GPU default, FSDP, DeepSpeed,
# multi-node). Grounded in the SOTA research report Section 3 / recommendation
# #5 (FSDP FULL_SHARD vs DeepSpeed ZeRO-2/3 + CPU offload; multi-node
# torchrun/deepspeed launch). The DEFAULT (single_gpu, one node, one GPU per
# node) is exactly today's single-process behavior; a TrainingConfig that omits
# the optional `distributed` field is treated as this default and its launch
# command remains the plain `python <train_script>`.
# ---------------------------------------------------------------------------

DISTRIBUTED_BACKENDS = ("single_gpu", "ddp", "fsdp", "deepspeed")
FSDP_SHARDING_STRATEGIES = ("full_shard", "shard_grad_op", "hybrid_shard")


class DistributedConfig(BaseModel):
    """Typed multi-GPU / multi-node parallelism plan for one training run.

    Backends:
      - ``single_gpu``  one process on one GPU (the unchanged default).
      - ``ddp``         DistributedDataParallel: replicate the model, shard the
                        data across ``nodes * gpus_per_node`` ranks.
      - ``fsdp``        Fully Sharded Data Parallel (PyTorch-native; a
                        ZeRO-3-equivalent full shard) with optional CPU offload.
      - ``deepspeed``   DeepSpeed ZeRO stage 2 or 3 with optional optimizer and
                        parameter CPU offload for models larger than aggregate
                        VRAM.

    Tensor- and sequence-parallel degrees subdivide the data-parallel world for
    large models / long context. ``world_size`` is always
    ``nodes * gpus_per_node`` so it composes with
    ``effective_batch_size = batch_size * gradient_accumulation_steps *
    world_size``.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["single_gpu", "ddp", "fsdp", "deepspeed"] = "single_gpu"
    nodes: int = Field(1, ge=1, description="Physical node count; 1 = single node.")
    gpus_per_node: int = Field(
        1, ge=1, description="Data-parallel processes (ranks) launched per node."
    )
    # FSDP-only options.
    fsdp_sharding: Literal["", "full_shard", "shard_grad_op", "hybrid_shard"] = ""
    fsdp_offload: bool = Field(
        False, description="FSDP CPU offload of params/grads (FSDP backend only)."
    )
    # DeepSpeed-only options.
    zero_stage: Literal[0, 2, 3] = Field(
        0, description="DeepSpeed ZeRO stage; 2 or 3 for the deepspeed backend."
    )
    offload_optimizer: bool = Field(
        False, description="DeepSpeed optimizer-state CPU offload (deepspeed backend)."
    )
    offload_params: bool = Field(
        False, description="DeepSpeed ZeRO-3 parameter CPU offload (deepspeed backend)."
    )
    # Model-parallel degrees that subdivide the world (fsdp/deepspeed only).
    tensor_parallel_size: int = Field(
        1, ge=1, description="Intra-node tensor-parallel degree; 1 = no TP."
    )
    sequence_parallel: bool = Field(
        False, description="Enable sequence/context parallelism for long context."
    )
    activation_checkpointing: bool = Field(
        False,
        description=(
            "Distributed activation checkpointing, coupled with (and additional "
            "to) TrainingConfig.gradient_checkpointing on multi-GPU runs."
        ),
    )

    @property
    def world_size(self) -> int:
        """Total process/rank count: nodes * gpus_per_node."""
        return self.nodes * self.gpus_per_node

    @model_validator(mode="after")
    def validate_backend_shape(self) -> "DistributedConfig":
        deepspeed_fields = bool(
            self.zero_stage or self.offload_optimizer or self.offload_params
        )
        fsdp_fields = bool(self.fsdp_sharding or self.fsdp_offload)

        if self.backend == "single_gpu":
            if self.nodes != 1 or self.gpus_per_node != 1:
                raise ValueError(
                    "single_gpu backend requires nodes=1 and gpus_per_node=1"
                )
            if fsdp_fields or deepspeed_fields:
                raise ValueError(
                    "single_gpu backend cannot set FSDP/DeepSpeed sharding or offload"
                )
            if self.tensor_parallel_size != 1 or self.sequence_parallel:
                raise ValueError(
                    "single_gpu backend cannot set tensor/sequence parallelism"
                )
            return self

        if self.backend == "ddp":
            if fsdp_fields or deepspeed_fields:
                raise ValueError(
                    "ddp backend replicates the model and cannot set FSDP/DeepSpeed "
                    "sharding or offload"
                )
            if self.tensor_parallel_size != 1 or self.sequence_parallel:
                raise ValueError(
                    "ddp backend cannot set tensor/sequence parallelism; use fsdp "
                    "or deepspeed"
                )
        elif self.backend == "fsdp":
            if self.fsdp_sharding not in FSDP_SHARDING_STRATEGIES:
                raise ValueError(
                    "fsdp backend requires fsdp_sharding in "
                    f"{FSDP_SHARDING_STRATEGIES}"
                )
            if deepspeed_fields:
                raise ValueError(
                    "fsdp backend cannot set DeepSpeed zero_stage/offload options"
                )
        elif self.backend == "deepspeed":
            if self.zero_stage not in (2, 3):
                raise ValueError("deepspeed backend requires zero_stage 2 or 3")
            if fsdp_fields:
                raise ValueError(
                    "deepspeed backend cannot set FSDP sharding/offload options"
                )
            if self.offload_params and self.zero_stage != 3:
                raise ValueError(
                    "DeepSpeed parameter offload requires zero_stage=3"
                )

        if self.tensor_parallel_size > 1:
            if self.backend not in ("fsdp", "deepspeed"):
                raise ValueError(
                    "tensor_parallel_size>1 requires the fsdp or deepspeed backend"
                )
            if self.gpus_per_node % self.tensor_parallel_size != 0:
                raise ValueError(
                    "tensor_parallel_size must evenly divide gpus_per_node"
                )
        if self.sequence_parallel and self.backend not in ("fsdp", "deepspeed"):
            raise ValueError(
                "sequence_parallel requires the fsdp or deepspeed backend"
            )
        return self


class DistributedLaunchPlan(BaseModel):
    """Deterministic launcher command derived from a DistributedConfig.

    This is the single source of truth for how a training script is launched
    across GPUs/nodes. Train generates the same command; the single-GPU plan is
    the plain ``python <train_script>`` invocation used today.
    """

    model_config = ConfigDict(extra="forbid")

    launcher: Literal["python", "torchrun", "deepspeed"]
    command: list[str] = Field(min_length=2)
    nnodes: int = Field(ge=1)
    nproc_per_node: int = Field(ge=1)
    fsdp_config: dict[str, Any] | None = Field(
        default=None,
        description="HuggingFace/accelerate FSDP config; set only for the fsdp backend.",
    )
    deepspeed_config: dict[str, Any] | None = Field(
        default=None,
        description="DeepSpeed ZeRO config JSON; set only for the deepspeed backend.",
    )


def build_fsdp_config(dist: DistributedConfig) -> dict[str, Any]:
    """HuggingFace TrainingArguments-shaped FSDP config for an FSDP run."""
    if dist.backend != "fsdp":
        raise ValueError("build_fsdp_config requires the fsdp backend")
    return {
        "fsdp": f"{dist.fsdp_sharding} auto_wrap"
        + (" offload" if dist.fsdp_offload else ""),
        "fsdp_config": {
            "sharding_strategy": dist.fsdp_sharding.upper(),
            "cpu_offload": dist.fsdp_offload,
            "auto_wrap_policy": "transformer_based_wrap",
            "backward_prefetch": "backward_pre",
            "activation_checkpointing": dist.activation_checkpointing,
        },
    }


def build_deepspeed_config(dist: DistributedConfig) -> dict[str, Any]:
    """DeepSpeed ZeRO config JSON for a DeepSpeed run.

    ``auto`` values are resolved by the DeepSpeed/Transformers integration from
    the realized TrainingConfig (batch size, grad accumulation, precision), so
    the recorded config never contradicts the common trainer knobs.
    """
    if dist.backend != "deepspeed":
        raise ValueError("build_deepspeed_config requires the deepspeed backend")
    zero: dict[str, Any] = {"stage": dist.zero_stage}
    if dist.offload_optimizer:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if dist.offload_params:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    return {
        "zero_optimization": zero,
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "activation_checkpointing": {
            "partition_activations": dist.activation_checkpointing,
        },
    }


def build_launch_command(
    dist: DistributedConfig,
    *,
    train_script: str = "train.py",
    rdzv_id: str = "zevo",
    rdzv_endpoint: str = "",
    extra_script_args: list[str] | None = None,
) -> DistributedLaunchPlan:
    """Build the launcher command for one distributed topology.

    - ``single_gpu`` -> plain ``python <train_script>`` (byte-for-byte the
      existing single-GPU path).
    - ``ddp``/``fsdp`` -> ``torchrun`` (``--standalone`` on one node; c10d
      rendezvous across nodes). FSDP runs also carry an ``fsdp_config``.
    - ``deepspeed`` -> the ``deepspeed`` launcher plus a ZeRO ``deepspeed_config``.

    ``rdzv_endpoint`` (``host:port``) is the runtime rank-0 address for
    multi-node launches; a shell placeholder is emitted when it is not yet
    known so the sbatch author fills it from the allocation.
    """
    if not train_script.strip():
        raise ValueError("train_script must be a non-empty path")
    extra = list(extra_script_args or [])
    nnodes = dist.nodes
    nproc = dist.gpus_per_node

    if dist.backend == "single_gpu":
        return DistributedLaunchPlan(
            launcher="python",
            command=["python", train_script, *extra],
            nnodes=1,
            nproc_per_node=1,
        )

    if dist.backend == "deepspeed":
        command = ["deepspeed", f"--num_nodes={nnodes}", f"--num_gpus={nproc}"]
        if nnodes > 1:
            command.append("--hostfile=hostfile.txt")
            host, sep, port = (rdzv_endpoint or "").partition(":")
            if host:
                command.append(f"--master_addr={host}")
            if sep and port:
                command.append(f"--master_port={port}")
        command += [train_script, "--deepspeed", "deepspeed_config.json", *extra]
        return DistributedLaunchPlan(
            launcher="deepspeed",
            command=command,
            nnodes=nnodes,
            nproc_per_node=nproc,
            deepspeed_config=build_deepspeed_config(dist),
        )

    # ddp or fsdp -> torchrun
    command = ["torchrun"]
    if nnodes == 1:
        command += ["--standalone", "--nnodes=1", f"--nproc_per_node={nproc}"]
    else:
        command += [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}",
            "--rdzv_backend=c10d",
            f"--rdzv_id={rdzv_id}",
            f"--rdzv_endpoint={rdzv_endpoint or '$MASTER_ADDR:$MASTER_PORT'}",
        ]
    command += [train_script, *extra]
    return DistributedLaunchPlan(
        launcher="torchrun",
        command=command,
        nnodes=nnodes,
        nproc_per_node=nproc,
        fsdp_config=build_fsdp_config(dist) if dist.backend == "fsdp" else None,
    )


class TrainingConfig(BaseModel):
    """Complete common trainer configuration selected for one iteration.

    Method/loss semantics live in their own closed namespaces.  This model owns
    the concrete runtime knobs shared across methods; ``implementation_config``
    records any additional framework argument that actually reaches the trainer
    so an implementation default can never be invisible.
    """

    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(ge=1)
    max_seq_len: int = Field(ge=1)
    batch_size: int = Field(ge=1, description="Per-device training batch size")
    gradient_accumulation_steps: int = Field(ge=1)
    world_size: int = Field(ge=1)
    effective_batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    optimizer: str = Field(min_length=1)
    lr_scheduler_type: str = Field(min_length=1)
    warmup_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    weight_decay: float = Field(ge=0, allow_inf_nan=False)
    max_grad_norm: float = Field(gt=0, allow_inf_nan=False)
    precision: Literal["fp32", "fp16", "bf16"]
    distributed_strategy: str = Field(
        min_length=1,
        description=(
            "Free-text strategy label retained for backward compatibility "
            "(e.g. 'single_gpu'). The typed `distributed` sub-config is the "
            "authoritative parallelism plan when present; omit it for the "
            "unchanged single-GPU path."
        ),
    )
    distributed: DistributedConfig | None = Field(
        default=None,
        description=(
            "Typed multi-GPU / multi-node parallelism plan. Omit (null) for the "
            "single-GPU default; when set, its world_size (nodes * gpus_per_node) "
            "must equal the realized world_size."
        ),
    )
    gradient_checkpointing: bool
    packing: bool
    logging_steps: Literal[20]
    eval_strategy: Literal["steps"]
    eval_steps: int = Field(ge=1)
    checkpoint_retention: CheckpointRetentionConfig
    seed: int = Field(ge=0)
    lora_r: int = Field(ge=0)
    lora_alpha: int = Field(ge=0)
    lora_dropout: float = Field(ge=0, lt=1, allow_inf_nan=False)
    lora_target_modules: list[str]
    implementation_config: dict[str, Any]
    software_versions: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_realized_batch_and_peft(self) -> "TrainingConfig":
        expected = self.batch_size * self.gradient_accumulation_steps * self.world_size
        if self.effective_batch_size != expected:
            raise ValueError(
                "effective_batch_size must equal batch_size * "
                "gradient_accumulation_steps * world_size"
            )
        if self.distributed is not None and (
            self.distributed.world_size != self.world_size
        ):
            raise ValueError(
                "distributed.world_size (nodes * gpus_per_node) must equal the "
                "realized world_size"
            )
        if self.lora_r == 0:
            if self.lora_alpha != 0 or self.lora_dropout != 0 or self.lora_target_modules:
                raise ValueError(
                    "non-PEFT training requires zero/empty LoRA configuration"
                )
        elif self.lora_alpha <= 0 or not self.lora_target_modules:
            raise ValueError(
                "PEFT training requires positive lora_alpha and lora_target_modules"
            )
        retention = self.checkpoint_retention
        if retention.strategy != "none":
            realized_save_only_model = self.implementation_config.get(
                "save_only_model"
            )
            if type(realized_save_only_model) is not bool:
                raise ValueError(
                    "checkpoint retention requires implementation_config."
                    "save_only_model to be an explicit boolean; use false when "
                    "full Trainer state is required for exact continuation"
                )
            expected_save_limit = retention.max_intermediate_checkpoints + 1
            realized_save_limit = self.implementation_config.get("save_total_limit")
            if (
                type(realized_save_limit) is not int
                or realized_save_limit != expected_save_limit
            ):
                raise ValueError(
                    "checkpoint retention requires implementation_config."
                    f"save_total_limit={expected_save_limit} "
                    "(max_intermediate_checkpoints + one temporary terminal "
                    "checkpoint slot)"
                )
        if any(not key.strip() or not value.strip() for key, value in self.software_versions.items()):
            raise ValueError("software_versions keys and values must be non-empty")
        return self


class TrainRunConfig(BaseModel):
    """Complete configuration actually used by one Train iteration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[7] = 7
    iteration: int = Field(ge=1)
    parent_model: str = Field(min_length=1)
    parent_kind: Literal["baseline", "run_checkpoint"]
    parent_selection_rationale: str = Field(min_length=1)
    data_signature: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Exact bound Data signature: 64 lowercase hexadecimal SHA-256 "
            "characters with no prefix. Copy it from the Train work order."
        ),
    )
    training_method: str = Field(min_length=1)
    method_config: dict[str, Any] = Field(default_factory=dict)
    loss_contract: LossContract
    training: TrainingConfig
    prompt: PromptContract
    tokenizer_source: str = Field(min_length=1)
    chat_template_source: str = ""
    chat_template_hash: str = Field(
        "",
        max_length=64,
        pattern=r"^(?:|[0-9a-f]{64})$",
        description=(
            "Copy inference_config.yaml chat_template_hash exactly; empty when "
            "no chat template applies, otherwise 64 lowercase hexadecimal "
            "SHA-256 characters with no prefix."
        ),
    )
    template_kwargs: dict[str, JsonValue] = Field(
        ...,
        description=(
            "Exact template_kwargs copied from baseline inference_config.yaml. "
            "Train must load and pass this mapping unchanged; it must not infer or "
            "hard-code model-specific template controls."
        ),
    )
    special_token_ids: dict[str, int | None] = Field(default_factory=dict)
    generation_backend: Literal["hf", "vllm"]
    inference_config_path: str = Field(min_length=1)
    inference_config_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Bare SHA-256 of the exact inference_config.yaml file bytes: 64 "
            "lowercase hexadecimal characters with no prefix."
        ),
    )
    prompt_alignment: PromptAlignmentEvidence
    training_data_example: TrainingDataExample
    direction: str = Field(min_length=1)
    method_diversity_status: Literal[
        "initial", "varied", "retained_in_branch",
        "retained_for_constraints", "user_pinned"
    ]
    method_selection_rationale: str = Field(min_length=1)
    suggestion_decisions: list[SuggestionDecision] = Field(default_factory=list)

    @field_validator("template_kwargs")
    @classmethod
    def validate_template_kwargs(
        cls, value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        reserved = {
            "tokenize", "add_generation_prompt", "continue_final_message",
            "return_dict", "return_tensors",
        }
        invalid = sorted(
            key for key in value
            if not key.strip() or key in reserved
        )
        if invalid:
            raise ValueError(
                "template_kwargs contains empty or execution-control keys: "
                + ", ".join(invalid)
            )
        return value

    @model_validator(mode="after")
    def validate_method_and_iteration(self) -> "TrainRunConfig":
        if not is_sha256(self.data_signature):
            raise ValueError("data_signature must be lowercase hexadecimal SHA-256")
        if not is_sha256(self.inference_config_sha256):
            raise ValueError(
                "inference_config_sha256 must be lowercase hexadecimal SHA-256"
            )
        self.training_method = self.training_method.strip().lower()
        self.method_config = normalize_method_config(self.method_config)
        errors = method_config_errors(
            self.training_method, self.method_config, require_dependencies=True,
            require_complete=True,
        )
        if errors:
            raise ValueError("; ".join(errors))
        uses_peft = self.training_method == "lora_sft" or bool(
            self.method_config.get("use_peft", False)
        )
        if uses_peft != (self.training.lora_r > 0):
            raise ValueError(
                "training LoRA configuration must match the selected method_config.use_peft"
            )
        required_packages = {"torch", "transformers", "trl"}
        if uses_peft:
            required_packages.add("peft")
        missing_packages = sorted(
            required_packages - set(self.training.software_versions)
        )
        if missing_packages:
            raise ValueError(
                "training.software_versions is incomplete; missing executed "
                f"packages: {missing_packages}"
            )
        expected_loss = derive_loss_contract(
            self.training_method, self.prompt.prompt_framing,
        )
        if (
            self.loss_contract.objective != expected_loss.objective
            or self.loss_contract.target_scope != expected_loss.target_scope
        ):
            raise ValueError(
                f"loss_contract does not match training_method={self.training_method!r}"
            )
        self.loss_contract.objective_config = validate_loss_objective_config(
            self.training_method, self.loss_contract.objective_config,
            require_complete=True,
        )
        if self.iteration == 1 and self.parent_kind != "baseline":
            raise ValueError("iteration 1 requires parent_kind='baseline'")
        if (
            self.prompt.prompt_framing == "chat"
            or self.prompt.prompt_framing.startswith("chat:")
        ) and not (self.chat_template_source and self.chat_template_hash):
            raise ValueError(
                "chat training config requires copied chat-template source and hash"
            )
        if self.chat_template_hash and not is_sha256(self.chat_template_hash):
            raise ValueError("chat_template_hash must be lowercase hexadecimal SHA-256")
        if not (
            self.prompt.prompt_framing == "chat"
            or self.prompt.prompt_framing.startswith("chat:")
        ) and self.template_kwargs:
            raise ValueError("non-chat training config requires empty template_kwargs")
        duplicate_template_keys = sorted(
            set(self.template_kwargs) & set(self.training.implementation_config)
        )
        if duplicate_template_keys:
            raise ValueError(
                "template kwargs must not be duplicated in "
                "training.implementation_config: "
                + ", ".join(duplicate_template_keys)
            )
        if "enable_thinking" in self.template_kwargs:
            enable_thinking = self.template_kwargs["enable_thinking"]
            if not isinstance(enable_thinking, bool):
                raise ValueError("template_kwargs.enable_thinking must be boolean")
            if enable_thinking is not True:
                raise ValueError(
                    "non-thinking models omit template_kwargs.enable_thinking"
                )
            if self.prompt.model_reasoning_type != "thinking":
                raise ValueError(
                    "template_kwargs.enable_thinking contradicts the selected "
                    "model reasoning type"
                )
        rendered_values = [
            self.prompt_alignment.rendered_prompt,
            *self.training_data_example.rendered_sequences.values(),
        ]
        if any(
            not block.strip()
            for rendered in rendered_values
            for block in _think_blocks(rendered)
        ):
            raise ValueError("thinking-model training rendering requires reasoning content")
        if self.prompt.model_reasoning_type == "non_thinking" and any(
            _contains_think_tag(rendered) for rendered in rendered_values
        ):
            raise ValueError("non-thinking model training rendering must not contain think tags")
        if self.prompt.model_reasoning_type == "thinking" and not any(
            any(block.strip() for block in _think_blocks(rendered))
            for rendered in self.training_data_example.rendered_sequences.values()
        ):
            raise ValueError(
                "thinking model requires reasoning content in "
                "training_data_example"
            )
        example_summary = self.training_data_example.loss_target_summary
        if (
            self.loss_contract.objective not in example_summary
            or f"target_scope={self.loss_contract.target_scope}" not in example_summary
        ):
            raise ValueError(
                "training_data_example.loss_target_summary must name the exact "
                "loss objective and target_scope"
            )
        if (
            self.prompt.prompt_framing == "chat"
            or self.prompt.prompt_framing.startswith("chat:")
        ) and not any(
            self.prompt.system_prompt in sequence
            for sequence in self.training_data_example.rendered_sequences.values()
        ):
            raise ValueError(
                "training_data_example rendered sequences omit the realized system prompt"
            )
        if len({d.key for d in self.suggestion_decisions}) != len(self.suggestion_decisions):
            raise ValueError("suggestion_decisions keys must be unique")
        return self


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yaml_config(path: str | Path, model: type[BaseModel]) -> BaseModel:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"configuration YAML does not exist: {source}")
    try:
        body = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read configuration YAML {source}: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"configuration YAML must contain one mapping: {source}")
    return model.model_validate(body)


def load_inference_config(path: str | Path) -> InferenceRunConfig:
    return InferenceRunConfig.model_validate(
        load_yaml_config(path, InferenceRunConfig).model_dump()
    )


def load_train_config(path: str | Path) -> TrainRunConfig:
    return TrainRunConfig.model_validate(load_yaml_config(path, TrainRunConfig).model_dump())


class VllmMemoryPlan(BaseModel):
    """Auditable absolute-memory target reused across one model lineage."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["model_and_workload_sized"]
    runtime_weight_gib: float = Field(gt=0, allow_inf_nan=False)
    peak_live_tokens: int = Field(ge=1)
    kv_bytes_per_token: int = Field(ge=1)
    kv_cache_gib: float = Field(gt=0, allow_inf_nan=False)
    tensor_parallel_size: int = Field(ge=1)
    target_gpu_memory_gib: int = Field(ge=1)
    utilization_step: Literal[0.05]
    min_utilization: Literal[0.1]
    max_utilization: Literal[0.9]
    free_memory_margin_gib: Literal[2.0]

    @model_validator(mode="after")
    def validate_derived_sizes(self) -> "VllmMemoryPlan":
        expected_kv = (
            self.peak_live_tokens * self.kv_bytes_per_token / (1024 ** 3)
        )
        if not math.isclose(
            self.kv_cache_gib, expected_kv, rel_tol=0.01, abs_tol=0.05,
        ):
            raise ValueError(
                "kv_cache_gib must match peak_live_tokens * "
                "kv_bytes_per_token"
            )
        expected_target = math.ceil(
            (
                self.runtime_weight_gib * 1.2
                + self.kv_cache_gib
            ) / self.tensor_parallel_size
            + 4.0
        )
        if self.target_gpu_memory_gib != expected_target:
            raise ValueError(
                "target_gpu_memory_gib must equal the engine-owned model and "
                "workload sizing formula"
            )
        return self


def validate_adaptive_vllm_memory_config(config: InferenceRunConfig) -> None:
    """Require an absolute vLLM memory plan instead of a fixed GPU fraction."""
    if config.generation_backend != "vllm":
        return
    implementation = config.implementation_config
    llm_kwargs = implementation.get("llm_kwargs")
    if not isinstance(llm_kwargs, dict):
        raise ValueError("vLLM inference requires implementation_config.llm_kwargs")
    if "gpu_memory_utilization" in llm_kwargs:
        raise ValueError(
            "gpu_memory_utilization is device-derived; record memory_plan "
            "instead of freezing a GPU fraction in llm_kwargs"
        )
    try:
        plan = VllmMemoryPlan.model_validate(implementation.get("memory_plan"))
    except Exception as exc:
        raise ValueError(f"invalid implementation_config.memory_plan: {exc}") from exc
    realized_tp = llm_kwargs.get("tensor_parallel_size")
    if type(realized_tp) is not int or realized_tp != plan.tensor_parallel_size:
        raise ValueError(
            "memory_plan.tensor_parallel_size must match "
            "implementation_config.llm_kwargs.tensor_parallel_size"
        )


def validate_cluster_train_config(config: TrainRunConfig) -> None:
    """Reject DataLoader subprocesses for distributed Cluster training.

    A Slurm Train job already uses one distributed process per GPU. Spawning
    additional DataLoader workers from every rank creates multiprocessing
    scratch state (notably ``pymp-*`` directories) whose teardown is unreliable
    on shared filesystems. Cluster jobs therefore load batches in their rank
    process and must realize this implementation value explicitly.
    """
    workers = config.training.implementation_config.get(
        "dataloader_num_workers"
    )
    if type(workers) is not int or workers != 0:
        raise ValueError(
            "cluster Train requires training.implementation_config."
            "dataloader_num_workers=0; the value must be explicit so each DDP "
            "rank does not create a nested multiprocessing worker pool"
        )


_CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "inference": InferenceRunConfig,
    "train": TrainRunConfig,
}


def _main(argv: list[str] | None = None) -> int:
    """Expose the exact runtime schemas and a side-effect-free validator.

    Agents receive the JSON schema inline, but this command lets generated
    files be checked before any remote command or GPU work begins.  It never
    canonicalizes or rewrites the file: a successful validation means the
    artifact already has the one accepted shape.
    """
    parser = ArgumentParser(prog="python -m zevo.contracts.configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("kind", choices=sorted(_CONFIG_MODELS))
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("kind", choices=sorted(_CONFIG_MODELS))
    validate_parser.add_argument("path")
    validate_parser.add_argument(
        "--cluster",
        action="store_true",
        help="also enforce Cluster-specific runtime constraints",
    )
    validate_parser.add_argument(
        "--adaptive-vllm-memory",
        action="store_true",
        help="require model/workload-sized vLLM memory planning",
    )
    args = parser.parse_args(argv)
    model = _CONFIG_MODELS[args.kind]
    if args.command == "schema":
        print(json.dumps(model.model_json_schema(), indent=2, sort_keys=True))
        return 0
    try:
        config = load_yaml_config(args.path, model)
        if args.cluster:
            if args.kind != "train":
                raise ValueError("--cluster is supported only for Train configuration")
            validate_cluster_train_config(TrainRunConfig.model_validate(config))
        if args.adaptive_vllm_memory:
            if args.kind != "inference":
                raise ValueError(
                    "--adaptive-vllm-memory is supported only for Inference "
                    "configuration"
                )
            validate_adaptive_vllm_memory_config(
                InferenceRunConfig.model_validate(config)
            )
    except Exception as exc:
        print(f"INVALID {model.__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"VALID {model.__name__}: {Path(args.path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
