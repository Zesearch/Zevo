"""Prompt, inference, and iteration-loss value models.

Prompt and measurement semantics are selected by baseline Inference. Loss
semantics belong to one executed Train iteration because method and data are
searchable decisions.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

ModelReasoningType = Literal["thinking", "non_thinking"]
LossTargetScope = Literal[
    "assistant_messages",
    "completion",
    "all_tokens",
    "method_defined",
    "not_applicable",
]


class LossContract(BaseModel):
    """The iteration's semantic loss that its selected Train Skill implements."""

    model_config = ConfigDict(extra="forbid")

    owner: Literal["train_skill"] = "train_skill"
    framework: Literal["trl"] = "trl"
    objective: str = Field(min_length=1)
    target_scope: LossTargetScope
    objective_config: dict[str, Any] = Field(default_factory=dict)


class PromptContract(BaseModel):
    """The exact prompt semantics shared by Data, Train, and Inference."""

    model_config = ConfigDict(extra="forbid")

    prompt_framing: str = Field(
        min_length=1,
        pattern=r"^(?:chat|chat:[^\s]+|completion|text)$",
        description=(
            "Exactly 'chat', 'chat:<template-model>' with no whitespace, "
            "'completion', or 'text'. The model reasoning type records the "
            "selected model's verified template class."
        ),
    )
    model_reasoning_type: ModelReasoningType = "non_thinking"
    system_prompt: str = ""

    @model_validator(mode="after")
    def canonicalize_prompt_values(self) -> "PromptContract":
        # Canonicalize before writing the YAML execution configuration. This
        # makes an empty chat system prompt and DEFAULT_SYSTEM_PROMPT one
        # logical value throughout Train and Inference.
        self.prompt_framing, self.model_reasoning_type, self.system_prompt = (
            canonical_prompt_contract(
                prompt_framing=self.prompt_framing,
                model_reasoning_type=self.model_reasoning_type,
                system_prompt=self.system_prompt,
            )
        )
        return self


class InferenceContract(BaseModel):
    """Task mapping and decoding values fixed before the baseline probe."""

    model_config = ConfigDict(extra="forbid")

    inference_config: dict[str, Any] = Field(default_factory=dict)
    decoding_strategy: Literal["greedy", "sampling"] = "greedy"
    max_new_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(1.0, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(0, ge=0)
    repetition_penalty: float = Field(1.0, gt=0, allow_inf_nan=False)
    seed: int = Field(0, ge=0)


DECODING_CONFIG_KEYS = frozenset({
    "decoding_strategy", "max_new_tokens", "temperature", "top_p", "top_k",
    "repetition_penalty", "seed",
})


def validate_decoding_config(value: dict[str, Any] | None) -> dict[str, Any]:
    """Validate user-pinned decoding fields without inventing omitted values."""
    config = dict(value or {})
    unknown = sorted(set(config) - DECODING_CONFIG_KEYS)
    if unknown:
        raise ValueError("unsupported decoding_config keys: " + ", ".join(unknown))
    strategy = config.get("decoding_strategy")
    temperature = config.get("temperature")
    if strategy == "greedy" and temperature is not None and float(temperature) != 0:
        raise ValueError("decoding_strategy=greedy requires temperature=0")
    if strategy == "sampling" and temperature is not None and float(temperature) <= 0:
        raise ValueError("decoding_strategy=sampling requires temperature>0")
    # Reuse the complete contract model for value/range validation, then retain
    # only keys the user actually pinned.
    validated = InferenceContract.model_validate({
        **config,
    }).model_dump()
    return {key: validated[key] for key in config}


LOSS_OBJECTIVE_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "lora_sft": frozenset({"loss_type"}),
    "full_sft": frozenset({"loss_type"}),
    "dpo": frozenset({
        "loss_type", "loss_weights", "beta", "label_smoothing",
        "f_divergence_type", "reference_free",
    }),
    "cpo": frozenset({
        "loss_type", "beta", "cpo_alpha", "simpo_gamma", "alpha",
    }),
    "gkd": frozenset({
        "lmbda", "beta", "seq_kd", "temperature", "max_new_tokens",
    }),
    "grpo": frozenset({
        "loss_type", "importance_sampling_level", "beta", "num_generations",
        "mask_truncated_completions", "clip_ratio", "clip_ratio_high",
        "reward_weights", "scale_rewards", "temperature", "top_p", "top_k",
        "max_completion_length", "num_iterations",
    }),
    "kto": frozenset({
        "loss_type", "beta", "desirable_weight", "undesirable_weight",
        "train_sampling_strategy",
    }),
    "online_dpo": frozenset({
        "loss_type", "beta", "reward_weights", "temperature", "top_p",
        "top_k", "max_new_tokens", "missing_eos_penalty",
    }),
    "orpo": frozenset({"beta"}),
    "rft": frozenset({
        "num_samples", "temperature", "top_p", "top_k", "answer_parser",
        "max_survivors_per_prompt", "loss_type",
    }),
    "rloo": frozenset({
        "beta", "num_generations", "normalize_advantages", "cliprange",
        "reward_clip", "reward_weights", "mask_truncated_completions",
        "temperature", "top_p", "top_k", "max_completion_length",
        "num_iterations",
    }),
}

# Recommended explicit Train starting values. Train may choose different
# values with rationale, but train_config.yaml must contain the complete
# realized mapping. Only controls that affect
# the mathematical objective or rollout/filtering distribution live here;
# irrelevant optional controls remain absent.
LOSS_OBJECTIVE_RECOMMENDED_STARTS: dict[str, dict[str, Any]] = {
    "lora_sft": {"loss_type": "nll"},
    "full_sft": {"loss_type": "nll"},
    "dpo": {
        "loss_type": "sigmoid", "beta": 0.1, "label_smoothing": 0.0,
        "reference_free": False,
    },
    "cpo": {"loss_type": "sigmoid", "beta": 0.1, "cpo_alpha": 1.0},
    "gkd": {
        "lmbda": 0.5, "beta": 0.5, "seq_kd": False,
        "temperature": 1.0, "max_new_tokens": 256,
    },
    "grpo": {
        "loss_type": "dapo", "importance_sampling_level": "token",
        "beta": 0.0, "num_generations": 4,
        "mask_truncated_completions": True, "clip_ratio": 0.2,
        "scale_rewards": "group", "temperature": 0.9, "top_p": 1.0,
        "top_k": 0, "max_completion_length": 256, "num_iterations": 1,
    },
    "kto": {
        "loss_type": "kto", "beta": 0.1, "desirable_weight": 1.0,
        "undesirable_weight": 1.0, "train_sampling_strategy": "sequential",
    },
    "online_dpo": {
        "loss_type": "sigmoid", "beta": 0.1, "temperature": 0.9,
        "top_p": 1.0, "top_k": 0, "max_new_tokens": 64,
        "missing_eos_penalty": 0.0,
    },
    "orpo": {"beta": 0.1},
    "rft": {
        "num_samples": 4, "temperature": 0.8, "top_p": 1.0,
        "top_k": 0, "answer_parser": "exact", "max_survivors_per_prompt": 1,
        "loss_type": "nll",
    },
    "rloo": {
        "beta": 0.05, "num_generations": 4,
        "normalize_advantages": True, "cliprange": 0.2,
        "mask_truncated_completions": True, "temperature": 0.9,
        "top_p": 1.0, "top_k": 0, "max_completion_length": 256,
        "num_iterations": 1,
    },
}


def validate_loss_objective_config(
    training_method: object, value: dict[str, Any] | None, *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Keep method-specific loss semantics in one explicit namespace."""
    method = training_method.strip().lower() if isinstance(training_method, str) else ""
    config = dict(value or {})
    allowed = LOSS_OBJECTIVE_CONFIG_KEYS.get(method, frozenset())
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported loss_contract.objective_config keys for {method or '<empty>'}: "
            f"{unknown}; allowed: {sorted(allowed)}"
        )
    if require_complete:
        required = set(LOSS_OBJECTIVE_RECOMMENDED_STARTS.get(method, {}))
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(
                f"loss_contract.objective_config for {method or '<empty>'} is incomplete; "
                f"missing required realized keys: {missing}"
            )
    return config


def recommended_loss_objective_config(
    training_method: object, value: dict[str, Any] | None,
) -> dict[str, Any]:
    """Complete starting guidance for validation and deterministic stubs."""
    method = training_method.strip().lower() if isinstance(training_method, str) else ""
    supplied = dict(value or {})
    completed = {**LOSS_OBJECTIVE_RECOMMENDED_STARTS.get(method, {}), **supplied}
    return validate_loss_objective_config(method, completed, require_complete=True)


def normalize_prompt_framing(value: object, *, allow_empty: bool = False) -> str:
    """Return one canonical framing or raise on obsolete/ambiguous syntax."""
    framing = value.strip() if isinstance(value, str) else ""
    if not framing:
        if allow_empty:
            return ""
        raise ValueError("prompt_framing must be non-empty")
    if "+think" in framing:
        raise ValueError(
            "prompt_framing no longer carries '+think'; classify the selected "
            "model under prompt.model_reasoning_type"
        )
    if framing in {"chat", "completion", "text"}:
        return framing
    if (
        framing.startswith("chat:")
        and framing[5:].strip() == framing[5:]
        and framing[5:]
        and not any(char.isspace() for char in framing[5:])
    ):
        return framing
    raise ValueError(
        "prompt_framing must be 'chat', 'chat:<template-model>', "
        "'completion', or 'text'"
    )


def is_chat_framing(framing: str) -> bool:
    return framing == "chat" or framing.startswith("chat:")


def canonical_prompt_contract(
    *,
    prompt_framing: object,
    model_reasoning_type: ModelReasoningType,
    system_prompt: object,
    allow_empty: bool = False,
) -> tuple[str, ModelReasoningType, str]:
    """Canonicalize framing around the selected model's reasoning class."""
    framing = normalize_prompt_framing(prompt_framing, allow_empty=allow_empty)
    system = system_prompt.strip() if isinstance(system_prompt, str) else ""
    if not framing:
        if model_reasoning_type != "non_thinking" or system:
            raise ValueError(
                "an empty prompt_framing requires a non-thinking model and no system_prompt"
            )
        return "", model_reasoning_type, ""
    if is_chat_framing(framing):
        return framing, model_reasoning_type, system or DEFAULT_SYSTEM_PROMPT
    if model_reasoning_type != "non_thinking":
        raise ValueError("a thinking model requires a chat prompt_framing")
    if system:
        raise ValueError("system_prompt is only valid with a chat prompt_framing")
    return framing, model_reasoning_type, ""


def derive_loss_contract(training_method: object, prompt_framing: object) -> LossContract:
    """Derive the immutable semantic loss after METHOD and framing are fixed.

    TRL and the selected Train Skill own the concrete implementation.  Only SFT
    exposes a generic token target scope; other methods define their target and
    objective in their own Skill and must not be guessed from prompt framing.
    """
    method = training_method.strip().lower() if isinstance(training_method, str) else ""
    framing = normalize_prompt_framing(prompt_framing, allow_empty=not method)
    if not method:
        return LossContract(objective="not_applicable", target_scope="not_applicable")
    if method in {"lora_sft", "full_sft"}:
        if is_chat_framing(framing):
            scope: LossTargetScope = "assistant_messages"
        elif framing == "completion":
            scope = "completion"
        elif framing == "text":
            scope = "all_tokens"
        else:  # pragma: no cover - normalize_prompt_framing owns this boundary
            raise ValueError(f"unsupported SFT prompt_framing {framing!r}")
        return LossContract(objective="supervised_causal_lm", target_scope=scope)
    return LossContract(objective=method, target_scope="method_defined")


def require_loss_contract(
    declared: LossContract | dict | None,
    *,
    training_method: object,
    prompt_framing: object,
) -> LossContract:
    """Materialize the derived contract and reject a contradictory declaration."""
    expected = derive_loss_contract(training_method, prompt_framing)
    if declared is None:
        return expected
    actual = (
        declared if isinstance(declared, LossContract)
        else LossContract.model_validate(declared)
    )
    actual.objective_config = validate_loss_objective_config(
        training_method, actual.objective_config,
    )
    if (
        actual.owner != expected.owner
        or actual.framework != expected.framework
        or actual.objective != expected.objective
        or actual.target_scope != expected.target_scope
    ):
        raise ValueError(
            "loss_contract conflicts with training_method/prompt_framing; "
            f"expected {expected.model_dump()}, got {actual.model_dump()}"
        )
    return actual


def validate_inference_config(
    value: dict[str, Any] | None, *, require_complete: bool = False,
) -> dict[str, Any]:
    """Validate the small task-dependent inference namespace once."""
    config = dict(value or {})
    supported = {
        "input_fields", "answer_regex", "answer_column", "batch_size", "stop",
        # Task-owned semantic protocol.  Unlike prompt_framing/chat-template
        # identity, these values describe what one evaluation row asks and how
        # the generated response should be represented.
        "task_instruction", "user_prompt_template", "output_instruction",
        "response_format", "answer_parser",
        # Opt-in multiple-choice option-scoring mode. Default (key absent) is
        # unchanged free generation. When set to "option_loglikelihood",
        # Inference scores each answer option under the frozen prompt and writes
        # the per-option log-likelihoods into the prediction column, which the
        # deterministic `mc_loglikelihood` metric then argmaxes. `option_fields`
        # names the columns carrying the answer options.
        "scoring_mode", "option_fields",
    }
    unknown = sorted(set(config) - supported)
    if unknown:
        raise ValueError("unsupported inference_config keys: " + ", ".join(unknown))
    for key in (
        "answer_regex", "answer_column", "task_instruction",
        "user_prompt_template", "output_instruction", "response_format",
        "answer_parser",
    ):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"inference_config.{key} must be a string")
    for key in ("task_instruction", "user_prompt_template", "output_instruction"):
        if key in config and not config[key].strip():
            raise ValueError(f"inference_config.{key} must not be blank")
    if "response_format" in config and config["response_format"] not in {
        "plain_text", "label", "choice", "boxed_answer", "code",
    }:
        raise ValueError("inference_config.response_format is not supported")
    if "answer_parser" in config and config["answer_parser"] not in {
        "raw", "choice", "boxed", "regex", "code",
    }:
        raise ValueError("inference_config.answer_parser is not supported")
    semantic_keys = {
        "task_instruction", "user_prompt_template", "output_instruction",
        "response_format", "answer_parser",
    }
    present_semantics = semantic_keys & set(config)
    if present_semantics and present_semantics != semantic_keys:
        missing = sorted(semantic_keys - present_semantics)
        raise ValueError(
            "inference_config task protocol is incomplete; missing: "
            + ", ".join(missing)
        )
    if present_semantics:
        parser = config["answer_parser"]
        response_format = config["response_format"]
        expected_format = {
            "boxed": "boxed_answer",
            "choice": "choice",
            "code": "code",
        }.get(parser)
        if expected_format and response_format != expected_format:
            raise ValueError(
                f"inference_config.answer_parser={parser!r} requires "
                f"response_format={expected_format!r}"
            )
        if parser == "regex" and not str(config.get("answer_regex") or "").strip():
            raise ValueError(
                "inference_config.answer_parser='regex' requires answer_regex"
            )
    for key in ("input_fields", "stop", "option_fields"):
        if key in config and (
            not isinstance(config[key], list)
            or not all(isinstance(item, str) and item for item in config[key])
        ):
            raise ValueError(f"inference_config.{key} must be a list of non-empty strings")
    if "input_fields" in config and len(config["input_fields"]) != len(set(config["input_fields"])):
        raise ValueError("inference_config.input_fields must not contain duplicates")
    if "scoring_mode" in config:
        if config["scoring_mode"] not in ("generate", "option_loglikelihood"):
            raise ValueError(
                "inference_config.scoring_mode must be 'generate' or "
                "'option_loglikelihood'"
            )
        if config["scoring_mode"] == "option_loglikelihood":
            option_fields = config.get("option_fields")
            if not option_fields:
                raise ValueError(
                    "inference_config.scoring_mode='option_loglikelihood' "
                    "requires a non-empty option_fields listing the answer-"
                    "option columns"
                )
            if len(option_fields) != len(set(option_fields)):
                raise ValueError(
                    "inference_config.option_fields must not contain duplicates"
                )
    elif "option_fields" in config:
        raise ValueError(
            "inference_config.option_fields is only valid with "
            "scoring_mode='option_loglikelihood'"
        )
    if "batch_size" in config and (
        isinstance(config["batch_size"], bool)
        or not isinstance(config["batch_size"], int)
        or config["batch_size"] <= 0
    ):
        raise ValueError("inference_config.batch_size must be a positive integer")
    if require_complete:
        missing = sorted({"input_fields", "answer_column"} - set(config))
        if missing:
            raise ValueError(
                "measurement.inference_config is incomplete; missing required "
                f"realized keys: {missing}"
            )
    return config
