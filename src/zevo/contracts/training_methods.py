"""Shared validation for method-specific Train configuration.

``method_config`` is authored by Train unless the Run/Setting pins values, and
is recorded in the Train YAML for the executed iteration. Auxiliary models are
deliberately references, not artifacts: this release supports Hugging Face
repository ids only and does not accept local paths, uploads, URLs, or Registry
models.
"""
from __future__ import annotations

import re
from typing import Any


AUXILIARY_MODEL_FIELD_BY_METHOD: dict[str, str] = {
    "gkd": "teacher_model",
    "online_dpo": "reward_model",
}

_AUXILIARY_MODEL_FIELDS = frozenset(AUXILIARY_MODEL_FIELD_BY_METHOD.values())

METHOD_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "lora_sft": frozenset(),
    "full_sft": frozenset(),
    "dpo": frozenset({"use_peft"}),
    "cpo": frozenset({"use_peft"}),
    "gkd": frozenset({"use_peft", "teacher_model"}),
    "grpo": frozenset({"use_peft"}),
    "kto": frozenset({"use_peft"}),
    "online_dpo": frozenset({"use_peft", "reward_model"}),
    "orpo": frozenset({"use_peft"}),
    "rft": frozenset({"use_peft"}),
    "rloo": frozenset({"use_peft"}),
}

# Supervised-finetuning family: teaches format and a first pass at reasoning.
# A reinforcement/verifiable-reward lever is a deliberate next step *after* one
# of these has established a reasonable baseline (see the orchestrator's
# reinforcement-progression policy in zevo.engine.method.loop_policy).
SFT_METHODS = frozenset({"lora_sft", "full_sft"})

# The verifiable-reward progression a SOTA practitioner runs on a task with a
# deterministic correctness check, IN ORDER, once an SFT baseline exists:
# SFT(+distilled) -> RFT (rejection-sampling bridge) -> GRPO (online RLVR).
# Both consume a +1/0 correctness reward derived from the gold answer; RFT is
# the stable bridge and GRPO the terminal lever. Kept ordered so the policy can
# name the single next method rather than a set.
REINFORCEMENT_PROGRESSION = ("rft", "grpo")

# These have one canonical home outside method_config. Prompt and loss fields
# are run contracts; shared hyperparameters are recorded in the Train YAML.
# Accepting a second copy here would let a selected Skill silently choose which
# one wins, recreating the framing/loss drift the contracts exist to prevent.
RESERVED_METHOD_CONFIG_FIELDS = frozenset({
    "base_model", "training_method", "prompt_framing", "model_reasoning_type",
    "system_prompt", "chat_template", "chat_template_path", "resolved_framing",
    "loss_contract", "assistant_only_loss", "completion_only_loss",
    "generation_backend", "num_epochs", "max_seq_len", "batch_size",
    "learning_rate", "lora_r", "lora_alpha",
})
_HF_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
_HF_MODEL_ID = re.compile(rf"^{_HF_COMPONENT}/{_HF_COMPONENT}$")


def is_huggingface_model_id(value: str) -> bool:
    """Return whether *value* is an explicit ``owner/model`` HF repository id."""
    model_id = (value or "").strip()
    return bool(
        len(model_id) <= 96
        and _HF_MODEL_ID.fullmatch(model_id)
        and ".." not in model_id
        and "--" not in model_id
    )


def normalize_method_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Copy config and normalize the two supported auxiliary-model references."""
    normalized = dict(config or {})
    for field in _AUXILIARY_MODEL_FIELDS:
        if field in normalized and isinstance(normalized[field], str):
            normalized[field] = normalized[field].strip()
    return normalized


def method_config_errors(
    training_method: str,
    config: dict[str, Any] | None,
    *,
    require_dependencies: bool = True,
    require_complete: bool = False,
) -> list[str]:
    """Validate cross-field method requirements without performing network I/O.

    The namespace is closed: accepting an unknown key would make the UI/API
    imply a control exists while the selected Skill may silently ignore it.
    """
    method = (training_method or "").strip().lower()
    values = normalize_method_config(config)
    errors: list[str] = []

    if method and method not in METHOD_CONFIG_KEYS:
        errors.append(
            f"unsupported training_method={method!r}; installed methods: "
            + ", ".join(sorted(METHOD_CONFIG_KEYS))
        )

    reserved = sorted(set(values) & RESERVED_METHOD_CONFIG_FIELDS)
    if reserved:
        errors.append(
            "method_config contains fields with another canonical owner: "
            f"{reserved}"
        )

    from zevo.contracts.prompting import LOSS_OBJECTIVE_CONFIG_KEYS
    misplaced_objective = sorted(
        set(values) & set().union(*LOSS_OBJECTIVE_CONFIG_KEYS.values())
    )
    if misplaced_objective:
        errors.append(
            "method_config contains loss-objective fields; move them to "
            f"loss_contract.objective_config: {misplaced_objective}"
        )

    already_named = (
        RESERVED_METHOD_CONFIG_FIELDS
        | _AUXILIARY_MODEL_FIELDS
        | set().union(*LOSS_OBJECTIVE_CONFIG_KEYS.values())
    )
    unknown = sorted(
        set(values) - METHOD_CONFIG_KEYS.get(method, frozenset()) - already_named
    )
    if unknown:
        errors.append(
            f"unsupported method_config keys for {method or '<empty>'}: {unknown}; "
            f"allowed: {sorted(METHOD_CONFIG_KEYS.get(method, frozenset()))}"
        )

    if "use_peft" in values and not isinstance(values["use_peft"], bool):
        errors.append("method_config.use_peft must be a boolean")

    if require_complete and method and "use_peft" in METHOD_CONFIG_KEYS.get(method, frozenset()) \
            and "use_peft" not in values:
        errors.append(
            f"training_method={method} requires explicit method_config.use_peft"
        )

    if values and not method:
        errors.append("method_config requires a selected training_method")

    expected = AUXILIARY_MODEL_FIELD_BY_METHOD.get(method)
    for field in _AUXILIARY_MODEL_FIELDS:
        if field not in values:
            continue
        value = values[field]
        if field != expected:
            errors.append(f"{field} is not valid for training_method={method or '<empty>'}")
            continue
        if not isinstance(value, str) or not is_huggingface_model_id(value):
            errors.append(
                f"method_config.{field} must be a Hugging Face model id in "
                "owner/model form; local paths, URLs, uploads, and Registry "
                "references are not supported"
            )

    if require_dependencies and expected and not str(values.get(expected) or "").strip():
        errors.append(
            f"training_method={method} requires method_config.{expected} as a "
            "Hugging Face owner/model id"
        )
    return errors
