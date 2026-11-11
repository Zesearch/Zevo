"""Explicit method decisions, with read compatibility for historical Tickets."""
from typing import Any

from zevo.contracts.training_methods import METHOD_CONFIG_KEYS


def normalize_method_selection(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    suggestions = dict(result.get("configuration_suggestions") or {})
    legacy = suggestions.pop("training_method", "")
    selected = result.get("selected_training_method", "")
    for item in (legacy, selected):
        if not isinstance(item, str):
            raise ValueError("selected_training_method must be a string")
    legacy, selected = legacy.strip().lower(), selected.strip().lower()
    if legacy and selected and legacy != selected:
        raise ValueError("conflicting legacy and explicit training method selections")
    selected = selected or legacy
    if selected and selected not in METHOD_CONFIG_KEYS:
        raise ValueError(f"unsupported selected_training_method={selected!r}")
    result["selected_training_method"] = selected
    if "configuration_suggestions" in result:
        result["configuration_suggestions"] = suggestions
    return result


def get_selected_training_method(value: Any) -> str:
    """Read old raw payloads as well as normalized typed work orders."""
    if not isinstance(value, dict):
        value = value.model_dump()
    return normalize_method_selection(value)["selected_training_method"]
