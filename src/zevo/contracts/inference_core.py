"""Answer-free model interface shared by Validation and held-out inference."""
from __future__ import annotations

from typing import Any

from zevo.contracts.configuration import InferenceRunConfig


def inference_core(config: InferenceRunConfig) -> dict[str, Any]:
    """Project only model-interface settings; never expose examples or scores."""
    data = config.model_dump(mode="json")
    core = {key: data[key] for key in (
        "base_model", "prompt", "generation_backend", "tokenizer_source",
        "chat_template_source", "chat_template_hash", "template_kwargs",
        "special_token_ids", "stop_token_ids",
    )}
    core["decoding"] = {key: value for key, value in data["measurement"].items()
                        if key not in {"inference_config", "max_new_tokens"}}
    core["stop"] = data["measurement"]["inference_config"].get("stop", [])
    impl = config.implementation_config
    rendering = impl.get("template_rendering") or {}
    core["add_generation_prompt"] = rendering.get(
        "add_generation_prompt", impl.get("add_generation_prompt", True))
    # Agents have historically used either namespace. Both must agree when
    # supplied, rather than allowing an alternate backend argument to override.
    sampling = dict(impl.get("sampling_extra") or {})
    for key, value in (impl.get("sampling_params") or {}).items():
        if key in sampling and sampling[key] != value:
            raise ValueError(f"conflicting sampling settings: {key}")
        sampling[key] = value
    for key in ("add_generation_prompt",):
        if key in rendering and key in impl and rendering[key] != impl[key]:
            raise ValueError(f"conflicting rendering settings: {key}")
    # Keep every additional sampling control, including backend-specific ones.
    # Duplicated authoritative controls may not silently override measurement.
    authoritative = {**core["decoding"], "stop_token_ids": core["stop_token_ids"],
                     "stop": core["stop"], "max_tokens": config.measurement.max_new_tokens}
    for key, expected in authoritative.items():
        if key in sampling:
            if sampling.pop(key) != expected:
                raise ValueError(f"sampling settings override authoritative {key}")
    core["sampling"] = {
        key: sampling.get(key, default) for key, default in {
            "n": 1, "ignore_eos": False, "skip_special_tokens": True,
            "include_stop_str_in_output": False, "detokenize": True,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "min_tokens": 0,
        }.items()
    }
    core["sampling"].update(sampling)
    return core


def validate_inference_core(config: InferenceRunConfig, expected: dict[str, Any]) -> None:
    if not expected:
        return
    actual = inference_core(config)
    changed = sorted(key for key in actual.keys() | expected.keys()
                     if actual.get(key) != expected.get(key))
    if changed:
        raise ValueError("inference configuration differs from shared Validation core: "
                         + ", ".join(changed))
